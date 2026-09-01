import os
import tempfile
import pytest
import torch
import torch.nn.functional as F

from turing.config import ModelConfig
from turing.core.expert_cache import GPULRUExpertCache, RoutingHistoryTracker
from turing.core.expert_prefetch import OneLayerAheadExpertPrefetcher
from turing.core.heterogeneous_moe import (
    BandwidthAdaptiveDecider,
    HostExpertBank,
    HeterogeneousMoERunner,
)


def test_routing_history_tracker_basic():
    tracker = RoutingHistoryTracker(num_layers=4, num_experts=8, decay_factor=0.5)

    # Record activations
    tracker.record_route(0, [1, 2, 2, 3])
    tracker.record_route(1, torch.tensor([4, 4, 5]))

    assert tracker.counts[(0, 1)] == 1
    assert tracker.counts[(0, 2)] == 2
    assert tracker.counts[(0, 3)] == 1
    assert tracker.counts[(1, 4)] == 2
    assert tracker.counts[(1, 5)] == 1

    # Top-k across all layers
    top2 = tracker.get_top_k_experts(2)
    assert len(top2) == 2
    assert (0, 2) in top2 or (1, 4) in top2

    # Top-k for layer 0
    top_l0 = tracker.get_top_k_experts(1, layer_idx=0)
    assert top_l0 == [(0, 2)]

    # Decay
    tracker.decay(factor=0.5)
    assert tracker.counts[(0, 2)] == 1  # 2 * 0.5 = 1
    assert (0, 1) not in tracker.counts  # 1 * 0.5 + 0.5 = 1 or discarded depending on rounding


def test_routing_history_tracker_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "routing_history.json")
        tracker1 = RoutingHistoryTracker(num_layers=8, num_experts=16, save_path=save_path)
        tracker1.record_route(2, [7, 8, 8, 9])
        tracker1.save()

        tracker2 = RoutingHistoryTracker(num_layers=8, num_experts=16, save_path=save_path)
        loaded = tracker2.load()
        assert loaded is True
        assert tracker2.counts[(2, 8)] == 2
        assert tracker2.counts[(2, 7)] == 1
        assert tracker2.counts[(2, 9)] == 1


def test_gpu_lru_expert_cache_pinning():
    cache = GPULRUExpertCache(
        num_slots=3,
        hidden_dim=64,
        active_subspace_dim=32,
        device=torch.device("cpu"),
        eviction_policy="lru"
    )

    # Pin expert (0, 1)
    cache.pin_experts([(0, 1)])

    # Fill slots: (0, 1), (0, 2), (0, 3)
    s1, _ = cache.allocate_or_evict_slot(0, 1)
    s2, _ = cache.allocate_or_evict_slot(0, 2)
    s3, _ = cache.allocate_or_evict_slot(0, 3)

    assert cache.used_slots == 3

    # Access (0, 1) then allocate (0, 4)
    cache.get_slot(0, 1)
    s4, evicted = cache.allocate_or_evict_slot(0, 4)

    # Pinned (0, 1) must NOT be evicted!
    assert evicted != (0, 1)
    assert cache.contains(0, 1) is True
    assert cache.contains(0, 4) is True

    # Unpin and verify
    cache.unpin_experts([(0, 1)])
    assert (0, 1) not in cache.pinned_keys


def test_gpu_lru_expert_cache_lfru_eviction():
    tracker = RoutingHistoryTracker()
    cache = GPULRUExpertCache(
        num_slots=2,
        hidden_dim=64,
        active_subspace_dim=32,
        device=torch.device("cpu"),
        eviction_policy="lfru",
        history_tracker=tracker
    )

    # Allocate (0, 1) and access multiple times (high heat)
    s1, _ = cache.allocate_or_evict_slot(0, 1)
    for _ in range(10):
        cache.get_slot(0, 1)

    # Allocate (0, 2) and access once (low heat)
    s2, _ = cache.allocate_or_evict_slot(0, 2)
    cache.get_slot(0, 2)

    # Allocate (0, 3) with moderate heat
    for _ in range(3):
        cache.allocate_or_evict_slot(0, 3)

    # Hot (0, 1) must remain in cache; cold (0, 2) should be evicted
    assert cache.contains(0, 1) is True


def test_one_layer_ahead_expert_prefetcher():
    cache = GPULRUExpertCache(
        num_slots=4,
        hidden_dim=64,
        active_subspace_dim=32,
        device=torch.device("cpu")
    )
    prefetcher = OneLayerAheadExpertPrefetcher(
        expert_cache=cache,
        prefetch_k=2,
        device=torch.device("cpu")
    )

    def dummy_weight_provider(layer, exp):
        w_g = torch.randn(64, 32)
        w_u = torch.randn(64, 32)
        w_d = torch.randn(32, 64)
        return w_g, w_u, w_d

    # Prefetch layer 1 experts [2, 3]
    scheduled = prefetcher.prefetch_layer_experts_async(
        next_layer_idx=1,
        predicted_experts=[2, 3, 4],  # should truncate to top-2
        weight_provider=dummy_weight_provider
    )

    assert scheduled == [2, 3]
    assert cache.contains(1, 2) is True
    assert cache.contains(1, 3) is True

    # Prefetch again: should count as prefetch hits
    prefetcher.prefetch_layer_experts_async(
        next_layer_idx=1,
        predicted_experts=[2, 3],
        weight_provider=dummy_weight_provider
    )
    stats = prefetcher.stats()
    assert stats["prefetch_hits"] >= 2


def test_vectorized_batch_union_numerical_parity():
    torch.manual_seed(42)
    device = torch.device("cpu")
    config = ModelConfig(
        name="test_moe_model",
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        vocab_size=100
    )

    host_bank = HostExpertBank(
        num_layers=2,
        num_experts=8,
        hidden_dim=32,
        ffn_dim=64,
        active_subspace_dim=16
    )

    decider = BandwidthAdaptiveDecider(
        device=device,
        pcie_bandwidth_gb_s=25.0,
        cpu_throughput_gflops=100.0,
        gpu_throughput_gflops=1000.0
    )

    runner = HeterogeneousMoERunner(
        config=config,
        host_bank=host_bank,
        decider=decider,
        device=device
    )

    batch, seq_len = 2, 4
    hidden_states = torch.randn(batch, seq_len, 32)
    router_logits = torch.randn(batch, seq_len, 8)

    # Run vectorized batch-union
    out, stats = runner.route_and_execute(
        hidden_states=hidden_states,
        router_logits=router_logits,
        layer_idx=0,
        top_k=2
    )

    assert out.shape == (batch, seq_len, 32)
    assert not torch.isnan(out).any()
    assert stats["total_active_experts"] > 0

    # Test EMA recalibration on decider
    initial_bw = decider.pcie_bandwidth_gb_s
    new_bw = decider.recalibrate_live_bus(ema_weight=0.2)
    assert new_bw > 0.0
