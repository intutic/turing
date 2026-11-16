"""HeterogeneousMoERunner on real converted MoE weights: hot experts in device cache slots, cold ones on the host,
output equal to the model's own SparseMoE."""

import pytest
import torch

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.expert_prefetch import OneLayerAheadExpertPrefetcher
from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider, HeterogeneousMoERunner, HostExpertBank
from turing.models.hf_loader import RealHuggingFaceLoader
from tests.test_hf_parity import _tiny

CPU = torch.device("cpu")


def _model(arch):
    return RealHuggingFaceLoader.convert_hf_model(_tiny(arch), sparsity_ratio=0.0, device="cpu",
                                                  enable_recirculation=False)


def _runner(model, num_slots, decider):
    bank = HostExpertBank.from_model(model)
    cache = GPULRUExpertCache(num_slots=num_slots, hidden_dim=bank.hidden_dim, active_subspace_dim=bank.ffn_dim,
                              device=CPU, eviction_policy="lru")
    return HeterogeneousMoERunner.from_model(model, cache, decider)


@pytest.mark.parametrize("arch", ["mixtral", "olmoe"])
@torch.no_grad()
def test_runner_matches_sparse_moe(arch):
    model = _model(arch)
    x = torch.randn(3, 5, model.config.hidden_dim, generator=torch.Generator().manual_seed(0))
    # device always cheaper: experts are copied into the 2 slots, evicting by LRU
    to_device = BandwidthAdaptiveDecider(CPU, pcie_bandwidth_gb_s=1e9, cpu_throughput_gflops=1.0,
                                         gpu_throughput_gflops=1e9)
    # host always cheaper: every miss runs on the CPU
    to_host = BandwidthAdaptiveDecider(CPU, pcie_bandwidth_gb_s=1e-9, cpu_throughput_gflops=1e9,
                                       gpu_throughput_gflops=1.0)
    for i, layer in enumerate(model.layers):
        ref = layer.mlp(x)
        for decider in (to_device, to_host):
            runner = _runner(model, num_slots=2, decider=decider)
            out, stats = runner.route_and_execute(x, layer_idx=i)
            torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
            assert stats["experts"] == stats["cache_hits"] + stats["loaded_to_device"] + stats["host_executed"]
        out, stats = runner.route_and_execute(x, layer_idx=i)  # host-only runner never caches
        assert stats["host_executed"] == stats["experts"] and stats["cache_hits"] == 0


@torch.no_grad()
def test_hot_experts_hit_the_cache_on_the_second_call():
    model = _model("mixtral")
    E = model.config.num_experts
    decider = BandwidthAdaptiveDecider(CPU, pcie_bandwidth_gb_s=1e9, cpu_throughput_gflops=1.0,
                                       gpu_throughput_gflops=1e9)
    runner = _runner(model, num_slots=E, decider=decider)  # room for every expert
    x = torch.randn(1, 8, model.config.hidden_dim, generator=torch.Generator().manual_seed(1))
    first, s1 = runner.route_and_execute(x, layer_idx=0)
    second, s2 = runner.route_and_execute(x, layer_idx=0)
    assert s1["loaded_to_device"] == s1["experts"] and s2["cache_hits"] == s2["experts"]
    torch.testing.assert_close(first, second, atol=0, rtol=0)


@torch.no_grad()
def test_prefetcher_loads_real_weights_from_the_bank():
    model = _model("mixtral")
    bank = HostExpertBank.from_model(model)
    cache = GPULRUExpertCache(num_slots=2, hidden_dim=bank.hidden_dim, active_subspace_dim=bank.ffn_dim, device=CPU)
    OneLayerAheadExpertPrefetcher(cache, host_bank=bank, prefetch_k=1, device=CPU).prefetch_layer_experts_async(1, [3])
    slot = cache.get_slot(1, 3)
    gate, up, down = bank.slot_weights(1, 3)
    assert torch.equal(cache.slot_gate_weights[slot], gate) and torch.equal(cache.slot_down_weights[slot], down)


@torch.no_grad()
def test_attached_runner_reproduces_the_whole_model():
    model = _model("olmoe")
    ids = torch.randint(0, model.config.vocab_size, (1, 12), generator=torch.Generator().manual_seed(2))
    ref_prefill, past = model(ids[:, :8])
    ref_steps = [model(ids[:, t:t + 1], past_key_values=past)[0] for t in range(8, 12)]
    decider = BandwidthAdaptiveDecider(CPU, pcie_bandwidth_gb_s=1e9, cpu_throughput_gflops=1.0,
                                       gpu_throughput_gflops=1e9)
    runner = _runner(model, num_slots=12, decider=decider)  # 16 (layer, expert) keys: some evicted
    old = runner.attach(model)
    assert sorted(old) == list(range(model.config.num_layers)) and not any(
        "experts" in n or "gate_up" in n for n, _ in model.named_parameters())
    prefill, past = model(ids[:, :8])
    torch.testing.assert_close(prefill, ref_prefill, atol=1e-5, rtol=0)
    for t, ref in zip(range(8, 12), ref_steps):
        torch.testing.assert_close(model(ids[:, t:t + 1], past_key_values=past)[0], ref, atol=1e-5, rtol=0)
    tot = runner.totals
    assert tot["experts"] == tot["cache_hits"] + tot["loaded_to_device"] + tot["host_executed"] > 0
    assert tot["cache_hits"] > 0 and tot["loaded_to_device"] > 0


def test_bank_rejects_dense_models():
    with pytest.raises(ValueError):
        HostExpertBank.from_model(_model("llama"))


def test_decider_recalibration_measures_a_positive_bandwidth():
    decider = BandwidthAdaptiveDecider(CPU, pcie_bandwidth_gb_s=25.0, cpu_throughput_gflops=100.0,
                                       gpu_throughput_gflops=1000.0)
    assert decider.recalibrate_live_bus(ema_weight=0.2) > 0.0
