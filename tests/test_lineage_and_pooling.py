import pytest
import torch

from turing.core.lineage import (
    CacheLineageEntry, CacheLineage, LineageDriftError,
    CleanBaseLineageBuffer, CleanBaseStrategy, AppendOnlyStrategy, NaiveStrategy
)
from turing.core.kslot_pooling import KSlotCachePooler, GatedZeroIdentityHead, GateSkipPolicy


def test_cache_lineage_sequential_ordering():
    """Verifies entries are strictly sequential."""
    lineage = CacheLineage()
    torch.manual_seed(42)
    kv = [torch.randn(1, 4, 2, 64)]
    lineage.record(0, 'clean_base', kv, kv)
    lineage.record(1, 'clean_base', kv, kv)
    with pytest.raises(LineageDriftError):
        lineage.record(3, 'clean_base', kv, kv)  # Skip turn 2
    assert len(lineage) == 2


def test_cache_lineage_hash_determinism():
    """Same KV tensors produce identical BLAKE2b hashes across runs."""
    lineage = CacheLineage()
    torch.manual_seed(42)
    kv = [torch.randn(1, 4, 2, 64)]
    entry1 = lineage.record(0, 'test', kv, kv)
    lineage2 = CacheLineage()
    entry2 = lineage2.record(0, 'test', kv, kv)
    assert entry1.read_hash == entry2.read_hash
    assert entry1.wrote_hash == entry2.wrote_hash


def test_cache_lineage_drift_detection():
    """Modified cache raises LineageDriftError."""
    lineage = CacheLineage()
    torch.manual_seed(42)
    kv = [torch.randn(1, 4, 2, 64)]
    lineage.record(0, 'test', kv, kv)
    modified = [kv[0] + 0.001]
    with pytest.raises(LineageDriftError):
        lineage.verify_read(0, modified)


def test_clean_base_strategy_returns_original():
    """Asserts clean_base always returns the frozen base."""
    strategy = CleanBaseStrategy()
    torch.manual_seed(42)
    original = [torch.randn(1, 4, 2, 64)]
    previous = [torch.randn(1, 4, 2, 64)]
    for turn in range(5):
        result = strategy.cache_for_turn(turn, original, previous)
        assert result is original
        assert strategy.translates_on_turn(turn) is True


def test_naive_strategy_returns_previous():
    """Asserts naive returns the mutated cache after turn 0."""
    strategy = NaiveStrategy()
    torch.manual_seed(42)
    original = [torch.randn(1, 4, 2, 64)]
    previous = [torch.randn(1, 4, 2, 64)]
    assert strategy.cache_for_turn(0, original, previous) is original
    assert strategy.cache_for_turn(1, original, previous) is previous
    assert strategy.translates_on_turn(3) is True


def test_append_only_translates_once():
    """AppendOnlyStrategy translates only on turn 0."""
    strategy = AppendOnlyStrategy()
    assert strategy.translates_on_turn(0) is True
    assert strategy.translates_on_turn(1) is False
    assert strategy.translates_on_turn(5) is False


def test_clean_base_buffer_isolation():
    """CleanBaseLineageBuffer prevents mutation of the original."""
    torch.manual_seed(42)
    original = [torch.randn(2, 4, 2, 64)]
    buffer = CleanBaseLineageBuffer(original)
    retrieved = buffer.get_clean_base()
    retrieved[0] += 100.0  # Mutate the retrieved copy
    clean = buffer.get_clean_base()
    assert torch.allclose(clean[0], original[0])  # Original is unchanged


def test_kslot_pooler_output_shapes():
    """Validates (B, L, H, k, D) output dimensions."""
    torch.manual_seed(42)
    pooler = KSlotCachePooler(num_layers=4, num_kv_heads=2, head_dim=64, num_slots=4)
    B, L, H, N, D = 2, 4, 2, 32, 64
    keys = torch.randn(B, L, H, N, D)
    values = torch.randn(B, L, H, N, D)
    pk, pv = pooler(keys, values)
    assert pk.shape == (B, L, H, 4, D)
    assert pv.shape == (B, L, H, 4, D)


def test_kslot_pooler_sequence_length_invariance():
    """Pooling different N values produces same output shape."""
    torch.manual_seed(42)
    pooler = KSlotCachePooler(num_layers=4, num_kv_heads=2, head_dim=64, num_slots=3)
    B, L, H, D = 1, 4, 2, 64
    for N in [16, 64, 256]:
        keys = torch.randn(B, L, H, N, D)
        values = torch.randn(B, L, H, N, D)
        pk, pv = pooler(keys, values)
        assert pk.shape == (B, L, H, 3, D)


def test_gated_zero_identity_invariant():
    """Untrained gate produces exact zero residual."""
    head = GatedZeroIdentityHead(latent_dim=128, num_kv_heads=4, head_dim=64)
    features = torch.randn(1, 32, 128)
    dk, dv = head(features)
    assert torch.allclose(dk, torch.zeros_like(dk), atol=1e-7)
    assert torch.allclose(dv, torch.zeros_like(dv), atol=1e-7)


def test_gate_skip_policy_threshold():
    """Low-norm residuals trigger skip."""
    policy = GateSkipPolicy(threshold=0.1)
    assert policy.should_skip(0.05) is True
    assert policy.should_skip(0.5) is False
    assert policy.should_skip(0.01) is True
    assert policy.skip_rate == pytest.approx(2.0 / 3.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_kslot_parity():
    """GPU kernel matches PyTorch reference."""
    from turing.kernels.triton_kslot_pool import fused_kslot_pool_cuda
    from turing.core.kslot_pooling import KSlotCachePooler
    
    torch.manual_seed(42)
    pooler = KSlotCachePooler(num_layers=4, num_kv_heads=2, head_dim=64, num_slots=4)
    B, L, H, N, D = 2, 4, 2, 128, 64
    keys = torch.randn(B, L, H, N, D, device='cuda')
    values = torch.randn(B, L, H, N, D, device='cuda')
    queries = pooler.queries.data.to('cuda')
    
    # PyTorch reference
    ref_k, ref_v = pooler.to('cuda')(keys, values)
    
    # Triton kernel
    tri_k, tri_v = fused_kslot_pool_cuda(keys, values, queries)
    
    assert torch.allclose(ref_k, tri_k, atol=1e-3, rtol=1e-3)
    assert torch.allclose(ref_v, tri_v, atol=1e-3, rtol=1e-3)
