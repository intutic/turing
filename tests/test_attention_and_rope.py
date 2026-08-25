import pytest
import torch
from turing.core.attention_cache import AttentionPatternCache, ChunkedLongPrefillEngine
from turing.core.rope import NTKDynamicRoPEScaling
from turing.core.optimal_transport import sinkhorn_knopp_eviction, OptimalTransportEviction

def test_attention_pattern_cache():
    apc = AttentionPatternCache(block_size=16, local_window=32, global_anchors=8)
    mask = apc.build_hybrid_block_mask(seq_len=64, device=torch.device("cpu"))
    assert mask.shape == (64, 64)
    # Check that root anchor is visible
    assert torch.all(mask[:, :8])

    row = apc.append_decode_row(current_pos=40, device=torch.device("cpu"))
    assert row.shape == (41,)
    assert bool(row[0]) is True # Root anchor
    assert bool(row[40]) is True # Self token

def test_ntk_rope_scaling():
    rope = NTKDynamicRoPEScaling(dim=64, max_position_embeddings=512)
    # Within window
    cos_short, sin_short = rope.compute_freqs(256, device=torch.device("cpu"))
    assert cos_short.shape == (256, 64)

    # Beyond window (triggers dynamic NTK alpha scaling)
    cos_long, sin_long = rope.compute_freqs(1024, device=torch.device("cpu"))
    assert cos_long.shape == (1024, 64)

def test_sinkhorn_knopp_ot_eviction():
    q = torch.randn(8, 64)
    k = torch.randn(32, 64)
    budget = 16

    retained_idx, key_mass = sinkhorn_knopp_eviction(q, k, budget=budget, epsilon=0.05)
    assert len(retained_idx) == budget
    assert key_mass.shape == (32,)
    assert torch.all(retained_idx < 32)
