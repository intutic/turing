import pytest
import torch
from turing.experimental.attention_cache import AttentionPatternCache, ChunkedLongPrefillEngine
from turing.core.rope import RotaryEmbedding
from turing.experimental.optimal_transport import sinkhorn_knopp_eviction, OptimalTransportEviction

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

def test_dynamic_ntk_rope_scaling_only_beyond_the_trained_context():
    rope = RotaryEmbedding(dim=64, max_position_embeddings=512, params={"rope_type": "dynamic", "factor": 2.0})
    plain = RotaryEmbedding(dim=64, max_position_embeddings=512)
    cos_short, _ = rope.compute_freqs(256, device=torch.device("cpu"))
    assert torch.equal(cos_short, plain.compute_freqs(256, device=torch.device("cpu"))[0])
    cos_long, _ = rope.compute_freqs(1024, device=torch.device("cpu"))  # the base grows past 512 tokens
    assert cos_long.shape == (1024, 64)
    assert not torch.allclose(cos_long, plain.compute_freqs(1024, device=torch.device("cpu"))[0])
    # the device path (one position per row, as decode uses) gives the same rows
    cos_dev, _ = rope.compute_freqs_device(torch.tensor([10, 900]))
    torch.testing.assert_close(cos_dev[0], rope.compute_freqs(11, torch.device("cpu"), start=10)[0][0])
    torch.testing.assert_close(cos_dev[1], rope.compute_freqs(901, torch.device("cpu"), start=900)[0][0])

def test_sinkhorn_knopp_ot_eviction():
    q = torch.randn(8, 64)
    k = torch.randn(32, 64)
    budget = 16

    retained_idx, key_mass = sinkhorn_knopp_eviction(q, k, budget=budget, epsilon=0.05)
    assert len(retained_idx) == budget
    assert key_mass.shape == (32,)
    assert torch.all(retained_idx < 32)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
def test_mps_attention_is_immune_to_sdpa_scale_caching():
    """PyTorch 2.9's MPS SDPA reuses a cached graph per shape with the first call's scale; the model must not use it."""
    import torch.nn.functional as F
    from turing.config import ModelConfig
    from turing.models.causal_lm import SubspaceAttention
    cfg = ModelConfig(name="t", hidden_dim=64, ffn_dim=128, num_heads=4, num_kv_heads=4, head_dim=16, num_layers=1)
    attn = SubspaceAttention(cfg).to("mps")
    q, k, v = [torch.randn(1, 4, 24, 16, device="mps") * 3 for _ in range(3)]
    F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)  # another caller caches this shape at scale 1
    ref = F.scaled_dot_product_attention(q.cpu(), k.cpu(), v.cpu(), is_causal=True)
    torch.testing.assert_close(attn._attend(q, k, v, None, causal=True).cpu(), ref, atol=1e-4, rtol=0)
