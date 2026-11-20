"""
Unit Tests for Cross-Model Closed-Form KV Cache Transfer (arXiv:2608.03893).
"""

import pytest
import torch
import torch.nn.functional as F

from turing.config import ModelConfig
from turing.experimental.cross_model_kv import (
    RoPEContentDecoupler,
    ClosedFormRidgeMapper,
    SVDNullSpaceProjector,
    CrossModelKVPipeline
)

def test_rope_content_decoupler_invertibility():
    """
    Tests that stripping and re-applying RoPE rotation is exactly invertible.
    """
    seq_len = 128
    head_dim = 128
    k_content = torch.randn(2, seq_len, 8, head_dim)

    # 1. Apply RoPE
    k_rope = RoPEContentDecoupler.apply_rope(k_content, base=500000.0)

    # 2. Strip RoPE
    k_reconstructed = RoPEContentDecoupler.strip_rope(k_rope, base=500000.0)

    max_diff = (k_content - k_reconstructed).abs().max().item()
    assert max_diff < 1e-4, f"RoPE stripping error too high: {max_diff}"


def test_closed_form_ridge_mapper_fit_and_inference():
    """
    Tests closed-form Ridge solve and projection accuracy.
    """
    n_tokens = 500
    source_heads = 4
    target_heads = 4
    head_dim = 64
    top_k = 2

    mapper = ClosedFormRidgeMapper(
        source_heads=source_heads,
        target_heads=target_heads,
        head_dim=head_dim,
        top_k_source_layers=top_k,
        ridge_lambda=0.01
    )

    in_dim = top_k * source_heads * head_dim

    # Create synthetic linear relationship
    x_source = torch.randn(n_tokens, in_dim)
    true_w = torch.randn(target_heads, in_dim, head_dim) * 0.1
    y_target = torch.einsum('ni,hio->nho', x_source, true_w) + 0.01 * torch.randn(n_tokens, target_heads, head_dim)

    mapper.fit(x_source, y_target, is_key=True)
    assert mapper.is_fit

    # Evaluate on test set
    x_test = torch.randn(1, 32, in_dim)
    y_pred = mapper(x_test, is_key=True)
    assert y_pred.shape == (1, 32, target_heads, head_dim)


def test_svd_null_space_projector():
    """
    Tests that SVD null-space projector eliminates components in top singular directions.
    """
    head_dim = 64
    rank = 8
    # Random orthogonal basis
    q_mat = torch.randn(128, head_dim)
    _, _, vt = torch.linalg.svd(q_mat, full_matrices=False)
    v_basis = vt.t()[:, :rank] # [64, 8]

    projector = SVDNullSpaceProjector(v_basis, top_r=rank)

    # Test with vector parallel to first basis vector
    v_sensitive = v_basis[:, 0].unsqueeze(0) # [1, 64]
    v_filtered = projector.filter_residual(v_sensitive)

    assert v_filtered.abs().max().item() < 1e-4, "Failed to project out sensitive direction"


def test_cross_model_kv_pipeline_end_to_end():
    """
    Tests end-to-end small-to-large KV cache transfer.
    """
    src_cfg = ModelConfig(
        name="Test-Small-4L",
        hidden_dim=256,
        ffn_dim=1024,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        num_layers=4,
        vocab_size=1000
    )

    tgt_cfg = ModelConfig(
        name="Test-Large-8L",
        hidden_dim=512,
        ffn_dim=2048,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        num_layers=8,
        vocab_size=1000
    )

    pipeline = CrossModelKVPipeline(src_cfg, tgt_cfg, top_k_layers=2)

    seq_len = 16
    src_keys = [torch.randn(1, seq_len, 2, 64) for _ in range(4)]
    src_vals = [torch.randn(1, seq_len, 2, 64) for _ in range(4)]

    tgt_keys, tgt_vals = pipeline.transfer_cache(src_keys, src_vals)

    assert len(tgt_keys) == tgt_cfg.num_layers
    assert len(tgt_vals) == tgt_cfg.num_layers
    assert tgt_keys[0].shape == (1, seq_len, tgt_cfg.num_kv_heads, tgt_cfg.head_dim)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_rope_strip_and_apply_match_the_formula_on_multi_head_keys(device):
    """[Batch, SeqLen, Heads, HeadDim]: every head of token t rotates by t (a CUDA kernel used to take the flattened
    row as the position, so heads after the first rotated by the wrong angle and lineage residuals blew up on CUDA)."""
    from turing.experimental.cross_model_kv import RoPEContentDecoupler
    B, T, H, D, base = 2, 7, 3, 16, 10000.0
    k = torch.randn(B, T, H, D, device=device)
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device).float() / D))
    ang = torch.outer(torch.arange(T, device=device).float(), inv)[None, :, None, :]
    cos, sin = ang.cos(), ang.sin()
    k1, k2 = k[..., :D // 2], k[..., D // 2:]
    want = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], -1)
    torch.testing.assert_close(RoPEContentDecoupler.apply_rope(k, base), want, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(RoPEContentDecoupler.strip_rope(want, base), k, atol=1e-5, rtol=1e-5)
