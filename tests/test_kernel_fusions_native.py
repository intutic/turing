"""
Numerical Parity & Correctness Tests for Native C++20 SIMD & Triton/CUDA Kernel Fusions.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False


@pytest.mark.skipif(not HAS_CSRC, reason="Requires native C++ turing_csrc extension")
def test_matryoshka_quadtree_native_parity():
    hidden_dim = 1024
    vocab_size = 4096
    slice_width = 512

    torch.manual_seed(42)
    hidden = torch.randn(hidden_dim, dtype=torch.float32)
    weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
    spatial_proj = torch.randn(2, hidden_dim, dtype=torch.float32)

    h_np = hidden.numpy()
    w_np = weight.numpy()
    s_np = spatial_proj.numpy()

    tok_arr, parent_arr, mask_arr = turing_csrc.generate_matryoshka_quadtree(h_np, w_np, s_np, slice_width)

    assert len(tok_arr) == 21
    assert len(parent_arr) == 21
    assert mask_arr.shape == (21, 21)
    assert parent_arr[0] == -1
    for p in parent_arr[1:5]:
        assert p == 0
    for p in parent_arr[5:]:
        assert 1 <= p <= 4

    # Verify DAG mask causality: root attends only to itself
    assert mask_arr[0, 0] == 0.0
    for j in range(1, 21):
        assert mask_arr[0, j] == float("-inf")


@pytest.mark.skipif(not HAS_CSRC, reason="Requires native C++ turing_csrc extension")
def test_svd_int8_quant_native_parity():
    seq_len = 16
    head_dim = 128
    rank = 64

    torch.manual_seed(42)
    k_tensor = torch.randn(seq_len, head_dim, dtype=torch.float32)
    u_proj = torch.randn(head_dim, rank, dtype=torch.float32)

    k_np = k_tensor.numpy()
    u_np = u_proj.numpy()

    q_np, s_np = turing_csrc.fused_svd_int8_quant(k_np, u_np)
    recon_np = turing_csrc.fused_int8_dequant_svd_recon(q_np, s_np, u_np)

    # Reference PyTorch calculation
    k_sub_ref = torch.matmul(k_tensor, u_proj)
    scale_ref = torch.amax(torch.abs(k_sub_ref), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    q_ref = torch.clamp(torch.round(k_sub_ref / scale_ref), -128, 127).to(torch.int8)
    recon_ref = torch.matmul(q_ref.to(torch.float32) * scale_ref, u_proj.t())

    np.testing.assert_array_equal(q_np, q_ref.numpy())
    np.testing.assert_allclose(s_np, scale_ref.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(recon_np, recon_ref.numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not HAS_CSRC, reason="Requires native C++ turing_csrc extension")
def test_ridge_forward_native_parity():
    n_tokens = 8
    in_dim = 512
    out_features = 256

    torch.manual_seed(42)
    x = torch.randn(n_tokens, in_dim, dtype=torch.float32)
    w = torch.randn(in_dim, out_features, dtype=torch.float32)
    b = torch.randn(out_features, dtype=torch.float32)

    out_cpp = turing_csrc.fused_ridge_forward(x.numpy(), w.numpy(), b.numpy())
    out_ref = (torch.matmul(x, w) + b).numpy()

    np.testing.assert_allclose(out_cpp, out_ref, rtol=1e-4, atol=1e-4)


def test_triton_matryoshka_fallback():
    from turing.kernels.triton_matryoshka_spec import matryoshka_sliced_gemv_triton
    x = torch.randn(2, 4, 1024, dtype=torch.float32)
    w = torch.randn(512, 1024, dtype=torch.float32)

    out = matryoshka_sliced_gemv_triton(x, w, slice_width=256)
    out_ref = torch.matmul(x[..., :256], w[:, :256].t())
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)


def test_triton_svd_paged_fallback():
    from turing.kernels.triton_svd_paged import fused_svd_int8_quant_cuda, fused_int8_dequant_svd_recon_cuda
    k = torch.randn(8, 128, dtype=torch.float32)
    u_proj = torch.randn(128, 64, dtype=torch.float32)

    q_int8, scale = fused_svd_int8_quant_cuda(k, u_proj)
    recon = fused_int8_dequant_svd_recon_cuda(q_int8, scale, u_proj)

    k_sub = torch.matmul(k, u_proj)
    scale_ref = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
    q_ref = torch.clamp(torch.round(k_sub / scale_ref), -128, 127).to(torch.int8)
    recon_ref = torch.matmul(q_ref.to(torch.float32) * scale_ref, u_proj.t())

    torch.testing.assert_close(q_int8, q_ref)
    torch.testing.assert_close(scale, scale_ref)
    torch.testing.assert_close(recon, recon_ref, rtol=1e-4, atol=1e-4)


def test_triton_cross_kv_fallback():
    from turing.kernels.triton_cross_kv import fused_inv_rope_cuda
    from turing.core.cross_model_kv import RoPEContentDecoupler
    k = torch.randn(2, 16, 4, 64, dtype=torch.float32)

    k_unrot = fused_inv_rope_cuda(k, base=500000.0)
    k_unrot_ref = RoPEContentDecoupler.strip_rope(k, base=500000.0)

    torch.testing.assert_close(k_unrot, k_unrot_ref, rtol=1e-4, atol=1e-4)


def test_triton_chunk_compression_fallback():
    from turing.kernels.triton_chunk_compression import hca_chunk_pool_cuda
    from turing.core.hierarchical_compression import HCAChunkCompressor
    k = torch.randn(1, 256, 4, 64, dtype=torch.float32)
    v = torch.randn(1, 256, 4, 64, dtype=torch.float32)

    k_out, v_out = hca_chunk_pool_cuda(k, v, chunk_size=128)
    assert k_out.shape == (1, 2, 4, 64)
    assert v_out.shape == (1, 2, 4, 64)


def test_triton_mhc_fuse_fallback():
    from turing.kernels.triton_mhc_fuse import mhc_stream_mix_cuda
    streams = torch.randn(1, 16, 4, 64, dtype=torch.float32)
    layer_up = torch.randn(1, 16, 64, dtype=torch.float32)
    res_map = torch.eye(4, dtype=torch.float32)
    post_w = torch.zeros(4, dtype=torch.float32)

    out = mhc_stream_mix_cuda(streams, layer_up, res_map, post_w)
    beta = torch.sigmoid(post_w).view(1, 1, 4, 1)
    out_ref = streams + beta * layer_up.unsqueeze(-2)

    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)


def test_triton_cca_fallback():
    from turing.kernels.triton_cca import fused_linear_conv1d_causal_cuda
    x = torch.randn(2, 16, 256, dtype=torch.float32)
    w_lin = torch.randn(64, 256, dtype=torch.float32)
    w_conv = torch.randn(64, 1, 3, dtype=torch.float32)

    out = fused_linear_conv1d_causal_cuda(x, w_lin, w_conv)
    assert out.shape == (2, 16, 64)
