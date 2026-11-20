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
    from turing.experimental.kernels.triton_matryoshka_spec import matryoshka_sliced_gemv_triton
    x = torch.randn(2, 4, 1024, dtype=torch.float32)
    w = torch.randn(512, 1024, dtype=torch.float32)

    out = matryoshka_sliced_gemv_triton(x, w, slice_width=256)
    out_ref = torch.matmul(x[..., :256], w[:, :256].t())
    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)


def test_triton_chunk_compression_fallback():
    from turing.experimental.kernels.triton_chunk_compression import hca_chunk_pool_cuda
    from turing.experimental.hierarchical_compression import HCAChunkCompressor
    k = torch.randn(1, 256, 4, 64, dtype=torch.float32)
    v = torch.randn(1, 256, 4, 64, dtype=torch.float32)

    k_out, v_out = hca_chunk_pool_cuda(k, v, chunk_size=128)
    assert k_out.shape == (1, 2, 4, 64)
    assert v_out.shape == (1, 2, 4, 64)


