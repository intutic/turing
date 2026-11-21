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


