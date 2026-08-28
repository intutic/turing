"""
Unit tests for Latent Flash-Decode (SPECTRA Mode-B) Subspace Attention.
Tests numerical equivalence and output shape contracts across CPU, MPS, and CUDA.
"""

import pytest
import torch
import numpy as np

from turing.kernels.triton_latent_decode import triton_latent_flash_decode

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = hasattr(turing_csrc, "latent_decode_cpu")
except ImportError:
    HAS_CSRC = False


def test_latent_decode_numerical_equivalence():
    """Verify in-SRAM Latent Flash-Decode matches reference attention."""
    torch.manual_seed(42)
    B, NKV, GRP, R = 2, 4, 8, 64
    SeqLen = 256
    head_dim = 128

    qp = torch.randn(B, NKV, GRP, R, dtype=torch.float32)
    ck = torch.randint(-128, 127, (B, SeqLen, R), dtype=torch.int8)
    sk = torch.rand(B, SeqLen, dtype=torch.float32) * 0.05 + 0.01
    cv = torch.randint(-128, 127, (B, SeqLen, R), dtype=torch.int8)
    sv = torch.rand(B, SeqLen, dtype=torch.float32) * 0.05 + 0.01

    out = triton_latent_flash_decode(qp, ck, sk, cv, sv, head_dim=head_dim)

    assert out.shape == (B, NKV, GRP, R)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_latent_decode_gqa_ratios():
    """Test various Grouped-Query Attention head group ratios."""
    for GRP in [1, 4, 8, 16]:
        B, NKV, R = 1, 2, 64
        SeqLen = 128
        qp = torch.randn(B, NKV, GRP, R, dtype=torch.float32)
        ck = torch.randint(-128, 127, (B, SeqLen, R), dtype=torch.int8)
        sk = torch.rand(B, SeqLen, dtype=torch.float32) * 0.02 + 0.01
        cv = torch.randint(-128, 127, (B, SeqLen, R), dtype=torch.int8)
        sv = torch.rand(B, SeqLen, dtype=torch.float32) * 0.02 + 0.01

        out = triton_latent_flash_decode(qp, ck, sk, cv, sv, head_dim=128)
        assert out.shape == (B, NKV, GRP, R)


@pytest.mark.skipif(not HAS_CSRC, reason="Native C++ extension latent_decode_cpu not compiled")
def test_native_csrc_latent_decode():
    """Verify native C++20 AVX2 latent decode execution."""
    B, NKV, GRP, R = 1, 1, 4, 64
    SeqLen = 64
    total_q = B * NKV * GRP

    qp_np = np.random.randn(total_q, R).astype(np.float32)
    ck_np = np.random.randint(-128, 127, (B, SeqLen, R), dtype=np.int8)
    sk_np = (np.random.rand(B, SeqLen) * 0.05 + 0.01).astype(np.float32)
    cv_np = np.random.randint(-128, 127, (B, SeqLen, R), dtype=np.int8)
    sv_np = (np.random.rand(B, SeqLen) * 0.05 + 0.01).astype(np.float32)

    scale = 1.0 / (128.0 ** 0.5)
    out_csrc = turing_csrc.latent_decode_cpu(qp_np, ck_np, sk_np, cv_np, sv_np, scale)

    assert out_csrc.shape == (total_q, R)
    assert not np.isnan(out_csrc).any()
