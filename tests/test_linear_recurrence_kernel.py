"""
Unit tests for Linear Recurrent Attention C++ SIMD and Triton Kernels.
Verifies numerical parity between C++ single-step decode, PyTorch reference, and state progression.
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from turing.core.hybrid_attention import LinearRecurrentAttention


def test_linear_recurrence_cpp_parity():
    if not HAS_CSRC:
        pytest.skip("turing_csrc not available")

    torch.manual_seed(42)
    B, H, D = 2, 4, 64
    decay = 0.95

    q = torch.randn(B, H, D, dtype=torch.float32)
    k = torch.randn(B, H, D, dtype=torch.float32)
    v = torch.randn(B, H, D, dtype=torch.float32)
    state = torch.randn(B, H, D, D, dtype=torch.float32)

    # PyTorch reference
    q_t = q.unsqueeze(-1)  # [B, H, D, 1]
    k_t = k.unsqueeze(-2)  # [B, H, 1, D]
    v_t = v.unsqueeze(-1)  # [B, H, D, 1]

    ref_state = decay * state + torch.matmul(v_t, k_t)
    ref_out = torch.matmul(ref_state, q_t).squeeze(-1) # [B, H, D]

    # Native C++ AVX2
    q_np = q.numpy()
    k_np = k.numpy()
    v_np = v.numpy()
    s_np = state.numpy()

    cpp_out_np, cpp_state_np = turing_csrc.linear_recurrence_step_cpu(q_np, k_np, v_np, s_np, decay)

    np.testing.assert_allclose(cpp_out_np, ref_out.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cpp_state_np, ref_state.numpy(), rtol=1e-4, atol=1e-4)


def test_linear_recurrent_layer_forward():
    torch.manual_seed(42)
    layer = LinearRecurrentAttention(hidden_dim=256, num_heads=4, head_dim=64, decay=0.95)

    # Decode step (L=1)
    x_dec = torch.randn(2, 1, 256)
    out_dec, state_1 = layer(x_dec)
    assert out_dec.shape == (2, 1, 256)
    assert state_1.shape == (2, 4, 64, 64)

    # Prefill step (L=128)
    x_pref = torch.randn(2, 128, 256)
    out_pref, state_2 = layer(x_pref)
    assert out_pref.shape == (2, 128, 256)
    assert state_2.shape == (2, 4, 64, 64)
