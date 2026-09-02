import math
import pytest
import torch
import numpy as np

from turing.core.hybrid_memory import (
    LearnedDecayRecurrenceState,
    HybridRecurrentAttention,
    AssociativeRecallProbe,
)
from turing.kernels.triton_chunked_decay import (
    fused_chunked_learned_decay_cuda,
    fused_gated_learned_decay_step_cuda,
)


def test_learned_decay_step_recurrence_numerical_parity():
    """Verify that closed-form pairwise decay matches step-by-step state recurrence."""
    torch.manual_seed(42)
    B, H, T, D = 2, 4, 16, 32

    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    # Learned decay factors in [0.7, 0.95]
    b = torch.sigmoid(torch.randn(B, H, T) + 1.5)
    log_b = torch.log(b + 1e-8)

    # 1. Closed-form pairwise calculation
    scale = 1.0 / math.sqrt(D)
    qc = q * scale
    cum = torch.cumsum(log_b, dim=-1).unsqueeze(-1)
    log_D = cum - cum.transpose(-1, -2)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    log_D = log_D.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    D_weights = torch.exp(log_D)

    scores = torch.matmul(qc, k.transpose(-1, -2)) * D_weights
    out_closed = torch.matmul(scores, v)

    # 2. Step-by-step recurrence calculation
    out_steps = []
    state = torch.zeros(B, H, D, D)

    for t in range(T):
        qt = qc[:, :, t]
        kt = k[:, :, t]
        vt = v[:, :, t]
        bt = b[:, :, t].unsqueeze(-1).unsqueeze(-1)

        # S_t = b_t S_{t-1} + K_t^T V_t
        kv_outer = torch.matmul(kt.unsqueeze(-1), vt.unsqueeze(-2))
        state = bt * state + kv_outer

        # O_t = Q_t S_t
        ot = torch.matmul(qt.unsqueeze(-2), state).squeeze(-2)
        out_steps.append(ot)

    out_recurrent = torch.stack(out_steps, dim=2)

    assert torch.allclose(out_closed, out_recurrent, atol=1e-5, rtol=1e-5)


def test_native_learned_decay_step_simd_parity():
    """Verify C++20 AVX2 SIMD learned_decay_step_fast matches PyTorch calculation."""
    try:
        import turing.turing_csrc as turing_csrc
    except ImportError:
        pytest.skip("turing_csrc extension not built")

    np.random.seed(42)
    D = 64
    q = np.random.randn(D).astype(np.float32)
    k = np.random.randn(D).astype(np.float32)
    v = np.random.randn(D).astype(np.float32)
    state = np.random.randn(D, D).astype(np.float32)
    b_decay = 0.88

    # Reference calculation in PyTorch
    q_t = torch.from_numpy(q)
    k_t = torch.from_numpy(k)
    v_t = torch.from_numpy(v)
    state_ref = torch.from_numpy(state.copy())

    state_new_ref = b_decay * state_ref + torch.outer(k_t, v_t)
    out_ref = (torch.matmul(q_t, state_new_ref) / math.sqrt(D)).numpy()

    # C++ SIMD execution (modifies state in-place)
    state_simd = state.copy()
    out_simd = turing_csrc.learned_decay_step_fast(q, k, v, state_simd, b_decay)

    np.testing.assert_allclose(state_simd, state_new_ref.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_simd, out_ref, rtol=1e-4, atol=1e-4)


def test_chunked_learned_decay_parity():
    """Verify chunked in-SRAM learned decay matches closed-form prefill."""
    torch.manual_seed(42)
    B, H, T, D = 1, 2, 64, 32
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    b = torch.sigmoid(torch.randn(B, H, T) + 1.2)
    log_b = torch.log(b + 1e-8)

    # Chunked prefill with chunk_size=16
    out_chunked, state_chunked = fused_chunked_learned_decay_cuda(q, k, v, log_b, chunk_size=16)

    # Closed-form reference
    scale = 1.0 / math.sqrt(D)
    qc = q * scale
    cum = torch.cumsum(log_b, dim=-1).unsqueeze(-1)
    log_D = cum - cum.transpose(-1, -2)
    causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    log_D = log_D.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
    D_weights = torch.exp(log_D)

    scores = torch.matmul(qc, k.transpose(-1, -2)) * D_weights
    out_ref = torch.matmul(scores, v)

    assert torch.allclose(out_chunked, out_ref, atol=1e-4, rtol=1e-4)


def test_hybrid_recurrent_attention_prefill_and_step_decode():
    """Verify HybridRecurrentAttention prefill and step-by-step decode execution."""
    torch.manual_seed(42)
    embed_dim = 64
    num_heads = 2
    model = HybridRecurrentAttention(embed_dim=embed_dim, num_heads=num_heads)

    x_prefill = torch.randn(1, 8, embed_dim)
    anchor_mask = torch.tensor([[True, False, False, True, False, False, False, False]])

    out_prefill, final_state = model.forward_prefill_closed_form(x_prefill, anchor_mask=anchor_mask)
    assert out_prefill.shape == (1, 8, embed_dim)
    assert final_state.step_count == 8

    # Step decode with next token
    x_next = torch.randn(1, 1, embed_dim)
    out_next, next_state = model.step_decode(x_next, final_state, is_anchor=False)
    assert out_next.shape == (1, 1, embed_dim)
    assert next_state.step_count == 9


def test_associative_recall_probe_generation():
    """Verify associative recall probe instance formatting."""
    probe = AssociativeRecallProbe(key_pool_size=16, val_pool_size=64, seq_len=192, seed=42)
    xs, ys, slots = probe.generate_batch(load_r=4, batch_size=8)

    assert xs.shape == (8, 192)
    assert ys.shape == (8,)
    assert slots.shape == (8,)

    # Marker token at seq[-1]
    assert (xs[:, -1] == probe.marker_token).all()
    # Query key is at seq[-2]
    assert (xs[:, -2] < probe.key_pool_size).all()
    # Target value is within value range
    assert (ys >= probe.key_pool_size).all()
    assert (ys < probe.key_pool_size + probe.val_pool_size).all()
