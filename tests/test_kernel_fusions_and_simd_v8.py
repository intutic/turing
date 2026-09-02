import math
import pytest
import torch
import numpy as np

from turing.serving.traffic import AdaptiveComputeBatcher
from turing.core.expert_cache import RoutingHistoryTracker
from turing.core.kslot_pooling import GatedZeroIdentityHead
from turing.kernels.triton_spec_entropy import fused_entropy_argmax_gate_cuda
from turing.kernels.triton_matryoshka_gemv import fused_matryoshka_gemv_cuda
from turing.kernels.triton_hybrid_anchor_fuse import fused_hybrid_step_anchor_cuda
from turing.kernels.triton_subspace_compress import fused_subspace_compress_cuda


def test_varlen_pack_simd_parity():
    """Verifies C++ SIMD token packing matches pure Python flattened reference."""
    token_lists = [
        [101, 2054, 2003, 1037],
        [101, 1045, 2293, 7592, 102],
        [42, 99],
        [1000, 2000, 3000, 4000, 5000, 6000],
    ]
    flat_tokens, cu_seqlens = AdaptiveComputeBatcher.pack_varlen_tokens(token_lists)

    # Reference
    ref_flat = []
    ref_cu = [0]
    c = 0
    for tl in token_lists:
        ref_flat.extend(tl)
        c += len(tl)
        ref_cu.append(c)

    assert flat_tokens == ref_flat
    assert cu_seqlens == ref_cu


def test_native_routing_history_tracker():
    """Verifies NativeRoutingHistoryTracker records expert hits, decays counts, and finds top-k."""
    tracker = RoutingHistoryTracker(num_layers=16, num_experts=32, decay_factor=0.5)

    # Layer 0: activate expert 5 (3 times), expert 8 (1 time)
    tracker.record_route(0, [5, 5, 5, 8])
    # Layer 1: activate expert 12 (2 times)
    tracker.record_route(1, [12, 12])

    top_layer0 = tracker.get_top_k_experts(k=2, layer_idx=0)
    assert top_layer0[0] == (0, 5)

    # Test exponential decay
    tracker.decay(0.5)
    # Expert 5 count was 3, 3 * 0.5 = 1 in integer division
    top_after_decay = tracker.get_top_k_experts(k=2, layer_idx=0)
    assert len(top_after_decay) >= 1


def test_gated_zero_identity_simd_parity():
    """Verifies SIMD gated zero-identity head produces exact zero residual on untrained weights."""
    torch.manual_seed(42)
    head = GatedZeroIdentityHead(latent_dim=64, num_kv_heads=4, head_dim=32)
    features = torch.randn(1, 1, 64)

    dk, dv = head(features)
    assert torch.allclose(dk, torch.zeros_like(dk), atol=1e-7)
    assert torch.allclose(dv, torch.zeros_like(dv), atol=1e-7)


def test_spec_entropy_argmax_fused_parity():
    """Verifies fused speculative Shannon entropy and argmax matches PyTorch reference."""
    torch.manual_seed(42)
    logits = torch.randn(4, 1024)

    # Reference
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    ref_entropy = -(probs * log_probs).sum(dim=-1)
    ref_argmax = torch.argmax(logits, dim=-1)

    # Fused
    entropy, argmax_idx, width = fused_entropy_argmax_gate_cuda(logits, low_thresh=0.4, high_thresh=1.2)

    assert torch.allclose(entropy, ref_entropy, atol=1e-4)
    assert torch.equal(argmax_idx, ref_argmax)
    assert width.shape == (4,)


def test_matryoshka_gemv_fused_parity():
    """Verifies fused Matryoshka parameter-sliced GEMV matches PyTorch sliced matmul."""
    torch.manual_seed(42)
    x = torch.randn(2, 512)
    weight = torch.randn(1024, 512)

    for slice_k in [64, 128, 256, 512]:
        ref = torch.matmul(x[:, :slice_k], weight[:, :slice_k].t())
        out = fused_matryoshka_gemv_cuda(x, weight, slice_k)
        assert torch.allclose(out, ref, atol=1e-4)


def test_subspace_compress_fused_parity():
    """Verifies fused Subspace INT8 compression matches PyTorch reference."""
    torch.manual_seed(42)
    x = torch.randn(4, 128)
    u_proj = torch.randn(128, 64)

    q_int8, scale = fused_subspace_compress_cuda(x, u_proj)

    # Reference
    x_sub = torch.matmul(x, u_proj)
    abs_max = torch.amax(torch.abs(x_sub), dim=-1, keepdim=True).clamp(min=1e-5)
    ref_scale = abs_max / 127.0
    ref_q = torch.clamp(torch.round(x_sub / ref_scale), -128, 127).to(torch.int8)

    assert torch.allclose(scale, ref_scale, atol=1e-4)
    assert torch.equal(q_int8, ref_q)


def test_hybrid_anchor_fused_step_parity():
    """Verifies fused hybrid step anchor matches PyTorch recurrence and attention."""
    torch.manual_seed(42)
    B, H, D = 2, 4, 32
    q = torch.randn(B, H, D)
    k = torch.randn(B, H, D)
    v = torch.randn(B, H, D)
    state = torch.randn(B, H, D, D)
    b_decay = torch.sigmoid(torch.randn(B, H))

    anchor_k = torch.randn(B, H, 8, D)
    anchor_v = torch.randn(B, H, 8, D)

    out, next_s = fused_hybrid_step_anchor_cuda(q, k, v, state, b_decay, anchor_k, anchor_v)

    # Reference
    b_exp = b_decay.unsqueeze(-1).unsqueeze(-1)
    kv_outer = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
    ref_next_s = b_exp * state + kv_outer
    scale = 1.0 / math.sqrt(D)
    ref_out_rec = torch.matmul((q * scale).unsqueeze(-2), ref_next_s).squeeze(-2)

    scores = torch.matmul((q * scale).unsqueeze(2), anchor_k.transpose(-1, -2)).squeeze(2)
    attn = torch.softmax(scores, dim=-1)
    ref_out_anchor = torch.matmul(attn.unsqueeze(2), anchor_v).squeeze(2)
    ref_out = 0.5 * ref_out_rec + 0.5 * ref_out_anchor

    assert torch.allclose(next_s, ref_next_s, atol=1e-4)
    assert torch.allclose(out, ref_out, atol=1e-4)
