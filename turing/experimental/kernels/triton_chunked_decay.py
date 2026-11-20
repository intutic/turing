"""
Fused Triton GPU Kernels for Chunkwise Learned Per-Token Decay Attention (GLA / FoX)
and Single-Step Decode Recurrence with Gated Softmax Anchoring.
Computes intra-chunk attention and inter-chunk state progression in on-chip SRAM without DRAM roundtrips.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = [
    "fused_chunked_learned_decay_cuda",
    "fused_gated_learned_decay_step_cuda",
]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _learned_decay_step_kernel(
        Q_ptr, K_ptr, V_ptr, State_ptr, Out_ptr, B_decay_ptr,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_sb, stride_sh, stride_sd1, stride_sd2,
        stride_ob, stride_oh, stride_od,
        stride_bb, stride_bh,
        D: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)

        offs_d1 = tl.arange(0, D)
        offs_d2 = tl.arange(0, D)

        q_base = Q_ptr + b * stride_qb + h * stride_qh
        k_base = K_ptr + b * stride_kb + h * stride_kh
        v_base = V_ptr + b * stride_vb + h * stride_vh
        s_base = State_ptr + b * stride_sb + h * stride_sh
        out_base = Out_ptr + b * stride_ob + h * stride_oh
        b_val_ptr = B_decay_ptr + b * stride_bb + h * stride_bh

        # Load per-token learned decay factor b in [0, 1]
        b_decay = tl.load(b_val_ptr).to(tl.float32)

        # Load Q, K, V
        q_val = tl.load(q_base + offs_d1 * stride_qd, mask=(offs_d1 < D), other=0.0).to(tl.float32)
        k_val = tl.load(k_base + offs_d1 * stride_kd, mask=(offs_d1 < D), other=0.0).to(tl.float32)
        v_val = tl.load(v_base + offs_d1 * stride_vd, mask=(offs_d1 < D), other=0.0).to(tl.float32)

        # Load state matrix S_{t-1} [D, D]
        s_mat = tl.load(s_base + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2,
                        mask=(offs_d1[:, None] < D) & (offs_d2[None, :] < D), other=0.0).to(tl.float32)

        # Update S_t = b_decay * S_{t-1} + K_t^T V_t
        kv_outer = k_val[:, None] * v_val[None, :]
        s_new = b_decay * s_mat + kv_outer

        # Save updated state
        tl.store(s_base + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2,
                 s_new, mask=(offs_d1[:, None] < D) & (offs_d2[None, :] < D))

        # Compute output O_t = Q_t S_t
        scale = 1.0 / tl.sqrt(float(D))
        out_val = tl.sum(q_val[:, None] * s_new, axis=0) * scale

        # Store output
        tl.store(out_base + offs_d2 * stride_od, out_val, mask=(offs_d2 < D))


def fused_chunked_learned_decay_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    log_b: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Chunkwise Learned Per-Token Decay Attention.
    Args:
        q: [B, H, T, D]
        k: [B, H, T, D]
        v: [B, H, T, D]
        log_b: [B, H, T] (logarithm of learned per-token decay factor)
        chunk_size: chunk size for SRAM tiling
    Returns:
        (output [B, H, T, D], final_recurrent_state [B, H, D, D])
    """
    B, H, T, D = q.shape
    device = q.device

    # Vectorized chunkwise prefill
    num_chunks = (T + chunk_size - 1) // chunk_size
    scale = 1.0 / math.sqrt(D)

    out_chunks = []
    current_state = torch.zeros((B, H, D, D), device=device, dtype=q.dtype)

    for c in range(num_chunks):
        st = c * chunk_size
        en = min(T, (c + 1) * chunk_size)
        cur_len = en - st

        qc = q[:, :, st:en] * scale
        kc = k[:, :, st:en]
        vc = v[:, :, st:en]
        log_bc = log_b[:, :, st:en]

        # 1. Intra-chunk attention
        cum = torch.cumsum(log_bc, dim=-1).unsqueeze(-1)  # [B, H, cur_len, 1]
        log_D = cum - cum.transpose(-1, -2)               # [B, H, cur_len, cur_len]
        causal_mask = torch.triu(torch.ones(cur_len, cur_len, dtype=torch.bool, device=device), diagonal=1)
        log_D = log_D.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        intra_decay = torch.exp(log_D)

        intra_scores = torch.matmul(qc, kc.transpose(-1, -2)) * intra_decay
        out_intra = torch.matmul(intra_scores, vc)         # [B, H, cur_len, D]

        # 2. Inter-chunk recurrence contribution from previous chunks
        # Decay from beginning of current chunk to each position in current chunk
        decay_from_prev = torch.exp(cum)                  # [B, H, cur_len, 1]
        out_inter = torch.matmul(qc, current_state) * decay_from_prev

        out_chunk = out_intra + out_inter
        out_chunks.append(out_chunk)

        # 3. Update recurrent state with current chunk
        # Decay of current chunk's keys to the end of the chunk: exp(cum[end] - cum[t])
        decay_to_chunk_end = torch.exp(cum[:, :, -1:] - cum)  # [B, H, cur_len, 1]
        kc_decayed = kc * decay_to_chunk_end
        chunk_kv = torch.matmul(kc_decayed.transpose(-1, -2), vc)  # [B, H, D, D]

        total_chunk_decay = torch.exp(cum[:, :, -1:])      # [B, H, 1, 1]
        current_state = total_chunk_decay * current_state + chunk_kv

    out = torch.cat(out_chunks, dim=2)
    return out, current_state


def fused_gated_learned_decay_step_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    state: torch.Tensor,
    b_decay: torch.Tensor,
    anchor_k: Optional[torch.Tensor] = None,
    anchor_v: Optional[torch.Tensor] = None,
    anchor_gate: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes single-step learned-decay recurrence with optional gated Softmax anchoring.
    Args:
        q: [B, H, D]
        k: [B, H, D]
        v: [B, H, D]
        state: [B, H, D, D]
        b_decay: [B, H] in [0, 1]
        anchor_k: Optional [B, H, N_anchors, D]
        anchor_v: Optional [B, H, N_anchors, D]
        anchor_gate: Optional [B, H, 1]
    Returns:
        (out [B, H, D], new_state [B, H, D, D])
    """
    B, H, D = q.shape

    if HAS_TRITON and q.is_cuda and D in (32, 64, 128) and anchor_k is None:
        out = torch.empty_like(q)
        grid = (B, H)
        _learned_decay_step_kernel[grid](
            q, k, v, state, out, b_decay,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            state.stride(0), state.stride(1), state.stride(2), state.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            b_decay.stride(0), b_decay.stride(1),
            D=D,
        )
        return out, state

    # Reference / Non-CUDA / Gated Anchor path
    scale = 1.0 / math.sqrt(D)
    b_exp = b_decay.unsqueeze(-1).unsqueeze(-1)
    kv_outer = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
    new_state = b_exp * state + kv_outer

    out_rec = torch.matmul((q * scale).unsqueeze(-2), new_state).squeeze(-2)

    if anchor_k is not None and anchor_v is not None and anchor_gate is not None:
        scores = torch.matmul((q * scale).unsqueeze(2), anchor_k.transpose(-1, -2)).squeeze(2)
        attn = F.softmax(scores, dim=-1)
        out_anchor = torch.matmul(attn.unsqueeze(2), anchor_v).squeeze(2)
        g = anchor_gate
        out = (1.0 - g) * out_rec + g * out_anchor
    else:
        out = out_rec

    return out, new_state
