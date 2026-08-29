"""
Triton Chunk-Parallel Linear Recurrent Attention Kernel.
Implements fused in-SRAM intra-chunk attention and inter-chunk state recurrence
(GLM-5.3-Flash / Qwen3.8-Flash-Next 3:1 linear attention layers).
"""

import math
from typing import Optional, Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _chunk_linear_recurrence_kernel(
        Q_ptr,          # [B, H, L, D]
        K_ptr,          # [B, H, L, D]
        V_ptr,          # [B, H, L, D]
        Out_ptr,        # [B, H, L, D]
        State_in_ptr,   # [B, H, D, D] (optional)
        State_out_ptr,  # [B, H, D, D]
        decay,
        L,
        stride_qb, stride_qh, stride_ql, stride_qd,
        stride_sb, stride_sh, stride_sd1, stride_sd2,
        D: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        HAS_STATE: tl.constexpr
    ):
        bh = tl.program_id(0)
        b = bh // stride_qh
        h = bh % stride_qh

        offs_d1 = tl.arange(0, D)
        offs_d2 = tl.arange(0, D)
        offs_c = tl.arange(0, CHUNK_SIZE)

        # Initialize recurrent state S in SRAM [D, D]
        s_mat = tl.zeros([D, D], dtype=tl.float32)
        if HAS_STATE:
            s_mat = tl.load(State_in_ptr + b * stride_sb + h * stride_sh + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2)

        num_chunks = tl.cdiv(L, CHUNK_SIZE)
        chunk_decay = tl.math.exp(CHUNK_SIZE * tl.math.log(decay))

        for c in range(num_chunks):
            start_l = c * CHUNK_SIZE
            offs_l = start_l + offs_c
            mask_l = offs_l < L

            # Load Q_c, K_c, V_c [CHUNK_SIZE, D]
            q_base = Q_ptr + b * stride_qb + h * stride_qh
            k_base = K_ptr + b * stride_qb + h * stride_qh
            v_base = V_ptr + b * stride_qb + h * stride_qh
            out_base = Out_ptr + b * stride_qb + h * stride_qh

            qc = tl.load(q_base + offs_l[:, None] * stride_ql + offs_d1[None, :] * stride_qd, mask=mask_l[:, None], other=0.0).to(tl.float32)
            kc = tl.load(k_base + offs_l[:, None] * stride_ql + offs_d1[None, :] * stride_qd, mask=mask_l[:, None], other=0.0).to(tl.float32)
            vc = tl.load(v_base + offs_l[:, None] * stride_ql + offs_d1[None, :] * stride_qd, mask=mask_l[:, None], other=0.0).to(tl.float32)

            # 1. Intra-chunk attention: S_intra = (Q_c @ K_c^T) * causal_decay
            # causal decay: decay^(i - j) for i >= j
            diff = offs_c[:, None] - offs_c[None, :]
            causal_mask = diff >= 0
            decay_intra = tl.where(causal_mask, tl.math.exp(diff * tl.math.log(decay)), 0.0)

            intra_scores = tl.dot(qc, tl.trans(kc)) * decay_intra
            intra_out = tl.dot(intra_scores, vc) # [CHUNK_SIZE, D]

            # 2. Inter-chunk state contribution: O_inter = (Q_c @ S^T) * decay^(i + 1)
            decay_inter = tl.math.exp((offs_c[:, None] + 1) * tl.math.log(decay))
            inter_out = tl.dot(qc, tl.trans(s_mat)) * decay_inter # [CHUNK_SIZE, D]

            chunk_out = intra_out + inter_out
            tl.store(out_base + offs_l[:, None] * stride_ql + offs_d1[None, :] * stride_qd, chunk_out, mask=mask_l[:, None])

            # 3. Update state: S = S * chunk_decay + V_c^T @ (K_c * decay^(CHUNK_SIZE - 1 - i))
            decay_k = tl.math.exp((CHUNK_SIZE - 1 - offs_c[:, None]) * tl.math.log(decay))
            kc_scaled = kc * decay_k
            kv_chunk = tl.dot(tl.trans(vc), kc_scaled) # [D, D]
            s_mat = s_mat * chunk_decay + kv_chunk

        # Store final state [D, D]
        tl.store(State_out_ptr + b * stride_sb + h * stride_sh + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2, s_mat)


def chunk_linear_recurrence_cuda(
    q: torch.Tensor,                # [B, H, L, D]
    k: torch.Tensor,                # [B, H, L, D]
    v: torch.Tensor,                # [B, H, L, D]
    decay: float = 0.95,
    state: Optional[torch.Tensor] = None # [B, H, D, D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes fused chunk-parallel linear recurrent attention on CUDA via Triton.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is required for chunk_linear_recurrence_cuda")

    b, h, seq_len, d = q.shape
    out = torch.empty_like(q)
    next_state = torch.empty(b, h, d, d, device=q.device, dtype=torch.float32)

    has_state = state is not None
    state_in = state if has_state else next_state

    chunk_size = 64
    grid = (b * h,)

    _chunk_linear_recurrence_kernel[grid](
        q, k, v, out,
        state_in, next_state,
        decay, seq_len,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        next_state.stride(0), next_state.stride(1), next_state.stride(2), next_state.stride(3),
        D=d,
        CHUNK_SIZE=chunk_size,
        HAS_STATE=has_state,
        num_warps=4,
        num_stages=2
    )

    return out, next_state
