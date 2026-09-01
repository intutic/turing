"""
Fused Triton GPU Kernel for Single-Step Linear Recurrence Attention.
Fuses recurrent state update (S_t = decay * S_{t-1} + K_t^T V_t) and output projection (O_t = Q_t S_t)
into a single persistent SRAM kernel launch for fast token decode.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = ["fused_linear_recurrence_step_cuda"]

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
    def _linear_recurrence_step_kernel(
        Q_ptr,          # [B, H, D]
        K_ptr,          # [B, H, D]
        V_ptr,          # [B, H, D]
        State_ptr,      # [B, H, D, D]
        Out_ptr,        # [B, H, D]
        decay,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_kh, stride_kd,
        stride_vb, stride_vh, stride_vd,
        stride_sb, stride_sh, stride_sd1, stride_sd2,
        stride_ob, stride_oh, stride_od,
        D: tl.constexpr
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

        # Load Q, K, V vectors [D]
        q_val = tl.load(q_base + offs_d1 * stride_qd, mask=(offs_d1 < D), other=0.0).to(tl.float32)
        k_val = tl.load(k_base + offs_d1 * stride_kd, mask=(offs_d1 < D), other=0.0).to(tl.float32)
        v_val = tl.load(v_base + offs_d1 * stride_vd, mask=(offs_d1 < D), other=0.0).to(tl.float32)

        # Load current state [D, D]
        s_mat = tl.load(s_base + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2,
                        mask=(offs_d1[:, None] < D) & (offs_d2[None, :] < D), other=0.0).to(tl.float32)

        # S_t = decay * S_{t-1} + K_t^T @ V_t: [D, 1] @ [1, D] -> [D, D]
        kv_outer = k_val[:, None] * v_val[None, :]
        s_new = s_mat * decay + kv_outer

        # O_t = Q_t @ S_t: [1, D] @ [D, D] -> [1, D]
        out_val = tl.sum(q_val[:, None] * s_new, axis=0)

        # Store updated state and output
        tl.store(s_base + offs_d1[:, None] * stride_sd1 + offs_d2[None, :] * stride_sd2,
                 s_new.to(tl.float32), mask=(offs_d1[:, None] < D) & (offs_d2[None, :] < D))
        tl.store(out_base + offs_d2 * stride_od, out_val.to(tl.float32), mask=(offs_d2 < D))


def fused_linear_recurrence_step_cuda(
    q: torch.Tensor,                # [B, H, D] or [B, H, 1, D]
    k: torch.Tensor,                # [B, H, D]
    v: torch.Tensor,                # [B, H, D]
    state: torch.Tensor,            # [B, H, D, D]
    decay: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes single-step linear recurrent attention on CUDA via Triton.
    Updates state in-place and returns (out, updated_state).
    """
    if q.dim() == 4:
        q_3d = q.squeeze(2)
    else:
        q_3d = q

    if k.dim() == 4:
        k_3d = k.squeeze(2)
    else:
        k_3d = k

    if v.dim() == 4:
        v_3d = v.squeeze(2)
    else:
        v_3d = v

    batch, num_heads, head_dim = q_3d.shape

    if not q.is_cuda or not HAS_TRITON:
        # PyTorch Reference Implementation
        # S_t = decay * S_{t-1} + K_t^T @ V_t
        kv_outer = torch.einsum('bhd,bhf->bhdf', k_3d, v_3d)
        new_state = state * decay + kv_outer
        # O_t = Q_t @ S_t
        out = torch.einsum('bhd,bhdf->bhf', q_3d, new_state)
        state.copy_(new_state)
        return (out.unsqueeze(2) if q.dim() == 4 else out), state

    out = torch.empty((batch, num_heads, head_dim), device=q.device, dtype=q.dtype)

    grid = (batch, num_heads)
    _linear_recurrence_step_kernel[grid](
        q_3d.contiguous(),
        k_3d.contiguous(),
        v_3d.contiguous(),
        state,
        out,
        decay,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_3d.stride(0), k_3d.stride(1), k_3d.stride(2),
        v_3d.stride(0), v_3d.stride(1), v_3d.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        D=head_dim,
        num_stages=2,
        num_warps=4
    )

    return (out.unsqueeze(2) if q.dim() == 4 else out), state
