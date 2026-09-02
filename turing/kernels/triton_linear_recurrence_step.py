"""
Fused Triton GPU Kernel for Single-Step Linear Recurrence Attention.
Fuses recurrent state update (S_t = decay * S_{t-1} + K_t^T V_t) and output projection (O_t = Q_t S_t)
into a single persistent SRAM kernel launch for fast token decode.
"""

import math
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


def fused_kda_conv_recurrence_step_cuda(
    qkv: torch.Tensor,                  # [B, 3, H, D]
    conv_state: torch.Tensor,           # [B, 3, H, D, KERNEL_SIZE]
    conv_weight: torch.Tensor,          # [3, H, D, KERNEL_SIZE]
    state: torch.Tensor,                # [B, H, D, D]
    gate: torch.Tensor,                 # [B, H, D] (log-decay alpha)
    beta: torch.Tensor,                 # [B, H] (delta learning rate)
    norm_eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused Kimi Delta Attention (KDA) step combining short causal convolution,
    SiLU activation, L2 normalization, delta state recurrence, and output projection.
    """
    B, _, H, D = qkv.shape
    kernel_size = conv_weight.shape[-1]

    # Update conv window: shift left and append current token
    new_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
    new_conv_state[..., -1] = qkv
    conv_state.copy_(new_conv_state)

    # Depthwise 1D conv + SiLU
    conv_out = torch.sum(new_conv_state * conv_weight.unsqueeze(0), dim=-1)  # [B, 3, H, D]
    mixed = F.silu(conv_out)

    q = mixed[:, 0]  # [B, H, D]
    k = mixed[:, 1]  # [B, H, D]
    v = mixed[:, 2]  # [B, H, D]

    # L2 normalization with eps inside sqrt (FLA convention)
    q_norm = q / torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True) + norm_eps)
    k_norm = k / torch.sqrt(torch.sum(k ** 2, dim=-1, keepdim=True) + norm_eps)
    q_scaled = q_norm * (1.0 / math.sqrt(D))

    # Delta recurrence: S = Diag(alpha) S; delta = (v - S^T k) * beta; S += k delta^T; out = S^T q
    alpha = torch.exp(gate).unsqueeze(-1)  # [B, H, D, 1]
    s_decayed = state * alpha              # [B, H, D, D]

    # S^T k -> [B, H, D]
    st_k = torch.einsum('bhde,bhd->bhe', s_decayed, k_norm)
    delta = (v - st_k) * beta.unsqueeze(-1)  # [B, H, D]

    # S = S + k @ delta^T
    k_delta = torch.einsum('bhd,bhe->bhde', k_norm, delta)
    new_state = s_decayed + k_delta
    state.copy_(new_state)

    # out = S^T q
    out = torch.einsum('bhde,bhd->bhe', new_state, q_scaled)
    return out, state, conv_state

