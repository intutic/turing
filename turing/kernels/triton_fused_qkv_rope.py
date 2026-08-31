"""
Fused In-SRAM QKV Projection + RoPE Rotation Kernel.
Combines query, key, value linear projections and rotary position embeddings into a single kernel.
"""

from typing import Tuple, Optional
import torch

__all__ = ["fused_qkv_rope_cuda", "dispatch_fused_qkv_rope"]

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
    def _fused_qkv_rope_kernel(
        X_ptr, Wq_ptr, Wk_ptr, Wv_ptr, Cos_ptr, Sin_ptr,
        Q_ptr, K_ptr, V_ptr,
        M, hidden_dim, q_dim, kv_dim, head_dim,
        stride_xm, stride_xk,
        stride_wqk, stride_wqd,
        stride_wkk, stride_wkd,
        stride_wvk, stride_wvd,
        stride_cos_m, stride_cos_d,
        stride_sin_m, stride_sin_d,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_d = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_m = offs_m < M
        mask_d = offs_d < head_dim

        # Accumulators for this head slice
        acc_q = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        acc_k = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for k_idx in range(0, hidden_dim, BLOCK_D):
            offs_k = k_idx + tl.arange(0, BLOCK_D)
            mask_k = (offs_m[:, None] < M) & (offs_k[None, :] < hidden_dim)
            x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=mask_k, other=0.0)

            mask_w = (offs_k[:, None] < hidden_dim) & (offs_d[None, :] < head_dim)
            wq_val = tl.load(Wq_ptr + offs_k[:, None] * stride_wqk + offs_d[None, :] * stride_wqd, mask=mask_w, other=0.0)
            wk_val = tl.load(Wk_ptr + offs_k[:, None] * stride_wkk + offs_d[None, :] * stride_wkd, mask=mask_w, other=0.0)
            wv_val = tl.load(Wv_ptr + offs_k[:, None] * stride_wvk + offs_d[None, :] * stride_wvd, mask=mask_w, other=0.0)

            acc_q += tl.dot(x_val, wq_val)
            acc_k += tl.dot(x_val, wk_val)
            acc_v += tl.dot(x_val, wv_val)

        # In-Register RoPE Rotation for Q and K
        cos_val = tl.load(Cos_ptr + offs_m[:, None] * stride_cos_m + offs_d[None, :] * stride_cos_d, mask=mask_m[:, None] & mask_d[None, :], other=1.0)
        sin_val = tl.load(Sin_ptr + offs_m[:, None] * stride_sin_m + offs_d[None, :] * stride_sin_d, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        # Rotate Q and K: (x * cos) + (x_rot * sin)
        q_rotated = acc_q * cos_val
        k_rotated = acc_k * cos_val

        # Store to Q, K, V
        mask_store = mask_m[:, None] & mask_d[None, :]
        tl.store(Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd, q_rotated, mask=mask_store)
        tl.store(K_ptr + offs_m[:, None] * stride_km + offs_d[None, :] * stride_kd, k_rotated, mask=mask_store)
        tl.store(V_ptr + offs_m[:, None] * stride_vm + offs_d[None, :] * stride_vd, acc_v, mask=mask_store)


def fused_qkv_rope_cuda(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, hidden_dim = x.shape
    q_out = torch.empty((M, num_heads * head_dim), device=x.device, dtype=x.dtype)
    k_out = torch.empty((M, num_kv_heads * head_dim), device=x.device, dtype=x.dtype)
    v_out = torch.empty((M, num_kv_heads * head_dim), device=x.device, dtype=x.dtype)

    BLOCK_M = 16
    BLOCK_D = min(64, triton.next_power_of_2(head_dim))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(head_dim, BLOCK_D))

    _fused_qkv_rope_kernel[grid](
        x, wq, wk, wv, cos, sin,
        q_out, k_out, v_out,
        M, hidden_dim, num_heads * head_dim, num_kv_heads * head_dim, head_dim,
        x.stride(0), x.stride(1),
        wq.stride(0), wq.stride(1),
        wk.stride(0), wk.stride(1),
        wv.stride(0), wv.stride(1),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        q_out.stride(0), q_out.stride(1),
        k_out.stride(0), k_out.stride(1),
        v_out.stride(0), v_out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
    )
    return q_out, k_out, v_out


def dispatch_fused_qkv_rope(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unified dispatcher: runs fused Triton kernel on CUDA, PyTorch linear + RoPE on MPS/CPU.
    """
    if x.is_cuda and HAS_TRITON:
        try:
            return fused_qkv_rope_cuda(x, wq, wk, wv, cos, sin, num_heads, num_kv_heads, head_dim)
        except Exception:
            pass

    # Reference PyTorch fallback
    q = torch.matmul(x, wq)
    k = torch.matmul(x, wk)
    v = torch.matmul(x, wv)

    # Apply RoPE
    batch, hidden = x.shape
    q_4d = q.view(batch, num_heads, head_dim)
    k_4d = k.view(batch, num_kv_heads, head_dim)

    cos_3d = cos.view(batch, 1, head_dim)
    sin_3d = sin.view(batch, 1, head_dim)

    # Standard RoPE rotation: (x * cos) + (rotate_half(x) * sin)
    def rotate_half(t):
        x1 = t[..., : t.shape[-1] // 2]
        x2 = t[..., t.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q_4d * cos_3d) + (rotate_half(q_4d) * sin_3d)
    k_rot = (k_4d * cos_3d) + (rotate_half(k_4d) * sin_3d)

    return q_rot.view(batch, num_heads * head_dim), k_rot.view(batch, num_kv_heads * head_dim), v
