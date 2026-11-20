"""
Fused Triton GPU Kernel for MoE Grouped Block-GEMM with In-SRAM SwiGLU.
Fuses expert routing, SwiGLU projection (silu(x @ W_g) * (x @ W_u) @ W_d),
routing probability scaling, and output accumulation into a single kernel dispatch.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F

__all__ = ["fused_moe_block_gemm_cuda"]

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
    def _fused_moe_gemm_token_kernel(
        X_ptr,              # [M, K]
        W_gate_ptr,         # [E, K, N]
        W_up_ptr,           # [E, K, N]
        W_down_ptr,         # [E, N, K]
        TopK_indices_ptr,   # [M, TopK]
        TopK_probs_ptr,     # [M, TopK]
        Out_ptr,            # [M, K]
        M,
        stride_xm, stride_xk,
        stride_ge, stride_gk, stride_gn,
        stride_ue, stride_uk, stride_un,
        stride_de, stride_dn, stride_dk,
        stride_im, stride_ik,
        stride_pm, stride_pk,
        stride_om, stride_ok,
        K: tl.constexpr,
        N: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr
    ):
        m_idx = tl.program_id(0)
        if m_idx >= M:
            return

        offs_k = tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)

        mask_k = offs_k < K
        mask_n = offs_n < N

        # Load token x: [BLOCK_K]
        x_val = tl.load(X_ptr + m_idx * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0).to(tl.float32)

        acc_out = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for top_idx in range(TOP_K):
            exp_id = tl.load(TopK_indices_ptr + m_idx * stride_im + top_idx * stride_ik)
            exp_prob = tl.load(TopK_probs_ptr + m_idx * stride_pm + top_idx * stride_pk)

            if exp_prob > 0.0:
                # 1. Gate and Up projections
                g_ptrs = W_gate_ptr + exp_id * stride_ge + offs_k[:, None] * stride_gk + offs_n[None, :] * stride_gn
                u_ptrs = W_up_ptr + exp_id * stride_ue + offs_k[:, None] * stride_uk + offs_n[None, :] * stride_un

                g_w = tl.load(g_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
                u_w = tl.load(u_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

                gate_val = tl.sum(x_val[:, None] * g_w, axis=0)
                up_val = tl.sum(x_val[:, None] * u_w, axis=0)

                # 2. In-SRAM SiLU(gate) * up
                silu_gate = gate_val / (1.0 + tl.exp(-gate_val))
                hidden_n = silu_gate * up_val  # [BLOCK_N]

                # 3. Down projection
                d_ptrs = W_down_ptr + exp_id * stride_de + offs_n[:, None] * stride_dn + offs_k[None, :] * stride_dk
                d_w = tl.load(d_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

                down_val = tl.sum(hidden_n[:, None] * d_w, axis=0) * exp_prob
                acc_out += down_val

        # Direct non-atomic store to output row
        out_ptrs = Out_ptr + m_idx * stride_om + offs_k * stride_ok
        tl.store(out_ptrs, acc_out.to(tl.float32), mask=mask_k)


def fused_moe_block_gemm_cuda(
    x: torch.Tensor,                # [M, K]
    w_gate: torch.Tensor,           # [E, K, N]
    w_up: torch.Tensor,             # [E, K, N]
    w_down: torch.Tensor,           # [E, N, K]
    topk_indices: torch.Tensor,     # [M, TopK]
    topk_probs: torch.Tensor        # [M, TopK]
) -> torch.Tensor:
    """
    Fused MoE block GEMM with in-SRAM SwiGLU activation and probability weighting.
    Falls back to PyTorch when Triton or CUDA is unavailable.
    """
    orig_shape = x.shape
    x_flat = x.view(-1, orig_shape[-1])
    m, k = x_flat.shape
    topk_indices_flat = topk_indices.view(-1, topk_indices.shape[-1])
    topk_probs_flat = topk_probs.view(-1, topk_probs.shape[-1])
    top_k = topk_indices_flat.shape[-1]
    num_experts = w_gate.shape[0]

    # Ensure shape alignment: [E, K, N], [E, K, N], [E, N, K]
    if w_gate.shape[1] != k and w_gate.shape[2] == k:
        w_g = w_gate.transpose(1, 2).contiguous()
        w_u = w_up.transpose(1, 2).contiguous()
    else:
        w_g = w_gate.contiguous()
        w_u = w_up.contiguous()

    if w_down.shape[2] != k and w_down.shape[1] == k:
        w_d = w_down.transpose(1, 2).contiguous()
    else:
        w_d = w_down.contiguous()

    n = w_g.shape[2]

    if not x.is_cuda or not HAS_TRITON:
        # PyTorch Reference Implementation
        out = torch.zeros((m, k), device=x.device, dtype=x.dtype)
        for t_idx in range(top_k):
            exp_idx = topk_indices_flat[:, t_idx]   # [M]
            exp_prob = topk_probs_flat[:, t_idx].unsqueeze(-1)  # [M, 1]

            for e in range(num_experts):
                mask = (exp_idx == e)
                if not mask.any():
                    continue
                sub_x = x_flat[mask]  # [Tokens, K]
                gate_out = F.silu(torch.matmul(sub_x, w_g[e]))  # [Tokens, N]
                up_out = torch.matmul(sub_x, w_u[e])            # [Tokens, N]
                down_out = torch.matmul(gate_out * up_out, w_d[e])  # [Tokens, K]
                out[mask] += down_out * exp_prob[mask].to(dtype=down_out.dtype)

        return out.view(orig_shape)

    out = torch.empty((m, k), device=x.device, dtype=torch.float32)

    BLOCK_K = triton.next_power_of_2(k)
    BLOCK_N = triton.next_power_of_2(n)

    # If tile sizes exceed 1024, clamp to 512
    if BLOCK_K * BLOCK_N > 1048576:
        BLOCK_K = min(1024, BLOCK_K)
        BLOCK_N = min(1024, BLOCK_N)

    grid = (m,)

    _fused_moe_gemm_token_kernel[grid](
        x_flat.contiguous(),
        w_g,
        w_u,
        w_d,
        topk_indices_flat.contiguous().to(torch.int32),
        topk_probs_flat.contiguous().to(torch.float32),
        out,
        m,
        x_flat.stride(0), x_flat.stride(1),
        w_g.stride(0), w_g.stride(1), w_g.stride(2),
        w_u.stride(0), w_u.stride(1), w_u.stride(2),
        w_d.stride(0), w_d.stride(1), w_d.stride(2),
        topk_indices_flat.stride(0), topk_indices_flat.stride(1),
        topk_probs_flat.stride(0), topk_probs_flat.stride(1),
        out.stride(0), out.stride(1),
        K=k,
        N=n,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        num_stages=2,
        num_warps=4
    )

    return out.to(dtype=x.dtype).view(orig_shape)
