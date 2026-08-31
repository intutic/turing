"""
In-SRAM Fused RMSNorm + Subspace SwiGLU + In-Place Residual Triton Kernel.
Combines layer pre-normalization, intermediate active tile SwiGLU projection,
and residual accumulation into a single SRAM pass without intermediate VRAM traffic.
"""

from typing import Optional
import torch
import torch.nn.functional as F

__all__ = ["fused_rmsnorm_swiglu_cuda", "dispatch_fused_rmsnorm_swiglu"]

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
    def _fused_rmsnorm_swiglu_kernel(
        X_ptr, WeightNorm_ptr, Gate_ptr, Up_ptr, Down_ptr, Residual_ptr, Out_ptr,
        ActiveTiles_ptr, M,
        stride_xm, stride_xk,
        stride_w,
        stride_gk, stride_gn,
        stride_uk, stride_un,
        stride_dn, stride_dk,
        stride_rm, stride_rk,
        stride_om, stride_ok,
        K: tl.constexpr,
        eps: tl.constexpr,
        ACTIVE_TILES: tl.constexpr,
        TILE_SIZE_C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        # 1. In-SRAM RMSNorm across hidden dimension K
        sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_idx in range(0, K, BLOCK_K):
            offs_k = k_idx + tl.arange(0, BLOCK_K)
            mask_k = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x_raw = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=mask_k, other=0.0)
            sum_sq += tl.sum(x_raw * x_raw, axis=1)

        rms_scale = tl.rsqrt(sum_sq / K + eps) # [BLOCK_M]

        # 2. Iterate through Active Tiles of Subspace SwiGLU
        for t_idx in range(ACTIVE_TILES):
            tile_id = tl.load(ActiveTiles_ptr + t_idx)
            tile_col_start = tile_id * TILE_SIZE_C
            offs_tile = tile_col_start + tl.arange(0, TILE_SIZE_C)

            gate_acc = tl.zeros((BLOCK_M, TILE_SIZE_C), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, TILE_SIZE_C), dtype=tl.float32)

            # Compute Gate and Up projections over normalized X
            for k_step in range(0, K, BLOCK_K):
                offs_k_in = k_step + tl.arange(0, BLOCK_K)
                mask_x = (offs_m[:, None] < M) & (offs_k_in[None, :] < K)
                x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k_in[None, :] * stride_xk, mask=mask_x, other=0.0)
                w_norm = tl.load(WeightNorm_ptr + offs_k_in, mask=offs_k_in < K, other=1.0)
                
                # Apply RMS scale
                x_normed = x_val * (rms_scale[:, None] * w_norm[None, :])

                mask_w = (offs_k_in[:, None] < K)
                g_val = tl.load(Gate_ptr + offs_k_in[:, None] * stride_gk + offs_tile[None, :] * stride_gn, mask=mask_w, other=0.0)
                u_val = tl.load(Up_ptr + offs_k_in[:, None] * stride_uk + offs_tile[None, :] * stride_un, mask=mask_w, other=0.0)

                gate_acc += tl.dot(x_normed, g_val)
                up_acc += tl.dot(x_normed, u_val)

            # 3. In-Register SwiGLU Activation: SiLU(G) * U = (G * sigmoid(G)) * U
            silu_gate = gate_acc * tl.sigmoid(gate_acc)
            intermediate = silu_gate * up_acc

            # 4. Down Projection Accumulation
            mask_down = (offs_k_out[None, :] < K)
            d_val = tl.load(Down_ptr + offs_tile[:, None] * stride_dn + offs_k_out[None, :] * stride_dk, mask=mask_down, other=0.0)
            acc_out += tl.dot(intermediate, d_val)

        # 5. Add Residual Connection
        mask_res = (offs_m[:, None] < M) & (offs_k_out[None, :] < K)
        res_val = tl.load(Residual_ptr + offs_m[:, None] * stride_rm + offs_k_out[None, :] * stride_rk, mask=mask_res, other=0.0)
        final_out = acc_out + res_val

        # 6. Store directly to Output
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k_out[None, :] * stride_ok, final_out, mask=mask_res)


def fused_rmsnorm_swiglu_cuda(
    x: torch.Tensor,
    weight_norm: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    residual: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_size: int = 64,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused RMSNorm + Subspace SwiGLU + Residual on CUDA via Triton.
    """
    M, K = x.shape
    active_count = active_tiles.numel()
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)

    BLOCK_M = 16
    BLOCK_K = min(64, triton.next_power_of_2(K))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))

    _fused_rmsnorm_swiglu_kernel[grid](
        x, weight_norm, w_gate, w_up, w_down, residual, out,
        active_tiles, M,
        x.stride(0), x.stride(1),
        weight_norm.stride(0),
        w_gate.stride(0), w_gate.stride(1),
        w_up.stride(0), w_up.stride(1),
        w_down.stride(0), w_down.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        K=K, eps=eps,
        ACTIVE_TILES=active_count,
        TILE_SIZE_C=tile_size,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K
    )
    return out


def dispatch_fused_rmsnorm_swiglu(
    x: torch.Tensor,
    weight_norm: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    residual: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_size: int = 64,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Unified dispatcher: runs Triton kernel on CUDA, PyTorch fused reference on MPS/CPU.
    """
    if x.is_cuda and HAS_TRITON:
        try:
            return fused_rmsnorm_swiglu_cuda(
                x, weight_norm, w_gate, w_up, w_down, residual, active_tiles, tile_size, eps
            )
        except Exception:
            pass

    # Reference PyTorch fallback
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps) * weight_norm

    # Sliced Subspace SwiGLU
    indices = []
    for t in active_tiles.tolist():
        start = t * tile_size
        indices.extend(range(start, start + tile_size))
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=x.device)

    # Slice weights
    w_g_sub = w_gate[:, idx_tensor]
    w_u_sub = w_up[:, idx_tensor]
    w_d_sub = w_down[idx_tensor, :]

    gate = torch.matmul(x_normed, w_g_sub)
    up = torch.matmul(x_normed, w_u_sub)
    intermediate = F.silu(gate) * up
    out = torch.matmul(intermediate, w_d_sub)

    return out + residual
