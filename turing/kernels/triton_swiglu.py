"""
In-SRAM Fused SwiGLU Triton Kernel with Active Tile Channel Pruning.
"""

from typing import Optional
import torch

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
    def _sparse_swiglu_kernel(
        X_ptr, Gate_ptr, Up_ptr, Down_ptr, Out_ptr,
        ActiveTiles_ptr, M,
        stride_xm, stride_xk,
        stride_gk, stride_gn,
        stride_uk, stride_un,
        stride_dn, stride_dk,
        stride_om, stride_ok,
        K: tl.constexpr,
        ACTIVE_TILES: tl.constexpr,
        TILE_SIZE_C: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k_out = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for t_idx in range(ACTIVE_TILES):
            tile_id = tl.load(ActiveTiles_ptr + t_idx)
            tile_col_start = tile_id * TILE_SIZE_C
            offs_tile = tile_col_start + tl.arange(0, TILE_SIZE_C)

            gate_acc = tl.zeros((BLOCK_M, TILE_SIZE_C), dtype=tl.float32)
            up_acc = tl.zeros((BLOCK_M, TILE_SIZE_C), dtype=tl.float32)

            for k_step in range(0, K, BLOCK_K):
                offs_k_in = k_step + tl.arange(0, BLOCK_K)
                mask_x = (offs_m[:, None] < M) & (offs_k_in[None, :] < K)
                x_val = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k_in[None, :] * stride_xk, mask=mask_x, other=0.0)

                mask_w = (offs_k_in[:, None] < K)
                g_val = tl.load(Gate_ptr + offs_k_in[:, None] * stride_gk + offs_tile[None, :] * stride_gn, mask=mask_w, other=0.0)
                u_val = tl.load(Up_ptr + offs_k_in[:, None] * stride_uk + offs_tile[None, :] * stride_un, mask=mask_w, other=0.0)

                gate_acc += tl.dot(x_val, g_val)
                up_acc += tl.dot(x_val, u_val)

            # In-SRAM SiLU: x * sigmoid(x)
            silu_g = gate_acc / (1.0 + tl.exp(-gate_acc))
            h = (silu_g * up_acc).to(tl.float16)

            # Down projection
            d_val = tl.load(Down_ptr + offs_tile[:, None] * stride_dn + offs_k_out[None, :] * stride_dk, mask=(offs_k_out[None, :] < K), other=0.0)
            acc_out += tl.dot(h, d_val)

        mask_out = (offs_m[:, None] < M) & (offs_k_out[None, :] < K)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_k_out[None, :] * stride_ok, acc_out.to(tl.float16), mask=mask_out)

def launch_triton_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    active_tiles: torch.Tensor,
    tile_size: int = 256
) -> torch.Tensor:
    """
    Python wrapper to launch fused SwiGLU Triton kernel.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in current environment")

    m, k = x.shape
    out = torch.empty((m, k), dtype=torch.float16, device=x.device)

    active_count = active_tiles.shape[0]
    BLOCK_M = 16
    BLOCK_K = 32

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(k, BLOCK_K))

    _sparse_swiglu_kernel[grid](
        x, w_gate, w_up, w_down, out,
        active_tiles, m,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        w_up.stride(0), w_up.stride(1),
        w_down.stride(0), w_down.stride(1),
        out.stride(0), out.stride(1),
        K=k,
        ACTIVE_TILES=active_count,
        TILE_SIZE_C=tile_size,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        num_stages=2,
        num_warps=4
    )

    return out
