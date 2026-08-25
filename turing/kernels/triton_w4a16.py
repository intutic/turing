"""
W4A16 / Marlin-Style Packed INT4 GEMM Triton Kernel with In-SRAM Dequantization.
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
    def _w4a16_subspace_gemm_kernel(
        X_ptr, W_packed_ptr, Scales_ptr, Out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk_half,
        stride_sn, stride_sk_groups,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_step in range(0, K, BLOCK_K):
            offs_k = k_step + tl.arange(0, BLOCK_K)

            # Load activation tile
            mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
            x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=mask_x, other=0.0)

            # Load packed weights (2 nibbles per byte)
            offs_k_byte = offs_k // 2
            w_ptrs = W_packed_ptr + offs_n[:, None] * stride_wn + offs_k_byte[None, :] * stride_wk_half
            w_byte = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0)

            # Bitwise nibble extraction
            is_odd = (offs_k[None, :] % 2 == 1)
            w_int4 = tl.where(is_odd, (w_byte >> 4) & 0x0F, w_byte & 0x0F)
            w_fp = (w_int4.to(tl.float16) - 8.0)

            # Load scale factors (group size 128)
            scale_ptrs = Scales_ptr + offs_n[:, None] * stride_sn + (offs_k[None, :] // 128) * stride_sk_groups
            w_scale = tl.load(scale_ptrs, mask=(offs_n[:, None] < N), other=1.0)

            w_final = w_fp * w_scale
            acc += tl.dot(x, tl.trans(w_final.to(tl.float16)))

        mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc.to(tl.float16), mask=mask_out)

def launch_triton_w4a16_gemm(
    x: torch.Tensor,
    w_packed: torch.Tensor, # [N, K // 2], uint8
    scales: torch.Tensor,   # [N, K // 128], fp16
    group_size: int = 128
) -> torch.Tensor:
    """
    Python launcher for W4A16 packed INT4 GEMM.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in current environment")

    m, k = x.shape
    n = w_packed.shape[0]

    out = torch.empty((m, n), dtype=torch.float16, device=x.device)

    BLOCK_M = 16
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    _w4a16_subspace_gemm_kernel[grid](
        x, w_packed, scales, out,
        m, n, k,
        x.stride(0), x.stride(1),
        w_packed.stride(0), w_packed.stride(1),
        scales.stride(0), scales.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K
    )

    return out
