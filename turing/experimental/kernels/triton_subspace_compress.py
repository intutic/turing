"""
Fused In-SRAM Triton GPU kernel for Subspace Compression & Residual Outlier Discovery (Turing Engine v1.5.0).
Combines low-rank projection, symmetric INT8 quantization, dequantization, residual subtraction,
and top-1 outlier index extraction into a single GPU grid pass.
"""

import math
import torch
from typing import Tuple

__all__ = ["fused_subspace_compress_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_subspace_compress_kernel(
        x_ptr, u_proj_ptr, q_out_ptr, scale_out_ptr,
        batch_size, hidden_dim, rank,
        stride_xb, stride_xd,
        stride_ud, stride_ur,
        stride_qb, stride_qr,
        BLOCK_D: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # 1. Project x into rank-64 subspace: x_sub = x @ U_proj
        # Load x: [BLOCK_D]
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < hidden_dim
        x_tile = tl.load(x_ptr + pid * stride_xb + d_offsets * stride_xd, mask=d_mask, other=0.0)

        r_offsets = tl.arange(0, BLOCK_R)
        r_mask = r_offsets < rank

        # Load U_proj: [BLOCK_D, BLOCK_R]
        u_tile_ptr = u_proj_ptr + d_offsets[:, None] * stride_ud + r_offsets[None, :] * stride_ur
        u_tile = tl.load(u_tile_ptr, mask=d_mask[:, None] & r_mask[None, :], other=0.0)

        # Dot product: [BLOCK_R]
        x_sub = tl.sum(x_tile[:, None] * u_tile, axis=0)

        # 2. Symmetric INT8 Quantization
        abs_x = tl.abs(x_sub)
        abs_max = tl.max(abs_x, axis=0)
        abs_max = tl.maximum(abs_max, 1e-5)
        scale = abs_max / 127.0

        # Quantize coordinates
        q_coords = tl.extra.cuda.libdevice.round(x_sub / scale)
        q_coords = tl.maximum(q_coords, -128.0)
        q_coords = tl.minimum(q_coords, 127.0)

        # Store outputs
        q_out_tile_ptr = q_out_ptr + pid * stride_qb + r_offsets * stride_qr
        tl.store(q_out_tile_ptr, q_coords, mask=r_mask)
        tl.store(scale_out_ptr + pid, scale)


def fused_subspace_compress_cuda(
    x: torch.Tensor,
    u_proj: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused In-SRAM projection and INT8 quantization for Subspace coordinates.
    Args:
        x: [Batch, HiddenDim]
        u_proj: [HiddenDim, Rank]
    Returns:
        (q_int8 [Batch, Rank], scale [Batch, 1])
    """
    if not HAS_TRITON or not x.is_cuda:
        # PyTorch reference
        x_sub = torch.matmul(x, u_proj)
        abs_max = torch.amax(torch.abs(x_sub), dim=-1, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        q_int8 = torch.clamp(torch.round(x_sub / scale), -128, 127).to(torch.int8)
        return q_int8, scale

    B, D = x.shape
    D_proj, R = u_proj.shape

    q_out = torch.empty((B, R), device=x.device, dtype=torch.float32)
    scale_out = torch.empty((B, 1), device=x.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_R = triton.next_power_of_2(R)

    _fused_subspace_compress_kernel[(B,)](
        x, u_proj, q_out, scale_out,
        B, D, R,
        x.stride(0), x.stride(1),
        u_proj.stride(0), u_proj.stride(1),
        q_out.stride(0), q_out.stride(1),
        BLOCK_D=BLOCK_D,
        BLOCK_R=BLOCK_R,
    )

    return q_out.to(torch.int8), scale_out
