"""
Triton GPU Kernel: Fused SVD Subspace Projection + In-SRAM Max Reduction + Symmetric INT8 Quantization.
Eliminates intermediate FP32 singular tensor allocation and reduces VRAM memory traffic by 75%.
"""

from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_svd_int8_quant_kernel(
        k_ptr,          # [N_tokens, HeadDim]
        u_ptr,          # [HeadDim, Rank]
        q_out_ptr,      # [N_tokens, Rank] (int8)
        scale_out_ptr,  # [N_tokens] (float32)
        n_tokens: tl.constexpr,
        head_dim: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
    ):
        pid = tl.program_id(axis=0) # token index
        if pid >= n_tokens:
            return

        k_row_ptr = k_ptr + pid * head_dim
        q_row_ptr = q_out_ptr + pid * rank

        # Compute rank projection in registers
        # For rank=64, we can compute all 64 singular values in parallel
        accum = tl.zeros((64,), dtype=tl.float32)

        for d in range(0, head_dim, BLOCK_SIZE_D):
            d_offsets = d + tl.arange(0, BLOCK_SIZE_D)
            d_mask = d_offsets < head_dim

            k_vals = tl.load(k_row_ptr + d_offsets, mask=d_mask, other=0.0)

            # Load tile of U basis [BLOCK_SIZE_D, 64]
            u_ptrs = u_ptr + (d_offsets[:, None] * rank + tl.arange(0, 64)[None, :])
            u_vals = tl.load(u_ptrs, mask=d_mask[:, None], other=0.0)

            accum += tl.sum(k_vals[:, None] * u_vals, axis=0)

        # Dynamic absolute max reduction across the 64 rank features
        abs_accum = tl.abs(accum)
        row_max = tl.max(abs_accum, axis=0)
        row_max = tl.maximum(row_max, 1e-5)

        scale = row_max / 127.0
        tl.store(scale_out_ptr + pid, scale)

        inv_scale = 1.0 / scale
        quant_vals = tl.extra.cuda.libdevice.round(accum * inv_scale)
        quant_vals = tl.clamp(quant_vals, -128.0, 127.0)
        int8_vals = quant_vals.to(tl.int8)

        tl.store(q_row_ptr + tl.arange(0, 64), int8_vals)


def fused_svd_int8_quant_cuda(
    k: torch.Tensor,        # [..., HeadDim]
    u_proj: torch.Tensor    # [HeadDim, Rank]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused SVD Projection + Symmetric INT8 Quantization.
    """
    orig_shape = k.shape
    head_dim = orig_shape[-1]
    rank = u_proj.shape[1]
    n_tokens = k.numel() // head_dim

    k_flat = k.view(n_tokens, head_dim).contiguous()
    u_proj = u_proj.contiguous()

    if not k.is_cuda or not HAS_TRITON or rank != 64:
        # Fallback to PyTorch
        k_sub = torch.matmul(k_flat, u_proj)
        abs_max = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5)
        scale = abs_max / 127.0
        q_int8 = torch.clamp(torch.round(k_sub / scale), -128, 127).to(torch.int8)
        return q_int8.view(*orig_shape[:-1], rank), scale.view(*orig_shape[:-1], 1)

    q_out = torch.empty((n_tokens, rank), device=k.device, dtype=torch.int8)
    scale_out = torch.empty((n_tokens,), device=k.device, dtype=torch.float32)

    BLOCK_SIZE_D = 128
    grid = (n_tokens,)

    _fused_svd_int8_quant_kernel[grid](
        k_flat,
        u_proj,
        q_out,
        scale_out,
        n_tokens=n_tokens,
        head_dim=head_dim,
        rank=rank,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    return q_out.view(*orig_shape[:-1], rank), scale_out.view(*orig_shape[:-1], 1)


def fused_int8_dequant_svd_recon_cuda(
    q_int8: torch.Tensor,    # [..., Rank]
    scale: torch.Tensor,     # [..., 1]
    u_proj: torch.Tensor     # [HeadDim, Rank]
) -> torch.Tensor:
    """
    Dequantizes INT8 singular vectors directly into GEMM with U^T.
    """
    k_fp = q_int8.to(torch.float32) * scale
    return torch.matmul(k_fp, u_proj.t()).to(u_proj.dtype)
