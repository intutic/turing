"""
Triton GPU Kernel: Fused Base GEMV + Multi-Tenant LoRA Rank-8 Contraction.
Fuses base model GEMV (x @ W_base) with low-rank tenant adapter ((x @ W_A) @ W_B * alpha)
in a single Tensor Core block without materializing intermediate rank-8 tensors in global VRAM.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_lora_gemv_kernel(
        x_ptr,           # [Batch, InDim]
        w_base_ptr,      # [InDim, OutDim]
        w_a_ptr,         # [InDim, Rank]
        w_b_ptr,         # [Rank, OutDim]
        out_ptr,         # [Batch, OutDim]
        alpha: tl.constexpr,
        in_dim: tl.constexpr,
        out_dim: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_SIZE_O: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_b = tl.program_id(0) # Batch
        pid_o = tl.program_id(1) # OutDim block

        o_offsets = pid_o * BLOCK_SIZE_O + tl.arange(0, BLOCK_SIZE_O)
        o_mask = o_offsets < out_dim

        x_base = x_ptr + pid_b * in_dim

        # 1. Compute LoRA intermediate z = x @ W_A (Rank <= 16) in SRAM
        r_offsets = tl.arange(0, 16)
        r_mask = r_offsets < rank

        z_lora = tl.zeros((16,), dtype=tl.float32)

        for k in range(0, in_dim, BLOCK_SIZE_K):
            k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < in_dim
            x_vals = tl.load(x_base + k_offsets, mask=k_mask, other=0.0)

            wa_ptrs = w_a_ptr + k_offsets[:, None] * rank + r_offsets[None, :]
            wa_vals = tl.load(wa_ptrs, mask=k_mask[:, None] & r_mask[None, :], other=0.0)
            z_lora += tl.sum(x_vals[:, None] * wa_vals, axis=0)

        # 2. Accumulate Base GEMV (x @ W_base)
        accum_base = tl.zeros((BLOCK_SIZE_O,), dtype=tl.float32)

        for k in range(0, in_dim, BLOCK_SIZE_K):
            k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < in_dim
            x_vals = tl.load(x_base + k_offsets, mask=k_mask, other=0.0)

            wb_ptrs = w_base_ptr + (k_offsets[:, None] * out_dim + o_offsets[None, :])
            wb_vals = tl.load(wb_ptrs, mask=k_mask[:, None] & o_mask[None, :], other=0.0)

            accum_base += tl.sum(wb_vals * x_vals[:, None], axis=0)

        # 3. Add LoRA W_B contribution: sum_r z[r] * W_B[r, o] * alpha
        wb2_ptrs = w_b_ptr + r_offsets[:, None] * out_dim + o_offsets[None, :]
        wb2_vals = tl.load(wb2_ptrs, mask=r_mask[:, None] & o_mask[None, :], other=0.0)
        accum_lora = tl.sum((z_lora[:, None] * alpha) * wb2_vals, axis=0)

        # 4. Store final sum
        total_out = accum_base + accum_lora
        out_base = out_ptr + pid_b * out_dim + o_offsets
        tl.store(out_base, total_out, mask=o_mask)



def fused_lora_gemv_cuda(
    x: torch.Tensor,
    w_base: torch.Tensor,
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Fused Base GEMV + LoRA Rank-8 contraction kernel.
    """
    orig_shape = x.shape
    in_dim = orig_shape[-1]
    out_dim = w_base.shape[1]
    rank = w_a.shape[1]

    x_flat = x.view(-1, in_dim).contiguous()
    batch_size = x_flat.shape[0]

    if not x.is_cuda or not HAS_TRITON or batch_size > 8 or rank > 16:
        # Large batch prefill or non-CUDA: use hardware cuBLAS Tensor Core GEMM / C++ SIMD
        if not x.is_cuda:
            try:
                import turing.turing_csrc as turing_csrc
                x_cpu = x_flat.detach().to(torch.float32).cpu().contiguous().numpy()
                wb_cpu = w_base.detach().to(torch.float32).cpu().contiguous().numpy()
                wa_cpu = w_a.detach().to(torch.float32).cpu().contiguous().numpy()
                wb2_cpu = w_b.detach().to(torch.float32).cpu().contiguous().numpy()
                out_np = turing_csrc.fused_lora_gemv_cpu(x_cpu, wb_cpu, wa_cpu, wb2_cpu, float(alpha))
                return torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype).view(*orig_shape[:-1], out_dim)
            except Exception:
                pass
        base_out = torch.matmul(x_flat, w_base)
        lora_out = torch.matmul(torch.matmul(x_flat, w_a) * alpha, w_b)
        return (base_out + lora_out).view(*orig_shape[:-1], out_dim)

    out = torch.empty((batch_size, out_dim), device=x.device, dtype=torch.float32)


    BLOCK_SIZE_O = 64
    BLOCK_SIZE_K = 64

    grid = (batch_size, triton.cdiv(out_dim, BLOCK_SIZE_O))

    _fused_lora_gemv_kernel[grid](
        x_flat,
        w_base,
        w_a,
        w_b,
        out,
        alpha=float(alpha),
        in_dim=in_dim,
        out_dim=out_dim,
        rank=min(rank, 16),
        BLOCK_SIZE_O=BLOCK_SIZE_O,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out.to(x.dtype).view(*orig_shape[:-1], out_dim)
