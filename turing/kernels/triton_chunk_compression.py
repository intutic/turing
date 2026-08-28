"""
Triton GPU Kernel: Fused Hierarchical Attention Chunk Pooling (HCA) & Top-K Sparsity Bitmask (CSA).
Compresses sequence chunks into summary representations in SRAM with zero 5D tensor allocations.
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
    def _hca_chunk_mean_pool_kernel(
        k_ptr,          # [TotalTokens, Heads, HeadDim]
        v_ptr,          # [TotalTokens, Heads, HeadDim]
        k_out_ptr,      # [NumChunks, Heads, HeadDim]
        v_out_ptr,      # [NumChunks, Heads, HeadDim]
        total_tokens: tl.constexpr,
        chunk_size: tl.constexpr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_SIZE_D: tl.constexpr,
    ):
        chunk_id = tl.program_id(0)
        h_idx = tl.program_id(1)

        t_start = chunk_id * chunk_size
        t_end = tl.minimum(t_start + chunk_size, total_tokens)
        actual_chunk_len = t_end - t_start

        d_offsets = tl.arange(0, BLOCK_SIZE_D)
        d_mask = d_offsets < head_dim

        accum_k = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
        accum_v = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)

        for t in range(t_start, t_end):
            offset = t * (num_heads * head_dim) + h_idx * head_dim + d_offsets
            k_val = tl.load(k_ptr + offset, mask=d_mask, other=0.0)
            v_val = tl.load(v_ptr + offset, mask=d_mask, other=0.0)

            accum_k += k_val
            accum_v += v_val

        inv_len = 1.0 / tl.maximum(actual_chunk_len, 1.0)
        out_offset = chunk_id * (num_heads * head_dim) + h_idx * head_dim + d_offsets

        tl.store(k_out_ptr + out_offset, accum_k * inv_len, mask=d_mask)
        tl.store(v_out_ptr + out_offset, accum_v * inv_len, mask=d_mask)


def hca_chunk_pool_cuda(
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused HCA chunk pooling kernel.
    """
    orig_shape = k.shape
    seq_len = orig_shape[1] if k.ndim == 4 else orig_shape[0]
    num_heads = orig_shape[2] if k.ndim == 4 else orig_shape[1]
    head_dim = orig_shape[3] if k.ndim == 4 else orig_shape[2]

    num_chunks = (seq_len + chunk_size - 1) // chunk_size

    if not k.is_cuda or not HAS_TRITON:
        from ..core.hierarchical_compression import HCAChunkCompressor
        comp = HCAChunkCompressor(hidden_dim=num_heads * head_dim, chunk_size=chunk_size)
        return comp.compress_chunk(k if k.ndim == 4 else k.unsqueeze(0), v if v.ndim == 4 else v.unsqueeze(0))

    k_flat = k.view(seq_len, num_heads, head_dim).contiguous()
    v_flat = v.view(seq_len, num_heads, head_dim).contiguous()

    k_out = torch.empty((num_chunks, num_heads, head_dim), device=k.device, dtype=k.dtype)
    v_out = torch.empty((num_chunks, num_heads, head_dim), device=v.device, dtype=v.dtype)

    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    grid = (num_chunks, num_heads)

    _hca_chunk_mean_pool_kernel[grid](
        k_flat,
        v_flat,
        k_out,
        v_out,
        total_tokens=seq_len,
        chunk_size=chunk_size,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )

    if k.ndim == 4:
        return k_out.unsqueeze(0), v_out.unsqueeze(0)
    return k_out, v_out
