"""
In-SRAM Fused Triton GPU kernels for variable-length token segment reduction (Mean-Pool and CLS).
Replaces naive Python loops and device-to-host .item() synchronization with single-pass GPU kernels.
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional

__all__ = [
    "fused_segment_mean_pool_cuda",
    "fused_segment_cls_pool_cuda",
]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_segment_mean_pool_kernel(
        hidden_ptr,
        cu_seqlens_ptr,
        output_ptr,
        num_seqs,
        hidden_dim,
        stride_h_tok,
        stride_h_dim,
        stride_o_seq,
        stride_o_dim,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        dim_block = tl.program_id(1)

        if seq_idx >= num_seqs:
            return

        start_tok = tl.load(cu_seqlens_ptr + seq_idx)
        end_tok = tl.load(cu_seqlens_ptr + seq_idx + 1)
        seq_len = end_tok - start_tok

        dim_offsets = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < hidden_dim

        acc = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

        for tok_offset in range(0, seq_len, BLOCK_TOKENS):
            cur_tok = start_tok + tok_offset
            for sub_i in range(BLOCK_TOKENS):
                sub_tok = cur_tok + sub_i
                is_valid = sub_tok < end_tok
                if is_valid:
                    h_ptrs = hidden_ptr + sub_tok * stride_h_tok + dim_offsets * stride_h_dim
                    val = tl.load(h_ptrs, mask=dim_mask, other=0.0)
                    acc += val.to(tl.float32)

        scale = 1.0 / tl.maximum(seq_len.to(tl.float32), 1.0)
        mean_val = acc * scale

        out_ptrs = output_ptr + seq_idx * stride_o_seq + dim_offsets * stride_o_dim
        tl.store(out_ptrs, mean_val.to(output_ptr.dtype.element_ty), mask=dim_mask)


    @triton.jit
    def _fused_segment_cls_pool_kernel(
        hidden_ptr,
        cu_seqlens_ptr,
        output_ptr,
        num_seqs,
        hidden_dim,
        stride_h_tok,
        stride_h_dim,
        stride_o_seq,
        stride_o_dim,
        BLOCK_DIM: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        dim_block = tl.program_id(1)

        if seq_idx >= num_seqs:
            return

        start_tok = tl.load(cu_seqlens_ptr + seq_idx)
        dim_offsets = dim_block * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < hidden_dim

        h_ptrs = hidden_ptr + start_tok * stride_h_tok + dim_offsets * stride_h_dim
        val = tl.load(h_ptrs, mask=dim_mask, other=0.0)

        out_ptrs = output_ptr + seq_idx * stride_o_seq + dim_offsets * stride_o_dim
        tl.store(out_ptrs, val, mask=dim_mask)


def fused_segment_mean_pool_cuda(
    hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_seqs: Optional[int] = None,
) -> torch.Tensor:
    """
    Fused variable-length token segment mean-pooling.
    
    Args:
        hidden: [total_tokens, hidden_dim] tensor of packed sequence representations.
        cu_seqlens: [num_seqs + 1] cumulative token count boundaries.
        num_seqs: Number of sequences (inferred from cu_seqlens.shape[0] - 1 if None).
        
    Returns:
        [num_seqs, hidden_dim] mean-pooled sequence embeddings.
    """
    if num_seqs is None:
        num_seqs = cu_seqlens.shape[0] - 1

    hidden_dim = hidden.shape[-1]
    output = torch.empty((num_seqs, hidden_dim), device=hidden.device, dtype=hidden.dtype)

    if not HAS_TRITON or not hidden.is_cuda:
        # High-performance PyTorch vectorized fallback using index_add_ (zero Python per-seq loops)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).unsqueeze(-1).to(hidden.dtype).clamp(min=1.0)
        
        # Expand sequence indices for every token in [total_tokens]
        total_tokens = hidden.shape[0]
        seq_idx = torch.repeat_interleave(
            torch.arange(num_seqs, device=hidden.device),
            cu_seqlens[1:] - cu_seqlens[:-1],
            output_size=total_tokens,
        )
        
        # Zero-sync vectorized scatter sum
        out_sum = torch.zeros((num_seqs, hidden_dim), device=hidden.device, dtype=hidden.dtype)
        out_sum.index_add_(0, seq_idx, hidden)
        return out_sum / lengths

    BLOCK_DIM = min(1024, triton.next_power_of_2(hidden_dim))
    BLOCK_TOKENS = 16
    grid = (num_seqs, triton.cdiv(hidden_dim, BLOCK_DIM))

    _fused_segment_mean_pool_kernel[grid](
        hidden,
        cu_seqlens,
        output,
        num_seqs,
        hidden_dim,
        hidden.stride(0),
        hidden.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_DIM=BLOCK_DIM,
        BLOCK_TOKENS=BLOCK_TOKENS,
    )
    return output


def fused_segment_cls_pool_cuda(
    hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_seqs: Optional[int] = None,
) -> torch.Tensor:
    """
    Fused variable-length token segment [CLS] token extraction (token 0 of each segment).
    
    Args:
        hidden: [total_tokens, hidden_dim] tensor of packed sequence representations.
        cu_seqlens: [num_seqs + 1] cumulative token count boundaries.
        num_seqs: Number of sequences (inferred from cu_seqlens.shape[0] - 1 if None).
        
    Returns:
        [num_seqs, hidden_dim] CLS sequence embeddings.
    """
    if num_seqs is None:
        num_seqs = cu_seqlens.shape[0] - 1

    hidden_dim = hidden.shape[-1]
    output = torch.empty((num_seqs, hidden_dim), device=hidden.device, dtype=hidden.dtype)

    if not HAS_TRITON or not hidden.is_cuda:
        # PyTorch index gather
        cls_indices = cu_seqlens[:-1]
        return hidden[cls_indices]

    BLOCK_DIM = min(1024, triton.next_power_of_2(hidden_dim))
    grid = (num_seqs, triton.cdiv(hidden_dim, BLOCK_DIM))

    _fused_segment_cls_pool_kernel[grid](
        hidden,
        cu_seqlens,
        output,
        num_seqs,
        hidden_dim,
        hidden.stride(0),
        hidden.stride(1),
        output.stride(0),
        output.stride(1),
        BLOCK_DIM=BLOCK_DIM,
    )
    return output
