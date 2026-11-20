"""
Batched PagedAttention Decode GEMV Triton Kernel for Continuous Batching.
Processes multiple concurrent requests (BatchSize >= 1) in a single GPU kernel launch
by looking up non-contiguous physical pages from StaticPagedKVPool via 2D block tables.
Includes Split-K Paged Flash-Decoding for long context acceleration.
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

__all__ = ["batched_paged_decode_gemv_cuda", "batched_paged_decode_split_k_gemv_cuda"]

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
    def _paged_decode_gemv_kernel(
        Q_ptr,              # [B, H, D]
        K_pool_ptr,         # [NumBlocks, BlockSize, H, D]
        V_pool_ptr,         # [NumBlocks, BlockSize, H, D]
        BlockTable_ptr,     # [B, MaxBlocks]
        SeqLens_ptr,        # [B]
        Out_ptr,            # [B, H, D]
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_vb, stride_vs, stride_vh, stride_vd,
        stride_tb, stride_tm,
        stride_ob, stride_oh, stride_od,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)

        seq_len = tl.load(SeqLens_ptr + b_idx)
        if seq_len <= 0:
            return

        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, BLOCK_SIZE)

        # Load query vector [D]
        q_ptr = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_d * stride_qd
        q_val = tl.load(q_ptr, mask=(offs_d < D), other=0.0).to(tl.float32)

        # Online softmax state
        max_score = float("-inf")
        sum_exp = 0.0
        acc_v = tl.zeros((D,), dtype=tl.float32)

        for block_idx in range(MAX_BLOCKS):
            block_start_tok = block_idx * BLOCK_SIZE
            if block_start_tok < seq_len:
                # Load physical block ID from table
                p_block_id = tl.load(BlockTable_ptr + b_idx * stride_tb + block_idx * stride_tm)
                if p_block_id >= 0:
                    tok_offsets = block_start_tok + offs_s
                    tok_mask = tok_offsets < seq_len

                    # Load K tile for this block [BLOCK_SIZE, D]
                    k_ptrs = (K_pool_ptr + p_block_id * stride_kb + offs_s[:, None] * stride_ks +
                              h_idx * stride_kh + offs_d[None, :] * stride_kd)
                    k_tile = tl.load(k_ptrs, mask=tok_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

                    # Dot product: q @ K^T -> [BLOCK_SIZE]
                    scores = tl.sum(k_tile * q_val[None, :], axis=1) * scale
                    scores = tl.where(tok_mask, scores, float("-inf"))

                    block_max = tl.max(scores, axis=0)
                    new_max = tl.maximum(max_score, block_max)

                    # Rescale previous sum and accumulator
                    alpha = tl.exp(max_score - new_max)
                    sum_exp = sum_exp * alpha

                    # Load V tile for this block [BLOCK_SIZE, D]
                    v_ptrs = (V_pool_ptr + p_block_id * stride_vb + offs_s[:, None] * stride_vs +
                              h_idx * stride_vh + offs_d[None, :] * stride_vd)
                    v_tile = tl.load(v_ptrs, mask=tok_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

                    weights = tl.exp(scores - new_max)
                    weights = tl.where(tok_mask, weights, 0.0)

                    sum_exp += tl.sum(weights, axis=0)
                    acc_v = acc_v * alpha + tl.sum(weights[:, None] * v_tile, axis=0)
                    max_score = new_max

        # Final normalization
        inv_sum = 1.0 / tl.maximum(sum_exp, 1e-6)
        out_val = acc_v * inv_sum

        out_ptrs = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, out_val.to(tl.float32), mask=(offs_d < D))


    @triton.jit
    def _paged_decode_split_k_gemv_kernel(
        Q_ptr,              # [B, H, D]
        K_pool_ptr,         # [NumBlocks, BlockSize, H, D]
        V_pool_ptr,         # [NumBlocks, BlockSize, H, D]
        BlockTable_ptr,     # [B, MaxBlocks]
        SeqLens_ptr,        # [B]
        MidOut_ptr,         # [B, H, NumSplits, D]
        MidLse_ptr,         # [B, H, NumSplits]
        scale,
        stride_qb, stride_qh, stride_qd,
        stride_kb, stride_ks, stride_kh, stride_kd,
        stride_vb, stride_vs, stride_vh, stride_vd,
        stride_tb, stride_tm,
        stride_mob, stride_moh, stride_mos, stride_mod,
        stride_mlb, stride_mlh, stride_mls,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)
        split_idx = tl.program_id(2)

        seq_len = tl.load(SeqLens_ptr + b_idx)
        if seq_len <= 0:
            return

        total_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        blocks_per_split = (total_blocks + NUM_SPLITS - 1) // NUM_SPLITS
        start_block = split_idx * blocks_per_split
        end_block = tl.minimum(start_block + blocks_per_split, total_blocks)

        if start_block >= total_blocks:
            ml_ptr = MidLse_ptr + b_idx * stride_mlb + h_idx * stride_mlh + split_idx * stride_mls
            tl.store(ml_ptr, float("-inf"))
            return

        offs_d = tl.arange(0, D)
        offs_s = tl.arange(0, BLOCK_SIZE)

        q_ptr = Q_ptr + b_idx * stride_qb + h_idx * stride_qh + offs_d * stride_qd
        q_val = tl.load(q_ptr, mask=(offs_d < D), other=0.0).to(tl.float32)

        max_score = float("-inf")
        sum_exp = 0.0
        acc_v = tl.zeros((D,), dtype=tl.float32)

        for block_idx in range(start_block, end_block):
            block_start_tok = block_idx * BLOCK_SIZE
            p_block_id = tl.load(BlockTable_ptr + b_idx * stride_tb + block_idx * stride_tm)
            if p_block_id >= 0:
                tok_offsets = block_start_tok + offs_s
                tok_mask = tok_offsets < seq_len

                k_ptrs = (K_pool_ptr + p_block_id * stride_kb + offs_s[:, None] * stride_ks +
                          h_idx * stride_kh + offs_d[None, :] * stride_kd)
                k_tile = tl.load(k_ptrs, mask=tok_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

                scores = tl.sum(k_tile * q_val[None, :], axis=1) * scale
                scores = tl.where(tok_mask, scores, float("-inf"))

                block_max = tl.max(scores, axis=0)
                new_max = tl.maximum(max_score, block_max)
                alpha = tl.exp(max_score - new_max)
                sum_exp = sum_exp * alpha

                v_ptrs = (V_pool_ptr + p_block_id * stride_vb + offs_s[:, None] * stride_vs +
                          h_idx * stride_vh + offs_d[None, :] * stride_vd)
                v_tile = tl.load(v_ptrs, mask=tok_mask[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

                weights = tl.exp(scores - new_max)
                weights = tl.where(tok_mask, weights, 0.0)

                sum_exp += tl.sum(weights, axis=0)
                acc_v = acc_v * alpha + tl.sum(weights[:, None] * v_tile, axis=0)
                max_score = new_max

        lse = max_score + tl.log(tl.maximum(sum_exp, 1e-12))
        
        mo_ptr = MidOut_ptr + b_idx * stride_mob + h_idx * stride_moh + split_idx * stride_mos + offs_d * stride_mod
        tl.store(mo_ptr, acc_v, mask=(offs_d < D))

        ml_ptr = MidLse_ptr + b_idx * stride_mlb + h_idx * stride_mlh + split_idx * stride_mls
        tl.store(ml_ptr, lse)


    @triton.jit
    def _paged_decode_split_k_reduce_kernel(
        MidOut_ptr,         # [B, H, NumSplits, D]
        MidLse_ptr,         # [B, H, NumSplits]
        Out_ptr,            # [B, H, D]
        stride_mob, stride_moh, stride_mos, stride_mod,
        stride_mlb, stride_mlh, stride_mls,
        stride_ob, stride_oh, stride_od,
        H: tl.constexpr,
        D: tl.constexpr,
        NUM_SPLITS: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        h_idx = tl.program_id(1)

        offs_d = tl.arange(0, D)

        global_max_lse = float("-inf")
        for s in range(NUM_SPLITS):
            ml_ptr = MidLse_ptr + b_idx * stride_mlb + h_idx * stride_mlh + s * stride_mls
            s_lse = tl.load(ml_ptr)
            global_max_lse = tl.maximum(global_max_lse, s_lse)

        if global_max_lse == float("-inf"):
            out_ptrs = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_d * stride_od
            tl.store(out_ptrs, 0.0, mask=(offs_d < D))
            return

        global_sum_exp = 0.0
        acc_v = tl.zeros((D,), dtype=tl.float32)

        for s in range(NUM_SPLITS):
            ml_ptr = MidLse_ptr + b_idx * stride_mlb + h_idx * stride_mlh + s * stride_mls
            s_lse = tl.load(ml_ptr)
            if s_lse > float("-inf"):
                weight = tl.exp(s_lse - global_max_lse)
                global_sum_exp += weight

                mo_ptr = MidOut_ptr + b_idx * stride_mob + h_idx * stride_moh + s * stride_mos + offs_d * stride_mod
                part_v = tl.load(mo_ptr, mask=(offs_d < D), other=0.0)
                acc_v += part_v * weight

        out_val = acc_v / tl.maximum(global_sum_exp, 1e-12)
        out_ptrs = Out_ptr + b_idx * stride_ob + h_idx * stride_oh + offs_d * stride_od
        tl.store(out_ptrs, out_val, mask=(offs_d < D))


def batched_paged_decode_gemv_cuda(
    q: torch.Tensor,                    # [B, H, 1, D] or [B, H, D]
    k_pool: torch.Tensor,               # [NumBlocks, BlockSize, H, D]
    v_pool: torch.Tensor,               # [NumBlocks, BlockSize, H, D]
    block_tables: torch.Tensor,         # [B, MaxBlocks] (int32)
    seq_lens: torch.Tensor,             # [B] (int32)
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    Executes batched continuous PagedAttention decode GEMV across all concurrent requests.
    Auto-dispatches to Split-K when sequence length >= 2048 for optimal memory bandwidth.
    Falls back to PyTorch when Triton or CUDA is unavailable.
    """
    if q.dim() == 4:
        q_3d = q.squeeze(2)  # [B, H, D]
    else:
        q_3d = q

    batch_size, num_heads, head_dim = q_3d.shape
    num_blocks, block_size, _, _ = k_pool.shape
    max_blocks = block_tables.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Auto-dispatch to Split-K for long contexts
    max_seq_len = int(seq_lens.max().item()) if seq_lens.numel() > 0 else 0
    if q.is_cuda and HAS_TRITON and max_seq_len >= 2048:
        return batched_paged_decode_split_k_gemv_cuda(
            q, k_pool, v_pool, block_tables, seq_lens, scale=scale, num_splits=4
        )

    if not q.is_cuda or not HAS_TRITON:
        # PyTorch Reference Implementation
        out = torch.zeros_like(q_3d)
        for b in range(batch_size):
            s_len = int(seq_lens[b].item())
            if s_len <= 0:
                continue
            b_tables = block_tables[b].tolist()
            needed_blocks = (s_len + block_size - 1) // block_size

            # Gather K and V for this request
            k_chunks = []
            v_chunks = []
            for blk_idx in range(needed_blocks):
                p_id = b_tables[blk_idx]
                if p_id >= 0:
                    k_chunks.append(k_pool[p_id, :, :, :])
                    v_chunks.append(v_pool[p_id, :, :, :])

            if not k_chunks:
                continue

            k_all = torch.cat(k_chunks, dim=0)[:s_len]  # [SeqLen, H, D]
            v_all = torch.cat(v_chunks, dim=0)[:s_len]  # [SeqLen, H, D]

            # [H, 1, D] @ [H, D, SeqLen] -> [H, 1, SeqLen]
            q_b = q_3d[b].unsqueeze(1)  # [H, 1, D]
            k_perm = k_all.permute(1, 2, 0)  # [H, D, SeqLen]
            v_perm = v_all.permute(1, 0, 2)  # [H, SeqLen, D]

            attn_weights = torch.matmul(q_b, k_perm) * scale  # [H, 1, SeqLen]
            attn_probs = F.softmax(attn_weights, dim=-1)
            ctx = torch.matmul(attn_probs, v_perm).squeeze(1)  # [H, D]
            out[b] = ctx

        return out.unsqueeze(2) if q.dim() == 4 else out

    out = torch.empty((batch_size, num_heads, head_dim), device=q.device, dtype=q.dtype)

    grid = (batch_size, num_heads)

    _paged_decode_gemv_kernel[grid](
        q_3d.contiguous(),
        k_pool.contiguous(),
        v_pool.contiguous(),
        block_tables.contiguous().to(torch.int32),
        seq_lens.contiguous().to(torch.int32),
        out,
        scale,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        H=num_heads,
        D=head_dim,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=max_blocks,
        num_stages=2,
        num_warps=4
    )

    return out.unsqueeze(2) if q.dim() == 4 else out


def batched_paged_decode_split_k_gemv_cuda(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
    num_splits: int = 4
) -> torch.Tensor:
    """
    Executes Split-K Paged Flash-Decoding on CUDA with parallel partial reductions.
    """
    if q.dim() == 4:
        q_3d = q.squeeze(2)
    else:
        q_3d = q

    batch_size, num_heads, head_dim = q_3d.shape
    num_blocks, block_size, _, _ = k_pool.shape
    max_blocks = block_tables.shape[1]

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if not q.is_cuda or not HAS_TRITON:
        # Reference PyTorch fallback
        return batched_paged_decode_gemv_cuda(q, k_pool, v_pool, block_tables, seq_lens, scale=scale)

    mid_out = torch.empty((batch_size, num_heads, num_splits, head_dim), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((batch_size, num_heads, num_splits), device=q.device, dtype=torch.float32)
    out = torch.empty((batch_size, num_heads, head_dim), device=q.device, dtype=q.dtype)

    grid_split = (batch_size, num_heads, num_splits)

    _paged_decode_split_k_gemv_kernel[grid_split](
        q_3d.contiguous(),
        k_pool.contiguous(),
        v_pool.contiguous(),
        block_tables.contiguous().to(torch.int32),
        seq_lens.contiguous().to(torch.int32),
        mid_out,
        mid_lse,
        scale,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2), k_pool.stride(3),
        v_pool.stride(0), v_pool.stride(1), v_pool.stride(2), v_pool.stride(3),
        block_tables.stride(0), block_tables.stride(1),
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        H=num_heads,
        D=head_dim,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=max_blocks,
        NUM_SPLITS=num_splits,
        num_stages=2,
        num_warps=4
    )

    grid_reduce = (batch_size, num_heads)
    _paged_decode_split_k_reduce_kernel[grid_reduce](
        mid_out,
        mid_lse,
        out,
        mid_out.stride(0), mid_out.stride(1), mid_out.stride(2), mid_out.stride(3),
        mid_lse.stride(0), mid_lse.stride(1), mid_lse.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        H=num_heads,
        D=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=2
    )

    return out.unsqueeze(2) if q.dim() == 4 else out

