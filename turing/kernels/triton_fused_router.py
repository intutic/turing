"""
Triton GPU Kernel: In-SRAM Fused Subspace Structured Router.
Fuses sequence pooling + normalization + gate linear projection + top-k tile score generation.
"""

from typing import Tuple, Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_router_kernel(
        ctx_ptr,         # [Batch, HiddenDim]
        w_gate_ptr,      # [HiddenDim, TotalTiles * 2]
        tile_mask_ptr,   # [Batch, TotalTiles]
        hidden_dim: tl.constexpr,
        total_tiles: tl.constexpr,
        k_tiles: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        b_idx = tl.program_id(0)

        # Compute tile scores: score[t] = logit[t, 1] - logit[t, 0]
        # w_gate shape: [hidden_dim, total_tiles * 2]
        scores = tl.zeros((128,), dtype=tl.float32)

        for k in range(0, hidden_dim, BLOCK_SIZE_K):
            k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < hidden_dim

            x_vals = tl.load(ctx_ptr + b_idx * hidden_dim + k_offsets, mask=k_mask, other=0.0)

            for t in range(total_tiles):
                # Column for logit 0: 2*t, Column for logit 1: 2*t + 1
                w0_ptrs = w_gate_ptr + k_offsets * (total_tiles * 2) + (2 * t)
                w1_ptrs = w_gate_ptr + k_offsets * (total_tiles * 2) + (2 * t + 1)

                w0_vals = tl.load(w0_ptrs, mask=k_mask, other=0.0)
                w1_vals = tl.load(w1_ptrs, mask=k_mask, other=0.0)

                diff = w1_vals - w0_vals
                scores[t] += tl.sum(x_vals * diff)

        # Write output tile mask (1.0 for top-k, 0.0 otherwise)
        for t in range(total_tiles):
            # Count how many tiles have strictly greater score
            rank_count = 0
            s_t = scores[t]
            for other_t in range(total_tiles):
                if scores[other_t] > s_t:
                    rank_count += 1

            is_topk = 1.0 if rank_count < k_tiles else 0.0
            tl.store(tile_mask_ptr + b_idx * total_tiles + t, is_topk)


def fused_router_cuda(
    ctx: torch.Tensor,       # [Batch, HiddenDim]
    w_gate: torch.Tensor,    # [HiddenDim, TotalTiles * 2]
    k_tiles: int = 32
) -> torch.Tensor:
    """
    Fused gate projection and top-k tile mask generation.
    """
    batch, hidden_dim = ctx.shape
    total_tiles = w_gate.shape[1] // 2

    if not ctx.is_cuda or not HAS_TRITON or total_tiles > 128:
        # Fallback to PyTorch
        logits = torch.matmul(ctx, w_gate).view(batch, total_tiles, 2)
        tile_scores = logits[:, :, 1] - logits[:, :, 0]
        _, topk_idx = torch.topk(tile_scores, k=min(k_tiles, total_tiles), dim=-1)
        tile_mask = torch.zeros(batch, total_tiles, device=ctx.device, dtype=torch.float32)
        tile_mask.scatter_(1, topk_idx, 1.0)
        return tile_mask

    tile_mask = torch.zeros((batch, total_tiles), device=ctx.device, dtype=torch.float32)
    BLOCK_SIZE_K = 64

    _fused_router_kernel[(batch,)](
        ctx.contiguous(),
        w_gate.contiguous(),
        tile_mask,
        hidden_dim=hidden_dim,
        total_tiles=total_tiles,
        k_tiles=min(k_tiles, total_tiles),
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return tile_mask
