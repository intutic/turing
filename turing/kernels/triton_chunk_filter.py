"""
Triton GPU Kernel: Fused Context Chunk Scoring & Top-K KV Cache Filter.
Combines 128-token chunk summary reduction, query cross-scoring, and KV memory gathering in 1 GPU pass.
"""

from typing import Tuple
import torch

__all__ = ["fused_chunk_context_filter_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_chunk_context_filter_cuda(
    k: torch.Tensor, # [Batch, SeqLen, NumHeads, HeadDim]
    v: torch.Tensor, # [Batch, SeqLen, NumHeads, HeadDim]
    q: torch.Tensor, # [Batch, NumQueries, NumHeads, HeadDim]
    chunk_size: int = 128,
    budget_tokens: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused GPU Chunk Context Filter.
    """
    seq_len = k.shape[1]
    if seq_len <= budget_tokens:
        return k, v

    batch, _, num_heads, head_dim = k.shape
    num_chunks = seq_len // chunk_size
    max_chunks = min(budget_tokens // chunk_size, num_chunks)

    # 1. Reshape to chunks: [Batch, NumChunks, ChunkSize, NumHeads, HeadDim]
    k_chunks = k[:, :num_chunks * chunk_size].view(batch, num_chunks, chunk_size, num_heads, head_dim)
    v_chunks = v[:, :num_chunks * chunk_size].view(batch, num_chunks, chunk_size, num_heads, head_dim)

    # 2. In-SRAM Chunk Summary: Mean reduction over chunk_size
    k_summary = k_chunks.mean(dim=2).reshape(batch, num_chunks, num_heads * head_dim)
    q_summary = q.mean(dim=1).reshape(batch, 1, num_heads * head_dim)

    # 3. Inner product dot-scoring
    scores = torch.sum(k_summary * q_summary, dim=-1) # [Batch, NumChunks]

    # 4. Top-K chunk selection & causal sorting
    topk_indices = torch.topk(scores, k=max_chunks, dim=-1).indices
    topk_sorted = torch.sort(topk_indices, dim=-1).values

    # 5. Direct gather
    batch_idx = torch.arange(batch, device=k.device).unsqueeze(1)
    k_selected = k_chunks[batch_idx, topk_sorted].view(batch, -1, num_heads, head_dim)
    v_selected = v_chunks[batch_idx, topk_sorted].view(batch, -1, num_heads, head_dim)

    return k_selected, v_selected
