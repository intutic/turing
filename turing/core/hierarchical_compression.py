"""
Hierarchical Sequence-Chunk Compression & Cross-Layer KV Sharing (DeepSeek V4 & Gemma 4).
Implements:
1. HCA (Heavily Compressed Attention): m'=128 token chunk pooling for global background attention on Huge (512) pages.
2. CSA (Compressed Sparse Attention): m=4 token chunk pooling with top-k block-sparse mask on Medium (64) pages.
3. Cross-Layer KV Sharing: Alternating sliding-window and full layers sharing KV projections (Gemma 4).
"""

from typing import List, Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

class HCAChunkCompressor(nn.Module):
    """
    Heavily Compressed Attention (HCA) Sequence Compressor (DeepSeek V4):
    Compresses sequence chunks of length m'=128 into 1 summary KV vector per chunk.
    For a 512-token Huge Page, this produces exactly 4 summary KV tokens (128x compression).
    """
    def __init__(self, hidden_dim: int, chunk_size: int = 128):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        # Lightweight learned attention pooling kernel over the chunk
        self.pool_proj = nn.Linear(hidden_dim, 1, bias=False)

    def compress_chunk(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k, v: [Batch, SeqLen, NumHeads, HeadDim]
        Returns: [Batch, NumChunks, NumHeads, HeadDim] where NumChunks = ceil(SeqLen / chunk_size)
        """
        batch, seq_len, num_heads, head_dim = k.shape
        if HAS_CSRC and batch == 1 and not k.is_cuda:
            k_cpu = k[0].detach().to(torch.float32).cpu().contiguous().numpy()
            v_cpu = v[0].detach().to(torch.float32).cpu().contiguous().numpy()
            k_out = turing_csrc.hca_chunk_pool(k_cpu, self.chunk_size)
            v_out = turing_csrc.hca_chunk_pool(v_cpu, self.chunk_size)
            return (
                torch.from_numpy(k_out).unsqueeze(0).to(device=k.device, dtype=k.dtype),
                torch.from_numpy(v_out).unsqueeze(0).to(device=v.device, dtype=v.dtype)
            )
        batch, seq_len, num_heads, head_dim = k.shape
        pad_len = (self.chunk_size - (seq_len % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        new_seq_len = k.shape[1]
        num_chunks = new_seq_len // self.chunk_size

        # Reshape into chunks: [Batch, NumChunks, ChunkSize, NumHeads, HeadDim]
        k_chunks = k.view(batch, num_chunks, self.chunk_size, num_heads, head_dim)
        v_chunks = v.view(batch, num_chunks, self.chunk_size, num_heads, head_dim)

        # Compute chunk-level attention pooling weights
        # [Batch, NumChunks, ChunkSize, NumHeads, 1]
        attn_logits = self.pool_proj(k_chunks)
        attn_weights = F.softmax(attn_logits, dim=2)

        # Weighted sum: [Batch, NumChunks, NumHeads, HeadDim]
        k_summary = (k_chunks * attn_weights).sum(dim=2)
        v_summary = (v_chunks * attn_weights).sum(dim=2)

        return k_summary, v_summary


class CSAChunkCompressor(nn.Module):
    """
    Compressed Sparse Attention (CSA) Chunk Compressor (DeepSeek V4):
    Compresses sequence in m=4 token chunks and selects top-k most relevant blocks for query attention.
    For a 64-token Medium Page, this produces 16 block descriptors.
    """
    def __init__(self, hidden_dim: int, chunk_size: int = 4):
        super().__init__()
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim

    def compress_blocks(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k, v: [Batch, SeqLen, NumHeads, HeadDim]
        Returns mean-pooled chunk descriptors: [Batch, NumBlocks, NumHeads, HeadDim]
        """
        batch, seq_len, num_heads, head_dim = k.shape
        pad_len = (self.chunk_size - (seq_len % self.chunk_size)) % self.chunk_size
        if pad_len > 0:
            k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

        new_seq_len = k.shape[1]
        num_blocks = new_seq_len // self.chunk_size

        k_blocks = k.view(batch, num_blocks, self.chunk_size, num_heads, head_dim).mean(dim=2)
        v_blocks = v.view(batch, num_blocks, self.chunk_size, num_heads, head_dim).mean(dim=2)

        return k_blocks, v_blocks

    def select_topk_blocks(
        self,
        q: torch.Tensor,
        k_blocks: torch.Tensor,
        top_k: int = 8
    ) -> torch.Tensor:
        """
        q: [Batch, NumQueryHeads, HeadDim]
        k_blocks: [Batch, NumBlocks, NumKVHeads, HeadDim]
        Returns boolean block mask: [Batch, NumBlocks]
        """
        batch, num_blocks, kv_heads, head_dim = k_blocks.shape
        # Score query against block key centroids
        # [Batch, NumBlocks, KVHeads]
        scores = torch.einsum('bhd,bnhd->bnh', q, k_blocks).max(dim=-1)[0] # [Batch, NumBlocks]

        actual_k = min(top_k, num_blocks)
        _, topk_indices = torch.topk(scores, k=actual_k, dim=-1) # [Batch, actual_k]

        mask = torch.zeros(batch, num_blocks, dtype=torch.bool, device=q.device)
        mask.scatter_(dim=1, index=topk_indices, value=True)
        return mask


class CrossLayerKVSharingManager:
    """
    Gemma 4 Cross-Layer KV Tensor Sharing Scheme:
    Later transformer layers reuse KV projections computed in earlier layers:
    - Sliding-window layers reuse KV from the most recent non-shared sliding-window layer.
    - Full-attention layers reuse KV from the most recent non-shared full-attention layer.
    Saves ~50% of the active KV cache memory footprint.
    """
    def __init__(self, num_layers: int, num_shared_layers: int = 16, sliding_window_ratio: int = 4):
        self.num_layers = num_layers
        self.num_shared = num_shared_layers
        self.sw_ratio = sliding_window_ratio

        # Map each layer to the source layer that computes its KV projections
        self.kv_source_layer_map: Dict[int, int] = {}
        self.is_layer_shared: Dict[int, bool] = {}

        # First (num_layers - num_shared_layers) layers compute their own KV
        dense_cutoff = max(1, num_layers - num_shared_layers)

        for l in range(num_layers):
            if l < dense_cutoff:
                self.kv_source_layer_map[l] = l
                self.is_layer_shared[l] = False
            else:
                is_sliding = (l % self.sw_ratio != 0)
                # Find most recent earlier layer of the same attention type
                source_layer = l - 1
                while source_layer >= 0:
                    src_is_sliding = (source_layer % self.sw_ratio != 0)
                    if src_is_sliding == is_sliding and not self.is_layer_shared.get(source_layer, False):
                        break
                    source_layer -= 1

                if source_layer < 0:
                    source_layer = 0

                self.kv_source_layer_map[l] = source_layer
                self.is_layer_shared[l] = True

    def get_source_layer(self, layer_idx: int) -> int:
        return self.kv_source_layer_map.get(layer_idx, layer_idx)

    def is_shared(self, layer_idx: int) -> bool:
        return self.is_layer_shared.get(layer_idx, False)
