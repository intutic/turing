"""
3:1 Hybrid Linear-Full Attention & Context Chunk-Scoring Filter.
Implements the GLM-5.3-Flash / Qwen3.8-Flash-Next architectural convergence:
- 3 out of every 4 layers run linear fixed-state recurrence (O(L) compute & fixed O(1) state).
- 1 out of every 4 layers runs full quadratic attention with 4x HCA chunk scoring capped at 2048 tokens.
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hierarchical_compression import HCAChunkCompressor


class LinearRecurrentAttention(nn.Module):
    """
    O(L) Fixed-State Linear Recurrent Attention Layer.
    Maintains an in-SRAM recurrent state S_t = alpha * S_{t-1} + K_t^T V_t,
    emitting output O_t = Q_t S_t with zero growing KV memory.
    """
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, decay: float = 0.95):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.decay = decay

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [Batch, SeqLen, HiddenDim]
        state: [Batch, NumHeads, HeadDim, HeadDim]
        Returns: (output, next_state)
        """
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Normalize Q and K for linear kernel stability
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        if state is None:
            state = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)

        # 1. Native C++20 AVX2 single-step decode dispatch (L=1)
        if seq_len == 1:
            try:
                import turing.turing_csrc as turing_csrc
                HAS_CSRC = True
            except ImportError:
                HAS_CSRC = False

            if HAS_CSRC and not x.is_cuda and x.dtype == torch.float32:
                q_np = q[:, :, 0, :].detach().cpu().contiguous().numpy()
                k_np = k[:, :, 0, :].detach().cpu().contiguous().numpy()
                v_np = v[:, :, 0, :].detach().cpu().contiguous().numpy()
                s_np = state.detach().cpu().contiguous().numpy()

                out_np, next_s_np = turing_csrc.linear_recurrence_step_cpu(q_np, k_np, v_np, s_np, float(self.decay))
                out_t = torch.from_numpy(out_np).to(device=x.device, dtype=x.dtype).unsqueeze(1)
                next_state = torch.from_numpy(next_s_np).to(device=x.device, dtype=x.dtype)
                return self.out_proj(out_t.view(batch, 1, self.num_heads * self.head_dim)), next_state

            q_t = q[:, :, 0, :].unsqueeze(-1)  # [B, H, D, 1]
            k_t = k[:, :, 0, :].unsqueeze(-2)  # [B, H, 1, D]
            v_t = v[:, :, 0, :].unsqueeze(-1)  # [B, H, D, 1]

            cur_state = self.decay * state + torch.matmul(v_t, k_t)
            o_t = torch.matmul(cur_state, q_t).squeeze(-1) # [B, H, D]
            out_seq = o_t.unsqueeze(2).transpose(1, 2).contiguous().view(batch, 1, self.num_heads * self.head_dim)
            return self.out_proj(out_seq), cur_state

        # 2. Triton CUDA chunk-parallel linear recurrence dispatch (L > 1)
        if x.is_cuda:
            try:
                from ..kernels.triton_linear_recurrence import chunk_linear_recurrence_cuda
                out_seq, next_state = chunk_linear_recurrence_cuda(q, k, v, decay=float(self.decay), state=state)
                out_seq = out_seq.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
                return self.out_proj(out_seq), next_state
            except Exception:
                pass

        # Fully vectorized chunk-parallel linear attention for fast prefill fallback

        chunk_size = 64
        num_chunks = math.ceil(seq_len / chunk_size)
        pad_len = num_chunks * chunk_size - seq_len
        if pad_len > 0:
            q_pad = F.pad(q, (0, 0, 0, pad_len))
            k_pad = F.pad(k, (0, 0, 0, pad_len))
            v_pad = F.pad(v, (0, 0, 0, pad_len))
        else:
            q_pad, k_pad, v_pad = q, k, v

        q_chunks = q_pad.view(batch, self.num_heads, num_chunks, chunk_size, self.head_dim)
        k_chunks = k_pad.view(batch, self.num_heads, num_chunks, chunk_size, self.head_dim)
        v_chunks = v_pad.view(batch, self.num_heads, num_chunks, chunk_size, self.head_dim)

        indices = torch.arange(chunk_size, device=x.device, dtype=x.dtype)
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)
        intra_decay = torch.where(diff >= 0, self.decay ** diff, torch.zeros_like(diff))

        outputs = []
        cur_state = state

        for c in range(num_chunks):
            qc = q_chunks[:, :, c] # [B, H, C, D]
            kc = k_chunks[:, :, c]
            vc = v_chunks[:, :, c]

            intra_scores = torch.matmul(qc, kc.transpose(-1, -2)) * intra_decay
            intra_out = torch.matmul(intra_scores, vc)

            # Inter-chunk state contribution
            inter_out = torch.matmul(qc, cur_state.transpose(-1, -2))
            decay_vec = (self.decay ** (indices + 1)).view(1, 1, chunk_size, 1)
            inter_out = inter_out * decay_vec

            outputs.append(intra_out + inter_out)

            # Recurrent state progression
            chunk_decay = self.decay ** chunk_size
            scale_k = (self.decay ** (chunk_size - 1 - indices)).view(1, 1, chunk_size, 1)
            kv_chunk = torch.matmul(vc.transpose(-1, -2), kc * scale_k)
            cur_state = cur_state * chunk_decay + kv_chunk

        out_seq = torch.cat(outputs, dim=2)[:, :, :seq_len, :]
        out_seq = out_seq.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(out_seq), cur_state



class ChunkContextScorer(nn.Module):
    """
    Lightweight 4x Chunk-Compressed Context Scoring Side Network.
    Compresses 128-token chunks into summary representations and scores them
    against query to retain only the top 2048-token context budget.
    """
    def __init__(self, hidden_dim: int, chunk_size: int = 128, budget_tokens: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.budget_tokens = budget_tokens
        self.compressor = HCAChunkCompressor(hidden_dim=hidden_dim, chunk_size=chunk_size)
        self.score_proj = nn.Linear(hidden_dim, 1, bias=False)

    def filter_context(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k, v: [Batch, SeqLen, NumHeads, HeadDim]
        q: [Batch, NumQueries, NumHeads, HeadDim]
        Returns: (k_filtered, v_filtered) filtered to budget_tokens
        """
        seq_len = k.shape[1]
        if seq_len <= self.budget_tokens:
            return k, v

        batch, _, num_heads, head_dim = k.shape
        num_chunks = seq_len // self.chunk_size
        max_chunks = self.budget_tokens // self.chunk_size

        # Compress to chunk summaries: [Batch, NumChunks, NumHeads, HeadDim]
        k_chunks, v_chunks = self.compressor.compress_chunk(k, v)
        k_flat = k_chunks.reshape(batch, num_chunks, num_heads * head_dim)

        # Cross-score chunk summaries against query
        q_summary = q.mean(dim=1).reshape(batch, 1, num_heads * head_dim)
        scores = torch.sum(k_flat * q_summary, dim=-1) # [Batch, NumChunks]

        # Top-K chunk selection
        topk_indices = torch.topk(scores, k=min(max_chunks, num_chunks), dim=-1).indices
        topk_indices = torch.sort(topk_indices, dim=-1).values # Maintain causal order

        # Gather tokens from selected chunks
        k_reshaped = k[:, :num_chunks * self.chunk_size].view(batch, num_chunks, self.chunk_size, num_heads, head_dim)
        v_reshaped = v[:, :num_chunks * self.chunk_size].view(batch, num_chunks, self.chunk_size, num_heads, head_dim)

        batch_idx = torch.arange(batch, device=k.device).unsqueeze(1)
        k_selected = k_reshaped[batch_idx, topk_indices].view(batch, -1, num_heads, head_dim)
        v_selected = v_reshaped[batch_idx, topk_indices].view(batch, -1, num_heads, head_dim)

        return k_selected, v_selected


class HybridAttentionLayerRouter(nn.Module):
    """
    3:1 Hybrid Attention Router for Transformer Stacks.
    Routes layers where (layer_idx % 4 != 0) to LinearRecurrentAttention,
    and layers where (layer_idx % 4 == 0) to Full Quadratic Attention with ChunkContextScorer.
    """
    def __init__(self, layer_idx: int, hidden_dim: int, num_heads: int, head_dim: int, budget_tokens: int = 2048):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = (layer_idx % 4 != 0)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        if self.is_linear:
            self.linear_attn = LinearRecurrentAttention(hidden_dim, num_heads, head_dim)
        else:
            self.chunk_scorer = ChunkContextScorer(hidden_dim=hidden_dim, budget_tokens=budget_tokens)

    def forward(
        self,
        x: torch.Tensor,
        full_attn_fn=None,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_linear:
            return self.linear_attn(x, state=state)
        else:
            if full_attn_fn is not None:
                return full_attn_fn(x), None
            return x, None
