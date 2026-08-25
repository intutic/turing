"""
Compressed Convolutional Attention (CCA) & Layer-Wise Budgeting (ZAYA1-8B & Poolside Laguna XS.2).
Implements:
1. Compressed Convolutional Attention: Performs attention directly in compressed latent space with 1D depthwise sequence convolution on Q and K.
2. Layer-wise Attention Head Budgeting: Configures varying query head counts per layer.
"""

from typing import Optional, Tuple, Dict, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerwiseHeadBudgeter:
    """
    Poolside Laguna XS.2 Layer-Wise Attention Head Budgeter:
    Allocates different numbers of query heads per transformer layer while keeping KV heads fixed.
    For example: 8 query heads for local sliding-window layers, and 6 query heads for global full-attention layers.
    """
    def __init__(self, num_layers: int, kv_heads: int = 8, sliding_heads: int = 8, global_heads: int = 6, sliding_ratio: int = 4):
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.sliding_heads = sliding_heads
        self.global_heads = global_heads
        self.sliding_ratio = sliding_ratio

        self.layer_query_heads: Dict[int, int] = {}
        for l in range(num_layers):
            if l % sliding_ratio == 0:
                self.layer_query_heads[l] = global_heads
            else:
                self.layer_query_heads[l] = sliding_heads

    def get_query_heads(self, layer_idx: int) -> int:
        return self.layer_query_heads.get(layer_idx, self.sliding_heads)


class CompressedConvolutionalAttention(nn.Module):
    """
    ZAYA1-8B Compressed Convolutional Attention (CCA):
    Compresses hidden states into latent space, performs 1D depthwise sequence convolution
    on latent Q and K to restore local spatial context, and computes attention directly in the latent space.
    """
    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = 512,
        num_heads: int = 8,
        kernel_size: int = 3
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.kernel_size = kernel_size

        # Latent Down-projections
        self.q_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.k_down = nn.Linear(hidden_dim, latent_dim, bias=False)
        self.v_down = nn.Linear(hidden_dim, latent_dim, bias=False)

        # 1D Depthwise Sequence Convolutions on Latent Q and K
        self.q_conv1d = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1, # Causal padding
            groups=latent_dim,
            bias=False
        )
        self.k_conv1d = nn.Conv1d(
            in_channels=latent_dim,
            out_channels=latent_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1, # Causal padding
            groups=latent_dim,
            bias=False
        )

        # Output Up-projection from latent space
        self.out_proj = nn.Linear(latent_dim, hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [Batch, SeqLen, HiddenDim]
        Returns: [Batch, SeqLen, HiddenDim]
        """
        batch, seq_len, _ = x.shape

        # Step 1: Compress to latent representations
        q_lat = self.q_down(x) # [Batch, SeqLen, LatentDim]
        k_lat = self.k_down(x)
        v_lat = self.v_down(x)

        # Step 2: Apply 1D depthwise causal convolution on Q and K
        # Transpose for Conv1d: [Batch, LatentDim, SeqLen]
        q_conv = self.q_conv1d(q_lat.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        k_conv = self.k_conv1d(k_lat.transpose(1, 2))[..., :seq_len].transpose(1, 2)

        # Step 3: Reshape for multi-head attention in latent space
        # [Batch, NumHeads, SeqLen, HeadDim]
        q = q_conv.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_conv.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_lat.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Step 4: Compute Attention in Latent Space
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale # [Batch, NumHeads, SeqLen, SeqLen]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            # Default causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(1), float("-inf"))

        attn_probs = F.softmax(scores, dim=-1)
        # [Batch, NumHeads, SeqLen, HeadDim]
        attn_out = torch.matmul(attn_probs, v)

        # Step 5: Merge heads and project up to hidden dimension
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, self.latent_dim)
        return self.out_proj(attn_out)
