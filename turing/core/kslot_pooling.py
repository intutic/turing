"""
k-Slot Symmetric Cache Pooling and Gated Zero-Identity modules.

This module implements k-slot attention-based cache pooling adapted from
kvloom's CachePooler (arXiv:2608.20617 Eq. 3-4), plus gated zero-identity
residual heads (Eq. 12-14) and a gate-skip policy.

Note: The KSlotCachePooler auto-dispatches to a fused Triton kernel on CUDA
when available (future integration).
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KSlotCachePooler", "GatedZeroIdentityHead", "GateSkipPolicy"]


class KSlotCachePooler(nn.Module):
    """
    Pools N-token KV caches into k learned summary slots per layer and head, making
    cross-model transfer O(1) in sequence length.
    
    Auto-dispatches to a fused Triton kernel on CUDA when available.
    """
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, num_slots: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_slots = num_slots
        
        self.queries = nn.Parameter(torch.empty(num_layers, num_kv_heads, num_slots, head_dim))
        nn.init.normal_(self.queries, std=head_dim**-0.5)

    def attention_weights(self, keys: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            keys: Tensor of shape (B, L, H, N, D)
            padding_mask: Optional boolean Tensor of shape (B, N) where True means valid token.
        """
        # Einsum details: l=num_layers, h=num_kv_heads, j=num_slots, d=head_dim, b=batch, n=sequence
        logits = torch.einsum('lhjd,blhnd->blhjn', self.queries.to(keys.dtype), keys)
        logits = logits / math.sqrt(self.head_dim)
        
        if padding_mask is not None:
            # Mask out invalid tokens with -inf
            logits = logits.masked_fill(~padding_mask[:, None, None, None, :], float('-inf'))
            
        return torch.softmax(logits, dim=-1)

    def forward(self, keys: torch.Tensor, values: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if keys.is_cuda and padding_mask is None:
            try:
                from ..kernels.triton_kslot_pool import fused_kslot_pool_cuda
                return fused_kslot_pool_cuda(keys, values, self.queries)
            except Exception:
                pass

        attn = self.attention_weights(keys, padding_mask)
        pooled_keys = torch.einsum('blhjn,blhnd->blhjd', attn, keys)
        pooled_values = torch.einsum('blhjn,blhnd->blhjd', attn, values)
        return pooled_keys, pooled_values


class GatedZeroIdentityHead(nn.Module):
    """
    Zero-initialized output heads guarantee that an untrained translator emits
    exactly zero residual, adapted from kvloom's PositionRetrieval gating.
    """
    def __init__(self, latent_dim: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.gate = nn.Linear(latent_dim, 2 * num_kv_heads)
        self.key_head = nn.Linear(latent_dim, num_kv_heads * head_dim, bias=False)
        self.value_head = nn.Linear(latent_dim, num_kv_heads * head_dim, bias=False)
        
        nn.init.zeros_(self.key_head.weight)
        nn.init.zeros_(self.value_head.weight)

    def forward(self, decoded_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies zero-initialized heads and sigmoid gating to decoded features.
        
        Args:
            decoded_features: shape (B, N, latent_dim) or (B, L, N, latent_dim)
            
        Returns:
            Tuple of K and V updates (dk, dv).
        """
        if decoded_features.is_cuda:
            try:
                from ..kernels.triton_gated_zero_identity import fused_gated_zero_identity_cuda
                return fused_gated_zero_identity_cuda(
                    decoded_features,
                    self.gate.weight,
                    self.gate.bias,
                    self.key_head.weight,
                    self.value_head.weight,
                    self.num_kv_heads,
                    self.head_dim
                )
            except Exception:
                pass

        gate_values = torch.sigmoid(self.gate(decoded_features))
        gate_k, gate_v = torch.split(gate_values, self.num_kv_heads, dim=-1)
        
        raw_k = self.key_head(decoded_features).view(*decoded_features.shape[:-1], self.num_kv_heads, self.head_dim)
        raw_v = self.value_head(decoded_features).view(*decoded_features.shape[:-1], self.num_kv_heads, self.head_dim)
        
        dk = gate_k.unsqueeze(-1) * raw_k
        dv = gate_v.unsqueeze(-1) * raw_v
        
        return dk, dv


class GateSkipPolicy:
    """
    Policy for skipping translation when residual norm is below threshold.
    """
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.considered = 0
        self.skipped = 0

    def should_skip(self, residual_norm: float) -> bool:
        self.considered += 1
        if self.threshold > 0 and residual_norm < self.threshold:
            self.skipped += 1
            return True
        return False

    @property
    def skip_rate(self) -> float:
        return self.skipped / max(self.considered, 1)

    def reset(self) -> None:
        self.considered = 0
        self.skipped = 0
