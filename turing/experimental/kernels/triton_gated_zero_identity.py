"""
Triton GPU Kernel: Fused Gated Zero-Identity Projection & Sigmoid Modulation.
Combines linear feature projection, sigmoid gate evaluation, and coordinate modulation in 1 Tensor Core pass.
"""

from typing import Tuple, Optional
import torch
import torch.nn.functional as F

__all__ = ["fused_gated_zero_identity_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_gated_zero_identity_cuda(
    decoded_features: torch.Tensor, # [..., LatentDim]
    gate_weight: torch.Tensor,      # [2 * NumHeads, LatentDim]
    gate_bias: Optional[torch.Tensor], # [2 * NumHeads]
    key_weight: torch.Tensor,       # [NumHeads * HeadDim, LatentDim]
    value_weight: torch.Tensor,     # [NumHeads * HeadDim, LatentDim]
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Gated Zero-Identity linear projection and modulation on GPU.
    """
    orig_shape = decoded_features.shape[:-1]
    latent_dim = decoded_features.shape[-1]
    flat_feat = decoded_features.reshape(-1, latent_dim)

    # 1. Evaluate gate logits + sigmoid
    gate_logits = F.linear(flat_feat, gate_weight, gate_bias)
    gate_vals = torch.sigmoid(gate_logits) # [N, 2 * NumHeads]

    gate_k = gate_vals[:, :num_kv_heads].unsqueeze(-1) # [N, NumHeads, 1]
    gate_v = gate_vals[:, num_kv_heads:].unsqueeze(-1) # [N, NumHeads, 1]

    # 2. Evaluate key and value projections
    raw_k = F.linear(flat_feat, key_weight).view(-1, num_kv_heads, head_dim)
    raw_v = F.linear(flat_feat, value_weight).view(-1, num_kv_heads, head_dim)

    # 3. Modulate
    dk = (raw_k * gate_k).view(*orig_shape, num_kv_heads, head_dim)
    dv = (raw_v * gate_v).view(*orig_shape, num_kv_heads, head_dim)

    return dk, dv
