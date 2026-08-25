"""
NTK-Aware Dynamic RoPE Scaling for Context Extrapolation (4K to 128K+).
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn

class NTKDynamicRoPEScaling(nn.Module):
    """
    NTK-Aware Dynamic RoPE (Rotary Position Embedding) Scaling.
    Dynamically scales the base frequency theta for sequence lengths exceeding max_position_embeddings,
    extending context resolution without fine-tuning or loss of high-frequency precision.
    """
    def __init__(
        self,
        dim: int = 128,
        max_position_embeddings: int = 4096,
        base_theta: float = 10000.0,
        scaling_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base_theta = base_theta
        self.scaling_factor = scaling_factor

    def compute_freqs(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cos and sin rotary embedding matrices.
        Returns (cos, sin) of shape [seq_len, dim]
        """
        if seq_len > self.max_position_embeddings:
            # Dynamic NTK alpha scaling equation:
            # alpha = scale ** (dim / (dim - 2))
            scale = seq_len / self.max_position_embeddings
            alpha = scale ** (self.dim / (self.dim - 2.0))
            theta = self.base_theta * alpha
        else:
            theta = self.base_theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq) # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]

        cos = torch.cos(emb)
        sin = torch.sin(emb)
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to query and key states.
    q, k: [Batch, Heads, SeqLen, HeadDim]
    cos, sin: [SeqLen, HeadDim] or [Batch, 1, SeqLen, HeadDim]
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0) # [1, 1, SeqLen, HeadDim]
        sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
