"""
NTK-Aware Dynamic RoPE Scaling for Context Extrapolation (4K to 128K+).
"""

import functools
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

    def compute_freqs(self, seq_len: int, device: torch.device, start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cos and sin rotary embedding matrices for positions start .. seq_len - 1.
        Returns (cos, sin) of shape [seq_len - start, dim]
        """
        return _rope_table(self.dim, self._theta(seq_len), start, seq_len, torch.device(device))

    def compute_freqs_at(self, positions: Tuple[int, ...], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos, sin [len(positions), dim] for one position per row (a batched decode step), each row's base set by
        its own sequence length position + 1, as compute_freqs(position + 1, start=position) would."""
        thetas = tuple(self._theta(p + 1) for p in positions)
        return _rope_rows(self.dim, tuple(positions), thetas, torch.device(device))

    def _theta(self, seq_len: int) -> float:
        if seq_len > self.max_position_embeddings:
            # Dynamic NTK alpha scaling equation:
            # alpha = scale ** (dim / (dim - 2))
            scale = seq_len / self.max_position_embeddings
            alpha = scale ** (self.dim / (self.dim - 2.0))
            return float(self.base_theta * alpha)
        return float(self.base_theta)


@functools.lru_cache(maxsize=256)
def _rope_table(dim: int, theta: float, start: int, end: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [end - start, dim]; cached so every layer of a forward step shares one table (callers never mutate it).
    Built outside inference mode: a cached inference tensor could not be used later by code that needs autograd."""
    with torch.inference_mode(False), torch.no_grad():
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(start, end, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq) # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1) # [seq_len, dim]
        return torch.cos(emb), torch.sin(emb)

@functools.lru_cache(maxsize=64)
def _rope_rows(dim: int, positions: Tuple[int, ...], thetas: Tuple[float, ...],
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode(False), torch.no_grad():
        t = torch.tensor(positions, dtype=torch.float32, device=device)[:, None]
        theta = torch.tensor(thetas, dtype=torch.float32, device=device)[:, None]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freqs = t * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)

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
