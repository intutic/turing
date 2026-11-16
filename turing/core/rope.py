"""
Rotary position embeddings with every RoPE scaling type HuggingFace transformers implements
(default, linear, dynamic NTK, YaRN, LongRoPE, Llama-3, proportional), computed the way transformers computes them
so a converted model reproduces HuggingFace's logits (tests/test_hf_parity.py).
"""

import functools
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

ROPE_TYPES = ("default", "linear", "dynamic", "yarn", "longrope", "llama3", "proportional")


def _freeze(params: Optional[dict]) -> tuple:
    """A hashable form of a rope-parameters dict (lists become tuples) for the table cache."""
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in (params or {}).items()))


def rope_inv_freq(dim: int, base: float, max_pos: int, params: tuple, seq_len: Optional[int] = None
                  ) -> Tuple[torch.Tensor, float]:
    """inv_freq [dim // 2] (fp32, CPU) and the attention factor that scales cos and sin, as transformers'
    ROPE_INIT_FUNCTIONS compute them. dim is the rotary dimension; seq_len matters for dynamic and longrope only."""
    p = dict(params)
    kind = p.get("rope_type", p.get("type", "default"))
    exponent = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
    factor = p.get("factor")
    if kind == "default":
        return 1.0 / (base ** exponent), 1.0
    if kind == "linear":
        return 1.0 / (base ** exponent) / factor, 1.0
    if kind == "dynamic":
        L = max(seq_len or max_pos, max_pos)
        b = base * ((factor * L / max_pos) - (factor - 1)) ** (dim / (dim - 2))
        return 1.0 / (b ** exponent), 1.0
    if kind == "yarn":
        old = p["original_max_position_embeddings"]
        factor = factor if factor is not None else max_pos / old

        def mscale(scale, m=1.0):
            return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
        att = p.get("attention_factor")
        if att is None:
            m, m_all = p.get("mscale"), p.get("mscale_all_dim")
            att = float(mscale(factor, m) / mscale(factor, m_all)) if (m and m_all) else mscale(factor)
        beta_fast, beta_slow = p.get("beta_fast") or 32, p.get("beta_slow") or 1

        def corr_dim(rot):
            return (dim * math.log(old / (rot * 2 * math.pi))) / (2 * math.log(base))
        low, high = corr_dim(beta_fast), corr_dim(beta_slow)
        if p.get("truncate", True):
            low, high = math.floor(low), math.ceil(high)
        low, high = max(low, 0), min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        extrap = 1 - ramp
        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        inv = (1.0 / (factor * pos_freqs)) * (1 - extrap) + (1.0 / pos_freqs) * extrap
        return inv, att
    if kind == "longrope":
        old = p["original_max_position_embeddings"]
        factor = factor if factor is not None else max_pos / old
        att = p.get("attention_factor")
        if att is None:
            att = 1.0 if factor <= 1.0 else math.sqrt(1 + math.log(factor) / math.log(old))
        ext = torch.tensor(p["long_factor"] if (seq_len and seq_len > old) else p["short_factor"], dtype=torch.float32)
        return 1.0 / (ext * base ** exponent), att
    if kind == "llama3":
        inv = 1.0 / (base ** exponent)
        low, high, old = p["low_freq_factor"], p["high_freq_factor"], p["original_max_position_embeddings"]
        wavelen = 2 * math.pi / inv
        scaled = torch.where(wavelen > old / low, inv / factor, inv)
        smooth = (old / wavelen - low) / (high - low)
        medium = ~(wavelen < old / high) * ~(wavelen > old / low)
        return torch.where(medium, (1 - smooth) * scaled / factor + smooth * scaled, scaled), 1.0
    if kind == "proportional":  # dim is the full head_dim here; the non-rotated part gets zero frequency
        angles = int(p.get("partial_rotary_factor", 1.0) * dim // 2)
        inv = 1.0 / (base ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / dim))
        inv = torch.cat((inv, torch.zeros(dim // 2 - angles)))
        return inv / (factor or 1.0), 1.0
    raise ValueError(f"RoPE scaling type '{kind}' is not supported (known: {', '.join(ROPE_TYPES)})")


class RotaryEmbedding(nn.Module):
    """cos/sin tables for one attention layer. dim is the rotary dimension (head_dim, or less with a partial rotary
    factor); params is the HuggingFace rope-parameters dict without rope_theta (None for plain RoPE)."""

    def __init__(self, dim: int = 128, max_position_embeddings: int = 4096, base_theta: float = 10000.0,
                 params: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base_theta = float(base_theta)
        self.params = _freeze(params)
        p = dict(self.params)
        self.kind = p.get("rope_type", p.get("type", "default"))
        rope_inv_freq(dim, self.base_theta, max_position_embeddings, self.params)  # validate the type now
        self._original = p.get("original_max_position_embeddings", max_position_embeddings)
        self._device_tables = {}

    def _variant(self, seq_len: int) -> Optional[int]:
        """The seq_len that selects inv_freq: dynamic and longrope depend on it, the other types do not."""
        if self.kind == "dynamic":
            return max(seq_len, self.max_position_embeddings)
        if self.kind == "longrope":
            return self._original + 1 if seq_len > self._original else None
        return None

    def compute_freqs(self, seq_len: int, device: torch.device, start: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos and sin [seq_len - start, dim] for positions start .. seq_len - 1 (seq_len is the sequence's total
        length, which picks the frequencies of dynamic and longrope scaling, as transformers does per forward)."""
        return _rope_table(self.dim, self.base_theta, self.max_position_embeddings, self.params,
                           self._variant(seq_len), start, seq_len, torch.device(device))

    def compute_freqs_device(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """cos, sin [B, dim] for one position per row from a device tensor, with no host sync (safe in a CUDA graph).
        Each row's frequencies follow its own length position + 1, as compute_freqs(position + 1, start=position)."""
        pos = positions.float()[:, None]
        dev = positions.device
        if self.kind == "dynamic":
            L = torch.clamp(pos + 1, min=float(self.max_position_embeddings))
            f = dict(self.params)["factor"]
            base = self.base_theta * ((f * L / self.max_position_embeddings) - (f - 1)) ** (self.dim / (self.dim - 2))
            inv = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=dev).float() / self.dim))
            att = 1.0
        elif self.kind == "longrope":
            (short, att), (long, _) = self._device(dev, None), self._device(dev, self._original + 1)
            inv = torch.where(pos + 1 > self._original, long, short)
        else:
            inv, att = self._device(dev, None)
        freqs = pos * inv
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb) * att, torch.sin(emb) * att

    def _device(self, device: torch.device, variant: Optional[int]):
        key = (device, variant)
        if key not in self._device_tables:
            inv, att = rope_inv_freq(self.dim, self.base_theta, self.max_position_embeddings, self.params, variant)
            self._device_tables[key] = (inv.to(device), att)
        return self._device_tables[key]


@functools.lru_cache(maxsize=256)
def _rope_table(dim: int, base: float, max_pos: int, params: tuple, variant: Optional[int], start: int, end: int,
                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [end - start, dim]; cached so every layer of a forward step shares one table (callers never mutate it).
    Built outside inference mode: a cached inference tensor could not be used later by code that needs autograd."""
    with torch.inference_mode(False), torch.no_grad():
        inv_freq, att = rope_inv_freq(dim, base, max_pos, params, variant)
        t = torch.arange(start, end, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq.to(device))  # [seq_len, dim // 2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]
        return torch.cos(emb) * att, torch.sin(emb) * att


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
    Applies rotary position embeddings to query and key states; when cos/sin cover fewer dimensions than the heads
    (a partial rotary factor), only the leading dimensions rotate and the rest pass through, as in GPT-NeoX and Phi.
    q, k: [Batch, Heads, SeqLen, HeadDim]
    cos, sin: [SeqLen, RotDim] or [Batch, 1, SeqLen, RotDim]
    """
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, SeqLen, RotDim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    r = cos.shape[-1]
    if r < q.shape[-1]:
        q_rot, q_pass, k_rot, k_pass = q[..., :r], q[..., r:], k[..., :r], k[..., r:]
        return (torch.cat(((q_rot * cos) + (rotate_half(q_rot) * sin), q_pass), dim=-1),
                torch.cat(((k_rot * cos) + (rotate_half(k_rot) * sin), k_pass), dim=-1))
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
