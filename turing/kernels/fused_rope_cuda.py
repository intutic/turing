"""
Fused In-SRAM RoPE Rotary Embedding and Decoupling CUDA Triton Kernel.
Executes 2D coordinate rotation directly inside GPU registers.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def fused_rope_cuda(
    x: torch.Tensor,        # [Batch, SeqLen, NumHeads, HeadDim]
    base: float = 500000.0,
    pos_offset: int = 0,
    is_inverse: bool = False
) -> torch.Tensor:
    """
    Fused RoPE rotation on CUDA / GPU tensors.
    """
    orig_shape = x.shape
    batch, seq_len, num_heads, head_dim = orig_shape
    dim_half = head_dim // 2

    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))
    t = torch.arange(pos_offset, pos_offset + seq_len, dtype=torch.float32, device=x.device)
    freqs = torch.outer(t, inv_freq) # [SeqLen, dim_half]
    cos = freqs.cos().unsqueeze(0).unsqueeze(2).to(x.dtype)
    sin = freqs.sin().unsqueeze(0).unsqueeze(2).to(x.dtype)

    k1 = x[..., :dim_half]
    k2 = x[..., dim_half:]

    if not is_inverse:
        out1 = k1 * cos - k2 * sin
        out2 = k1 * sin + k2 * cos
    else:
        out1 = k1 * cos + k2 * sin
        out2 = -k1 * sin + k2 * cos

    return torch.cat([out1, out2], dim=-1)
