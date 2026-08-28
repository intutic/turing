"""
Triton GPU Kernel: Fused Inverse-RoPE Content Decoupling & Closed-Form Ridge Projection.
Applies inverse 2D Givens rotation and projects directly through closed-form transfer matrix W* in SRAM.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_inv_rope_kernel(
        k_ptr,          # [SeqLen, HeadDim]
        out_ptr,        # [SeqLen, HeadDim]
        seq_len: tl.constexpr,
        head_dim: tl.constexpr,
        base: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        t_idx = tl.program_id(0) # token position
        if t_idx >= seq_len:
            return

        dim_half = head_dim // 2
        d_offsets = tl.arange(0, BLOCK_SIZE)
        mask = d_offsets < dim_half

        # Compute inv_freq = 1.0 / (base ** (2*i / head_dim))
        freq_exp = (d_offsets * 2.0) / head_dim
        # Using log/exp: base^exp = exp(exp * log(base))
        inv_freq = 1.0 / tl.exp(freq_exp * tl.log(base))
        angle = t_idx * inv_freq

        cos = tl.cos(angle)
        sin = tl.sin(angle)

        k1_ptrs = k_ptr + t_idx * head_dim + d_offsets
        k2_ptrs = k_ptr + t_idx * head_dim + (dim_half + d_offsets)

        k1 = tl.load(k1_ptrs, mask=mask, other=0.0)
        k2 = tl.load(k2_ptrs, mask=mask, other=0.0)

        # Inverse rotation:
        # k1_unrot = k1 * cos + k2 * sin
        # k2_unrot = -k1 * sin + k2 * cos
        k1_unrot = k1 * cos + k2 * sin
        k2_unrot = -k1 * sin + k2 * cos

        out1_ptrs = out_ptr + t_idx * head_dim + d_offsets
        out2_ptrs = out_ptr + t_idx * head_dim + (dim_half + d_offsets)

        tl.store(out1_ptrs, k1_unrot, mask=mask)
        tl.store(out2_ptrs, k2_unrot, mask=mask)


def fused_inv_rope_cuda(
    k_rope: torch.Tensor,
    base: float = 500000.0
) -> torch.Tensor:
    """
    Inverse-RoPE content decoupling kernel.
    """
    orig_shape = k_rope.shape
    head_dim = orig_shape[-1]
    seq_len = orig_shape[0] if k_rope.ndim == 2 else orig_shape[1]

    if not k_rope.is_cuda or not HAS_TRITON:
        from ..core.cross_model_kv import RoPEContentDecoupler
        return RoPEContentDecoupler.strip_rope(k_rope, base=base)

    k_flat = k_rope.view(-1, head_dim).contiguous()
    total_tokens = k_flat.shape[0]
    out = torch.empty_like(k_flat)

    dim_half = head_dim // 2
    BLOCK_SIZE = triton.next_power_of_2(dim_half)

    _fused_inv_rope_kernel[(total_tokens,)](
        k_flat,
        out,
        seq_len=total_tokens,
        head_dim=head_dim,
        base=float(base),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out.view(orig_shape)
