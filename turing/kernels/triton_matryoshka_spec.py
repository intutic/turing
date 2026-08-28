"""
Triton GPU Kernel: Fused Nested Matryoshka Sliced Draft Projection & Candidate Speculation.
Computes O(W * V) logits directly in GPU SRAM without slicing/materializing intermediate global tensors.
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _matryoshka_sliced_gemv_kernel(
        x_ptr,           # [BatchSize, HiddenDim]
        w_ptr,           # [VocabSize, HiddenDim]
        out_ptr,         # [BatchSize, VocabSize]
        slice_width: tl.constexpr,
        hidden_dim: tl.constexpr,
        vocab_size: tl.constexpr,
        BLOCK_SIZE_V: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_v = tl.program_id(axis=0)
        pid_b = tl.program_id(axis=1)

        v_offsets = pid_v * BLOCK_SIZE_V + tl.arange(0, BLOCK_SIZE_V)
        v_mask = v_offsets < vocab_size

        x_base_ptr = x_ptr + pid_b * hidden_dim
        out_base_ptr = out_ptr + pid_b * vocab_size

        accum = tl.zeros((BLOCK_SIZE_V,), dtype=tl.float32)

        for k in range(0, slice_width, BLOCK_SIZE_K):
            k_offsets = k + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < slice_width

            x_vals = tl.load(x_base_ptr + k_offsets, mask=k_mask, other=0.0)

            # Load tile of weights [BLOCK_SIZE_V, BLOCK_SIZE_K]
            w_ptrs = w_ptr + (v_offsets[:, None] * hidden_dim + k_offsets[None, :])
            w_vals = tl.load(w_ptrs, mask=v_mask[:, None] & k_mask[None, :], other=0.0)

            accum += tl.sum(w_vals * x_vals[None, :], axis=1)

        tl.store(out_base_ptr + v_offsets, accum, mask=v_mask)


def matryoshka_sliced_gemv_triton(
    x: torch.Tensor,          # [..., HiddenDim]
    w: torch.Tensor,          # [VocabSize, HiddenDim]
    slice_width: int
) -> torch.Tensor:
    """
    Executes in-SRAM Matryoshka sliced draft projection on CUDA devices.
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    vocab_size = w.shape[0]

    effective_w = min(slice_width, hidden_dim)
    x_flat = x.view(-1, hidden_dim).contiguous()
    batch_size = x_flat.shape[0]

    if not x.is_cuda or not HAS_TRITON:
        # Fallback to PyTorch sliced linear
        return torch.matmul(x_flat[:, :effective_w], w[:, :effective_w].t()).view(*orig_shape[:-1], vocab_size)

    out = torch.empty((batch_size, vocab_size), device=x.device, dtype=torch.float32)

    BLOCK_SIZE_V = 64
    BLOCK_SIZE_K = 128

    grid = (triton.cdiv(vocab_size, BLOCK_SIZE_V), batch_size)

    _matryoshka_sliced_gemv_kernel[grid](
        x_flat,
        w,
        out,
        slice_width=effective_w,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        BLOCK_SIZE_V=BLOCK_SIZE_V,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return out.to(x.dtype).view(*orig_shape[:-1], vocab_size)

