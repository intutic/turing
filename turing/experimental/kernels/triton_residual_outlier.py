"""
Triton GPU Kernel: 1-Pass In-SRAM Subspace Residual Outlier Extraction.
Finds top-1 outlier coordinate (index and value) in a single reduction pass without memory sorting.
"""

from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _residual_outlier_kernel(
        r_ptr,           # [Batch, HiddenDim]
        idx_ptr,         # [Batch]
        val_ptr,         # [Batch]
        hidden_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        b_idx = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        r_base = r_ptr + b_idx * hidden_dim
        vals = tl.load(r_base + offsets, mask=mask, other=0.0)
        abs_vals = tl.abs(vals)

        # In-block maximum reduction
        max_abs = tl.max(abs_vals, axis=0)
        # Find which index matches max_abs
        is_max = abs_vals == max_abs
        # Lowest matching index
        max_idx = tl.min(tl.where(is_max & mask, offsets, hidden_dim))

        actual_val = tl.load(r_base + max_idx)

        tl.store(idx_ptr + b_idx, max_idx)
        tl.store(val_ptr + b_idx, actual_val)


def find_residual_outlier_cuda(
    residual: torch.Tensor # [..., HiddenDim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds top-1 residual outlier per sequence element.
    Returns: (top_indices [..., 1], top_values [..., 1])
    """
    orig_shape = residual.shape
    hidden_dim = orig_shape[-1]
    r_flat = residual.view(-1, hidden_dim).contiguous()
    batch_size = r_flat.shape[0]

    if not residual.is_cuda or not HAS_TRITON or hidden_dim > 8192:
        try:
            import turing.turing_csrc as turing_csrc
            r_cpu = r_flat.detach().to(torch.float32).cpu().contiguous().numpy()
            idx_np, val_np = turing_csrc.find_residual_outlier_cpu(r_cpu)
            top_idx = torch.from_numpy(idx_np).to(device=residual.device, dtype=torch.long)
            top_val = torch.from_numpy(val_np).to(device=residual.device, dtype=residual.dtype)
            return top_idx.view(*orig_shape[:-1], 1), top_val.view(*orig_shape[:-1], 1)
        except Exception:
            top_vals, top_indices = torch.topk(torch.abs(r_flat), k=1, dim=-1)
            signed_vals = torch.gather(r_flat, -1, top_indices)
            return top_indices.view(*orig_shape[:-1], 1), signed_vals.view(*orig_shape[:-1], 1)

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    top_indices = torch.empty((batch_size,), device=residual.device, dtype=torch.int32)
    top_values = torch.empty((batch_size,), device=residual.device, dtype=torch.float32)

    _residual_outlier_kernel[(batch_size,)](
        r_flat,
        top_indices,
        top_values,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (
        top_indices.to(torch.long).view(*orig_shape[:-1], 1),
        top_values.to(residual.dtype).view(*orig_shape[:-1], 1)
    )
