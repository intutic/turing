"""
Triton Fused Hexagonal Topological Codebook Quantization Kernel.
Computes in-SRAM cosine similarity against hexagonal codebook prototypes (total_cells <= 64)
and performs in-register minimum distance search for zero-allocation BMU quantization.
"""

from typing import Tuple
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _hex_quant_bmu_kernel(
        X_ptr,          # [N, Dim] (normalized)
        Codebook_ptr,   # [NumCells, Dim] (normalized)
        Bmu_out_ptr,    # [N] (int64)
        Dist_out_ptr,   # [N] (float32)
        N,
        stride_xb, stride_xd,
        stride_cb, stride_cd,
        DIM: tl.constexpr,
        NUM_CELLS: tl.constexpr
    ):
        pid = tl.program_id(0)
        if pid >= N:
            return

        offs_d = tl.arange(0, DIM)
        x_vec = tl.load(X_ptr + pid * stride_xb + offs_d * stride_xd).to(tl.float32)

        min_dist = 1e30
        best_bmu = 0

        for c in range(NUM_CELLS):
            cb_vec = tl.load(Codebook_ptr + c * stride_cb + offs_d * stride_cd).to(tl.float32)
            sim = tl.sum(x_vec * cb_vec)
            dist = 1.0 - sim
            if dist < min_dist:
                min_dist = dist
                best_bmu = c

        tl.store(Bmu_out_ptr + pid, best_bmu)
        tl.store(Dist_out_ptr + pid, min_dist)


def hex_quant_cuda(
    x: torch.Tensor,        # [N, Dim]
    codebook: torch.Tensor # [NumCells, Dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes in-SRAM hexagonal BMU search on CUDA.
    """
    if not HAS_TRITON:
        x_norm = F.normalize(x, p=2, dim=-1)
        sim = torch.matmul(x_norm, codebook.t())
        dists = 1.0 - sim
        min_dists, bmu_indices = torch.min(dists, dim=-1)
        return bmu_indices, min_dists

    n, dim = x.shape
    num_cells = codebook.shape[0]

    x_norm = F.normalize(x.float(), p=2, dim=-1)
    bmu_out = torch.empty(n, device=x.device, dtype=torch.int64)
    dist_out = torch.empty(n, device=x.device, dtype=torch.float32)

    dim_pow2 = triton.next_power_of_2(dim)

    grid = (n,)
    _hex_quant_bmu_kernel[grid](
        x_norm, codebook.float(),
        bmu_out, dist_out,
        n,
        x_norm.stride(0), x_norm.stride(1),
        codebook.stride(0), codebook.stride(1),
        DIM=dim_pow2,
        NUM_CELLS=num_cells,
        num_warps=2
    )

    return bmu_out, dist_out
