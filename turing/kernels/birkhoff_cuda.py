"""
Fused in-SRAM Birkhoff Polytope Manifold Projection CUDA Triton Kernel.
Keeps mixing matrix in GPU registers/SRAM across all Sinkhorn-Knopp iterations.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _birkhoff_project_kernel(
        mat_ptr,
        out_ptr,
        N: tl.constexpr,
        NUM_ITERS: tl.constexpr,
        EPS: tl.constexpr,
        stride_b,
        stride_m,
        stride_n
    ):
        pid = tl.program_id(0)
        # Load N x N matrix into SRAM
        offs_m = tl.arange(0, N)
        offs_n = tl.arange(0, N)
        ptrs = mat_ptr + (pid * stride_b) + (offs_m[:, None] * stride_m) + (offs_n[None, :] * stride_n)
        
        m = tl.load(ptrs)
        max_val = tl.max(m, axis=1)[:, None]
        p = tl.exp(m - max_val) + EPS

        for _ in range(NUM_ITERS):
            # Row norm
            row_sum = tl.sum(p, axis=1)[:, None] + EPS
            p = p / row_sum
            # Col norm
            col_sum = tl.sum(p, axis=0)[None, :] + EPS
            p = p / col_sum

        out_ptrs = out_ptr + (pid * stride_b) + (offs_m[:, None] * stride_m) + (offs_n[None, :] * stride_n)
        tl.store(out_ptrs, p)

def birkhoff_project_cuda(matrix: torch.Tensor, num_iterations: int = 20, eps: float = 1e-6) -> torch.Tensor:
    if not HAS_TRITON or not matrix.is_cuda:
        # Fallback to PyTorch or C++
        p = torch.exp(matrix - matrix.max(dim=-1, keepdim=True)[0]) + eps
        for _ in range(num_iterations):
            p = p / (p.sum(dim=-1, keepdim=True) + eps)
            p = p / (p.sum(dim=-2, keepdim=True) + eps)
        return p

    orig_shape = matrix.shape
    n = orig_shape[-1]
    if n not in [2, 4, 8, 16]:
        # Fallback for non-power-of-2 / larger matrices
        p = torch.exp(matrix - matrix.max(dim=-1, keepdim=True)[0]) + eps
        for _ in range(num_iterations):
            p = p / (p.sum(dim=-1, keepdim=True) + eps)
            p = p / (p.sum(dim=-2, keepdim=True) + eps)
        return p

    matrix_2d = matrix.view(-1, n, n).contiguous()
    batch_size = matrix_2d.shape[0]
    out = torch.empty_like(matrix_2d)

    grid = (batch_size,)
    _birkhoff_project_kernel[grid](
        matrix_2d,
        out,
        N=n,
        NUM_ITERS=num_iterations,
        EPS=eps,
        stride_b=matrix_2d.stride(0),
        stride_m=matrix_2d.stride(1),
        stride_n=matrix_2d.stride(2)
    )
    return out.view(orig_shape)
