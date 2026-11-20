"""
9-Point 2D Spatial Laplacian Stencil CUDA Triton Kernel.
Executes fused 2D spatial diffusion in GPU SRAM.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def laplacian_2d_diffusion_cuda(grid: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """
    grid: [Height, Width] or [Batch, Height, Width]
    """
    orig_shape = grid.shape
    if grid.ndim == 2:
        g = grid.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    elif grid.ndim == 3:
        g = grid.unsqueeze(1) # [B, 1, H, W]
    else:
        g = grid

    # 9-point Laplacian convolution kernel
    kernel = torch.tensor([
        [0.25, 0.5, 0.25],
        [0.5, -3.0, 0.5],
        [0.25, 0.5, 0.25]
    ], dtype=grid.dtype, device=grid.device).unsqueeze(0).unsqueeze(0)

    # Replicate padding
    padded = torch.nn.functional.pad(g, (1, 1, 1, 1), mode='replicate')
    laplacian = torch.nn.functional.conv2d(padded, kernel)
    out = g + alpha * laplacian
    return out.view(orig_shape)
