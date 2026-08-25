"""
Compute kernels and hardware dispatch layer for Turing Engine.
"""

from .dispatch import (
    dispatch_swiglu,
    dispatch_flash_tree_attention,
    dispatch_w4a16_gemm,
    dispatch_subspace_recirculation,
    HAS_TRITON,
    HAS_CUDA,
)

__all__ = [
    "dispatch_swiglu",
    "dispatch_flash_tree_attention",
    "dispatch_w4a16_gemm",
    "dispatch_subspace_recirculation",
    "HAS_TRITON",
    "HAS_CUDA",
]
