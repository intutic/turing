"""
AMD ROCm Architecture Tuning & Wavefront Heuristics for Turing Engine.
Configures optimal Triton kernel parameters (Wave32 on RDNA vs Wave64 on CDNA).
"""

from typing import Dict, Any
import torch

def get_rocm_arch_info() -> Dict[str, Any]:
    """
    Identifies AMD GPU architecture family and returns wavefront/Matrix Core specs.
    """
    if not (torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None):
        return {
            "is_rocm": False,
            "arch_family": "unknown",
            "wavefront_size": 32,
            "optimal_warps": 4,
            "has_matrix_cores": False,
        }

    device_name = torch.cuda.get_device_name(0).lower()

    # CDNA Architecture (Instinct MI100, MI200, MI210, MI250X, MI300X, MI325X)
    if any(k in device_name for k in ["mi100", "mi200", "mi210", "mi250", "mi300", "mi325", "cdna"]):
        return {
            "is_rocm": True,
            "arch_family": "CDNA",
            "wavefront_size": 64, # Wave64 for CDNA Matrix Core (MFMA)
            "optimal_warps": 8,
            "has_matrix_cores": True,
            "instruction_set": "MFMA (Matrix Fused Multiply-Add)",
        }
    
    # RDNA Architecture (Radeon RX 7900 XTX, 7900 GRE, 7800 XT, 8800 XT)
    return {
        "is_rocm": True,
        "arch_family": "RDNA",
        "wavefront_size": 32, # Wave32 mode for RDNA Dual-Issue SIMD
        "optimal_warps": 4,
        "has_matrix_cores": True,
        "instruction_set": "WMMA (Wave Matrix Multiply-Accumulate)",
    }

def get_rocm_triton_config(batch_size: int, hidden_dim: int) -> Dict[str, int]:
    """
    Returns optimal Triton launch dimensions for AMD ROCm GPUs.
    """
    arch = get_rocm_arch_info()
    if arch["arch_family"] == "CDNA":
        return {
            "BLOCK_SIZE_M": 64 if batch_size > 16 else 32,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "num_warps": 8,
            "num_stages": 2,
        }
    else:
        # RDNA3/4 Consumer
        return {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "num_warps": 4,
            "num_stages": 2,
        }
