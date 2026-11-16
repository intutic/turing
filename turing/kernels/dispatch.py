"""
Dynamic Hardware Dispatcher for CUDA Triton, Apple Silicon MPS, and Vectorized CPU Ops.
"""

from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F

HAS_CUDA = torch.cuda.is_available()
IS_AMD_ROCM = torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None
IS_NVIDIA_CUDA = torch.cuda.is_available() and getattr(torch.version, "hip", None) is None
IS_INTEL_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()
IS_APPLE_METAL = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
IS_VULKAN = hasattr(torch, "is_vulkan_available") and torch.is_vulkan_available()

def get_hardware_backend_info() -> dict:
    """
    Returns comprehensive hardware probe metadata across all accelerators.
    """
    if IS_NVIDIA_CUDA:
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {
            "backend": "cuda",
            "vendor": "NVIDIA",
            "device_name": dev_name,
            "vram_gb": round(vram_gb, 2),
            "triton_acceleration": HAS_TRITON,
            "instruction_set": "Tensor Cores (CUDA / PTX)",
        }
    elif IS_AMD_ROCM:
        dev_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {
            "backend": "rocm",
            "vendor": "AMD",
            "device_name": dev_name,
            "vram_gb": round(vram_gb, 2),
            "triton_acceleration": HAS_TRITON,
            "instruction_set": "Matrix Core (HIP / CDNA / RDNA)",
        }
    elif IS_INTEL_XPU:
        dev_name = torch.xpu.get_device_name(0)
        return {
            "backend": "xpu",
            "vendor": "Intel",
            "device_name": dev_name,
            "triton_acceleration": False,
            "instruction_set": "Intel XMX / SYCL / OneAPI",
        }
    elif IS_APPLE_METAL:
        return {
            "backend": "mps",
            "vendor": "Apple",
            "device_name": "Apple Silicon (Metal Performance Shaders)",
            "triton_acceleration": False,
            "instruction_set": "Metal GPU Matrix Engines (MPS)",
        }
    elif IS_VULKAN:
        return {
            "backend": "vulkan",
            "vendor": "Cross-Vendor",
            "device_name": "Vulkan SPIR-V Compute Device",
            "triton_acceleration": False,
            "instruction_set": "SPIR-V Compute Shaders",
        }
    else:
        return {
            "backend": "cpu",
            "vendor": "Host CPU",
            "device_name": "x86_64 / ARM CPU",
            "triton_acceleration": False,
            "instruction_set": "C++20 AVX2 / AVX-512 / NEON SIMD",
        }

try:
    from .triton_swiglu import launch_triton_swiglu
    from .triton_flash_tree import launch_triton_flash_tree_attention
    from .triton_recirculation import launch_triton_subspace_recirculation
    from .triton_swiglu import HAS_TRITON
except ImportError:
    HAS_TRITON = False

def dispatch_flash_tree_attention(
    q: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_tree: torch.Tensor,
    v_tree: torch.Tensor,
    tree_mask: torch.Tensor
) -> torch.Tensor:
    """
    Dispatches Flash-Tree-Attention DAG speculative verification.
    """
    if HAS_CUDA and HAS_TRITON and q.is_cuda and q.dtype == torch.float16:
        try:
            return launch_triton_flash_tree_attention(q, k_prefix, v_prefix, k_tree, v_tree, tree_mask)
        except Exception:
            pass

    # PyTorch Fallback
    # Concatenate prefix + tree keys/values
    batch, heads, n_q, head_dim = q.shape
    prefix_len = k_prefix.shape[2] if k_prefix is not None and k_prefix.numel() > 0 else 0

    if prefix_len > 0:
        k_all = torch.cat([k_prefix, k_tree], dim=2)
        v_all = torch.cat([v_prefix, v_tree], dim=2)
    else:
        k_all = k_tree
        v_all = v_tree

    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k_all.transpose(-1, -2)) * scale # [Batch, Heads, N_Q, PrefixLen + N_Q]

    # Apply causal tree mask to the tree part
    if prefix_len > 0:
        scores[:, :, :, prefix_len:] += tree_mask.unsqueeze(0).unsqueeze(0)
    else:
        scores += tree_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.matmul(attn_weights, v_all)
    return out

def dispatch_subspace_recirculation(
    h_shallow: torch.Tensor,
    h_deep: torch.Tensor,
    u_proj: torch.Tensor,
    alpha: float = 0.15
) -> torch.Tensor:
    """
    Dispatches Subspace Recirculation mixing.
    """
    if HAS_CUDA and HAS_TRITON and h_shallow.is_cuda and h_shallow.dtype == torch.float16:
        try:
            return launch_triton_subspace_recirculation(h_shallow, h_deep, u_proj, alpha)
        except Exception:
            pass

    # PyTorch Fallback
    dtype = h_shallow.dtype
    s_sub = torch.matmul(h_deep, u_proj.to(dtype=dtype, device=h_shallow.device))
    recon = torch.matmul(s_sub, u_proj.t().to(dtype=dtype, device=h_shallow.device))
    return h_shallow + (alpha * recon)
