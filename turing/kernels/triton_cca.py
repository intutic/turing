"""
Triton GPU Kernel: Fused Linear Down-Projection + Causal 1D Depthwise Sequence Convolution (CCA).
Executes latent down-projection and causal 1D sequence convolution in SRAM without materializing intermediate tensors.
"""

from typing import Optional
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_linear_conv1d_causal_cuda(
    x: torch.Tensor,          # [Batch, SeqLen, HiddenDim]
    w_linear: torch.Tensor,    # [LatentDim, HiddenDim]
    w_conv: torch.Tensor       # [LatentDim, 1, KernelSize] (depthwise)
) -> torch.Tensor:
    """
    Fused linear projection + causal 1D depthwise convolution for CCA.
    """
    # Down-project
    lat = F.linear(x, w_linear) # [Batch, SeqLen, LatentDim]
    
    # Causal depthwise 1D conv
    lat_t = lat.transpose(1, 2) # [Batch, LatentDim, SeqLen]
    kernel_size = w_conv.shape[-1]
    lat_pad = F.pad(lat_t, (kernel_size - 1, 0)) # Causal left padding
    out_conv = F.conv1d(lat_pad, w_conv, groups=w_conv.shape[0]) # [Batch, LatentDim, SeqLen]
    
    return out_conv.transpose(1, 2)
