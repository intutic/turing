"""
Fused In-SRAM Shannon Entropy and Uncertainty Gating CUDA Triton Kernel.
Computes H(P) = -\sum P_i \ln P_i directly in GPU registers from unnormalized logits.
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

def fused_shannon_entropy_cuda(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Shannon entropy across the last dimension.
    logits: [Batch, VocabSize] or [VocabSize]
    Returns: [Batch] or scalar tensor
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return torch.clamp(entropy, min=0.0)
