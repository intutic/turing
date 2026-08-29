"""
Fused In-SRAM Shannon Entropy and Uncertainty Gating CUDA Triton Kernel.
Computes H(P) = -\sum P_i \ln P_i directly in GPU registers from unnormalized logits
using a 1-pass online normalizer.
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


if HAS_TRITON:
    @triton.jit
    def _fused_shannon_entropy_kernel(
        Logits_ptr,
        Entropy_out_ptr,
        vocab_size,
        stride_b, stride_v,
        BLOCK_V: tl.constexpr
    ):
        b = tl.program_id(0)
        offs_v = tl.arange(0, BLOCK_V)
        mask = offs_v < vocab_size

        logits_base = Logits_ptr + b * stride_b
        vals = tl.load(logits_base + offs_v * stride_v, mask=mask, other=-float("inf")).to(tl.float32)

        # 1. Online Max
        m = tl.max(vals, axis=0)

        # 2. Exponentials and Normalizer Sum
        shifted = vals - m
        exp_vals = tl.exp(shifted)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        sum_exp = tl.sum(exp_vals, axis=0)
        inv_sum = 1.0 / tl.maximum(sum_exp, 1e-12)

        # 3. Probabilities and Log-Probabilities
        # log_p = shifted - log(sum_exp)
        # H = -\sum p * log_p = log(sum_exp) - (1 / sum_exp) * \sum (shifted * exp_vals)
        sum_weighted_shift = tl.sum(shifted * exp_vals, axis=0)
        entropy = tl.math.log(sum_exp) - (sum_weighted_shift * inv_sum)

        entropy = tl.maximum(entropy, 0.0)
        tl.store(Entropy_out_ptr + b, entropy)


def fused_shannon_entropy_cuda(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Shannon entropy across the last dimension via fused 1-pass Triton reduction.
    logits: [Batch, VocabSize] or [VocabSize]
    Returns: [Batch] or scalar tensor
    """
    if not logits.is_cuda or not HAS_TRITON:
        probs = F.softmax(logits.float(), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return torch.clamp(entropy, min=0.0)

    orig_shape = logits.shape
    if logits.dim() == 1:
        l_2d = logits.unsqueeze(0)
    elif logits.dim() == 3:
        l_2d = logits.view(-1, logits.shape[-1])
    else:
        l_2d = logits

    batch_size, vocab_size = l_2d.shape
    out = torch.empty(batch_size, device=logits.device, dtype=torch.float32)

    # Next power of 2 for block size
    block_v = triton.next_power_of_2(vocab_size)
    block_v = max(64, min(block_v, 65536))

    grid = (batch_size,)
    _fused_shannon_entropy_kernel[grid](
        l_2d, out,
        vocab_size,
        l_2d.stride(0), l_2d.stride(1),
        BLOCK_V=block_v,
        num_warps=8 if block_v >= 4096 else 4
    )

    if logits.dim() == 1:
        return out.squeeze(0)
    elif logits.dim() == 3:
        return out.view(orig_shape[0], orig_shape[1])
    return out
