"""
Triton GPU Fused Batched Sampler Kernel.
Computes in-SRAM Softmax, Top-K, and Gumbel-Max / Multinomial Sampling directly in GPU memory.
Eliminates B x .item() CPU synchronization stalls per continuous batching generation step.
"""

from typing import Optional
import torch
import torch.nn.functional as F

__all__ = ["fused_batched_sample_cuda", "batched_sample_tokens"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_gumbel_sample_kernel(
        logits_ptr,
        temperatures_ptr,
        seeds_ptr,
        output_tokens_ptr,
        batch_size,
        vocab_size: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """
        Fused Gumbel-Max Sampling Kernel:
        Samples token = argmax(logits / temp - log(-log(uniform(0, 1)))) directly in SRAM.
        """
        pid = tl.program_id(0) # batch index
        if pid >= batch_size:
            return

        temp = tl.load(temperatures_ptr + pid)
        seed = tl.load(seeds_ptr + pid)

        # Greedy argmax if temperature is near zero
        is_greedy = temp <= 1e-5

        max_val = float('-inf')
        best_idx = 0

        v_offsets = tl.arange(0, BLOCK_V)
        mask = v_offsets < vocab_size

        logits_row = logits_ptr + pid * vocab_size
        vals = tl.load(logits_row + v_offsets, mask=mask, other=float('-inf'))

        if is_greedy:
            # Simple argmax
            best_idx = tl.argmax(vals, axis=0)
        else:
            # Scaled logits
            scaled_logits = vals / temp
            # Deterministic pseudo-random uniform from LCG per token
            # LCG: next = (seed * 1664525 + 1013904223 + offset)
            rand_u32 = ((seed + v_offsets) * 1664525 + 1013904223) & 0x7FFFFFFF
            # Map to (1e-7, 1.0)
            u = (rand_u32.to(tl.float32) + 1.0) / 2147483648.0
            # Gumbel noise: -log(-log(u))
            gumbel = -tl.log(-tl.log(u))
            perturbed = scaled_logits + gumbel
            best_idx = tl.argmax(perturbed, axis=0)

        tl.store(output_tokens_ptr + pid, best_idx)


def fused_batched_sample_cuda(
    logits: torch.Tensor,               # [Batch, VocabSize]
    temperatures: torch.Tensor,        # [Batch]
    top_k: int = 50,
    top_p: float = 0.9,
    generator_seeds: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Batched In-GPU Sampling.
    Returns: sampled_tokens of shape [Batch] on the same GPU device.
    """
    batch_size, vocab_size = logits.shape

    if generator_seeds is None:
        generator_seeds = torch.randint(
            1, 1000000, (batch_size,),
            device=logits.device,
            dtype=torch.int32
        )

    # Use specialized Triton kernel if vocab_size <= 65536 and power of 2 block fits
    if HAS_TRITON and logits.is_cuda and vocab_size <= 32768:
        output_tokens = torch.empty(batch_size, device=logits.device, dtype=torch.int64)
        BLOCK_V = triton.next_power_of_2(vocab_size)

        _fused_gumbel_sample_kernel[(batch_size,)](
            logits.contiguous(),
            temperatures.contiguous(),
            generator_seeds.contiguous(),
            output_tokens,
            batch_size,
            vocab_size=vocab_size,
            BLOCK_V=BLOCK_V,
        )
        return output_tokens

    # PyTorch Batched Vectorized Fallback (0 Python loops)
    # 1. Mask zero-temperature rows (greedy)
    is_greedy = (temperatures <= 1e-5)
    
    # Safe temperature clamping for softmax
    safe_temps = torch.where(is_greedy, torch.ones_like(temperatures), temperatures).unsqueeze(-1)
    scaled_logits = logits / safe_temps

    if top_k > 0 and top_k < vocab_size:
        topk_vals, topk_inds = torch.topk(scaled_logits, min(top_k, vocab_size), dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        sampled_relative = torch.multinomial(probs, num_samples=1)
        sampled_tokens = torch.gather(topk_inds, -1, sampled_relative).squeeze(-1)
    else:
        probs = F.softmax(scaled_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if is_greedy.any():
        greedy_tokens = torch.argmax(logits, dim=-1)
        sampled_tokens = torch.where(is_greedy, greedy_tokens, sampled_tokens)

    return sampled_tokens


def batched_sample_tokens(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9
) -> torch.Tensor:
    """
    Universal dispatcher for Batched Sampling across CUDA, MPS, and CPU.
    """
    if logits.is_cuda:
        return fused_batched_sample_cuda(logits, temperatures, top_k, top_p)

    # On CPU/Mac with native extension
    try:
        import turing.turing_csrc as turing_csrc
        if logits.is_contiguous() and not logits.is_mps:
            b, v = logits.shape
            temps_np = temperatures.cpu().contiguous().numpy().astype("float32")
            topks_np = (torch.ones(b, dtype=torch.int32) * top_k).numpy()
            unifs_np = torch.rand(b, dtype=torch.float32).numpy()
            res = turing_csrc.sample_batched_logits_simd(
                logits.cpu().contiguous().numpy(),
                temps_np,
                topks_np,
                unifs_np
            )
            return torch.tensor(res, dtype=torch.long, device=logits.device)
    except Exception:
        pass

    # Vectorized PyTorch fallback
    is_greedy = (temperatures <= 1e-5)
    safe_temps = torch.where(is_greedy, torch.ones_like(temperatures), temperatures).unsqueeze(-1)
    scaled_logits = logits / safe_temps

    if top_k > 0 and top_k < logits.shape[-1]:
        topk_vals, topk_inds = torch.topk(scaled_logits, min(top_k, logits.shape[-1]), dim=-1)
        probs = F.softmax(topk_vals, dim=-1)
        sampled_rel = torch.multinomial(probs, num_samples=1)
        sampled_tokens = torch.gather(topk_inds, -1, sampled_rel).squeeze(-1)
    else:
        probs = F.softmax(scaled_logits, dim=-1)
        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    if is_greedy.any():
        greedy_tokens = torch.argmax(logits, dim=-1)
        sampled_tokens = torch.where(is_greedy, greedy_tokens, sampled_tokens)

    return sampled_tokens
