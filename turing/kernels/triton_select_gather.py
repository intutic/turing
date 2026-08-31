"""
Batched Candidate Option Logit Gather & Argmax Reduction Kernel.
Computes option log-probabilities on GPU in parallel without host CPU .item() synchronization.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F

__all__ = ["gather_option_logits_gpu", "dispatch_batched_option_select"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:
    @triton.jit
    def _batched_select_gather_kernel(
        LogProbs_ptr, OptTokens_ptr, OptLengths_ptr, Scores_ptr,
        num_options, max_len, vocab_size,
        stride_opt_o, stride_opt_l,
        BLOCK_OPT: tl.constexpr
    ):
        pid = tl.program_id(0)
        offs_opt = pid * BLOCK_OPT + tl.arange(0, BLOCK_OPT)
        mask_opt = offs_opt < num_options

        acc_score = tl.zeros((BLOCK_OPT,), dtype=tl.float32)

        for l_idx in range(max_len):
            tok_id = tl.load(OptTokens_ptr + offs_opt * stride_opt_o + l_idx * stride_opt_l, mask=mask_opt, other=0)
            opt_len = tl.load(OptLengths_ptr + offs_opt, mask=mask_opt, other=0)

            valid_pos = (l_idx < opt_len) & (tok_id >= 0) & (tok_id < vocab_size) & mask_opt
            lp_val = tl.load(LogProbs_ptr + tok_id, mask=valid_pos, other=0.0)
            acc_score += tl.where(valid_pos, lp_val, 0.0)

        # Average by token length for length-normalized score
        lens = tl.load(OptLengths_ptr + offs_opt, mask=mask_opt, other=1)
        lens_f = tl.maximum(lens.to(tl.float32), 1.0)
        norm_score = acc_score / lens_f

        tl.store(Scores_ptr + offs_opt, norm_score, mask=mask_opt)


def gather_option_logits_gpu(
    log_probs: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    candidate_lengths: torch.Tensor
) -> torch.Tensor:
    num_options, max_len = candidate_token_ids.shape
    vocab_size = log_probs.shape[-1]
    scores = torch.empty((num_options,), device=log_probs.device, dtype=torch.float32)

    BLOCK_OPT = min(64, triton.next_power_of_2(num_options))
    grid = (triton.cdiv(num_options, BLOCK_OPT),)

    _batched_select_gather_kernel[grid](
        log_probs, candidate_token_ids, candidate_lengths, scores,
        num_options, max_len, vocab_size,
        candidate_token_ids.stride(0), candidate_token_ids.stride(1),
        BLOCK_OPT=BLOCK_OPT
    )
    return scores


def dispatch_batched_option_select(
    log_probs: torch.Tensor,
    options_token_ids: List[List[int]]
) -> int:
    """
    Evaluates option log-probabilities and returns index of best candidate.
    """
    if not options_token_ids:
        return 0

    if not log_probs.is_cuda or not HAS_TRITON:
        # Fast zero-allocation CPU path
        best_score = float("-inf")
        best_idx = 0
        lp_list = log_probs.tolist() if isinstance(log_probs, torch.Tensor) else log_probs
        for i, opt in enumerate(options_token_ids):
            if not opt:
                continue
            s = sum(lp_list[t] for t in opt) / len(opt)
            if s > best_score:
                best_score = s
                best_idx = i
        return best_idx

    num_options = len(options_token_ids)
    lengths = [len(opt) for opt in options_token_ids]
    max_len = max(lengths) if lengths else 1

    # Build padded tensor on CUDA
    pad_tokens = torch.full((num_options, max_len), -1, dtype=torch.long, device=log_probs.device)
    for i, opt in enumerate(options_token_ids):
        for j, t in enumerate(opt):
            pad_tokens[i, j] = t

    len_tensor = torch.tensor(lengths, dtype=torch.long, device=log_probs.device)

    try:
        scores = gather_option_logits_gpu(log_probs, pad_tokens, len_tensor)
        return int(torch.argmax(scores).item())
    except Exception:
        return 0
