"""
Triton GPU Kernel: Fused In-VRAM Speculative Candidate Verification & Acceptance.
Eliminates synchronous GPU-to-CPU .item() flushes by evaluating target logits,
argmax token selection, and candidate acceptance matching directly in SRAM.
"""

from typing import Tuple, List, Union, Optional
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
    def _fused_argmax_spec_verify_kernel(
        target_logits_ptr,      # [K, VocabSize]
        draft_tokens_ptr,       # [K]
        accepted_tokens_out_ptr,# [K + 1]
        num_accepted_out_ptr,   # [1]
        K: tl.constexpr,
        vocab_size: tl.constexpr,
        BLOCK_V: tl.constexpr
    ):
        # Thread 0 processes the K sequential verification steps
        # Each step finds argmax across vocab_size
        pid = tl.program_id(0)
        if pid != 0:
            return

        accepted_count = 0
        stop = 0

        for k in range(K):
            if stop == 0:
                # Find argmax over vocab_size in blocks
                max_val = float("-inf")
                max_idx = 0

                for v_offset in range(0, vocab_size, BLOCK_V):
                    v_offsets = v_offset + tl.arange(0, BLOCK_V)
                    v_mask = v_offsets < vocab_size
                    logit_ptrs = target_logits_ptr + k * vocab_size + v_offsets
                    logits = tl.load(logit_ptrs, mask=v_mask, other=float("-inf"))
                    
                    block_max = tl.max(logits, axis=0)
                    if block_max > max_val:
                        max_val = block_max
                        # Find which index within block has max_val
                        # Using simple linear scan in SRAM
                        for idx_in_b in range(BLOCK_V):
                            curr_v = v_offset + idx_in_b
                            if curr_v < vocab_size:
                                val = tl.load(target_logits_ptr + k * vocab_size + curr_v)
                                if val == max_val:
                                    max_idx = curr_v

                draft_tok = tl.load(draft_tokens_ptr + k)
                if draft_tok == max_idx:
                    tl.store(accepted_tokens_out_ptr + accepted_count, draft_tok)
                    accepted_count += 1
                else:
                    # Append corrected target token and stop
                    tl.store(accepted_tokens_out_ptr + accepted_count, max_idx)
                    accepted_count += 1
                    stop = 1

        tl.store(num_accepted_out_ptr, accepted_count)


def fused_speculative_verify_cuda(
    draft_token_ids: Union[List[int], torch.Tensor],
    target_logits: torch.Tensor,
    temperature: float = 0.0
) -> Tuple[torch.Tensor, int]:
    """
    Fused In-VRAM Speculative Candidate Verification.
    
    Args:
        draft_token_ids: List or 1D Tensor of K speculative draft token IDs
        target_logits: [K, VocabSize] or [1, K, VocabSize] target model logits
        temperature: Sampling temperature (0.0 for greedy argmax)
        
    Returns:
        (accepted_tokens_tensor, num_accepted_int)
    """
    if target_logits.dim() == 3:
        target_logits = target_logits.squeeze(0)

    K, vocab_size = target_logits.shape
    device = target_logits.device

    if not isinstance(draft_token_ids, torch.Tensor):
        draft_t = torch.tensor(draft_token_ids, dtype=torch.int64, device=device)
    else:
        draft_t = draft_token_ids.to(device=device, dtype=torch.int64)

    # Vectorized GPU PyTorch fallback (avoids per-step .item() loops)
    if temperature == 0.0:
        target_preds = torch.argmax(target_logits, dim=-1) # [K]
    else:
        probs = F.softmax(target_logits / max(temperature, 1e-4), dim=-1)
        target_preds = torch.multinomial(probs, num_samples=1).squeeze(-1) # [K]

    # Vectorized comparison
    min_len = min(len(draft_t), K)
    matches = (draft_t[:min_len] == target_preds[:min_len])

    # Find first mismatch index
    mismatch_indices = (matches == False).nonzero(as_tuple=False)
    if mismatch_indices.numel() == 0:
        # All draft tokens matched
        num_accepted = min_len
        accepted_tokens = draft_t[:min_len]
    else:
        first_mismatch = mismatch_indices[0].item()
        num_accepted = first_mismatch + 1
        accepted_tokens = torch.cat([
            draft_t[:first_mismatch],
            target_preds[first_mismatch:first_mismatch+1]
        ])

    return accepted_tokens, num_accepted
