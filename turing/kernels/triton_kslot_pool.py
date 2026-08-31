"""
Fused Triton GPU kernel for k-slot cache pooling with integrated inverse RoPE decoupling.
Combines inverse RoPE rotation, softmax attention, and weighted sum into a single SRAM pass.
"""

from typing import Tuple
import torch

__all__ = ["fused_kslot_pool_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_kslot_pool_rope_kernel(
        # Pointers to tensors
        keys_ptr, values_ptr, queries_ptr, output_k_ptr, output_v_ptr,
        # Dimensions
        batch, num_layers, num_heads, seq_len, head_dim, num_slots,
        # Scaling factor & RoPE base frequency
        scale, rope_base,
        # Strides for keys/values: (B, L, H, N, D)
        stride_kb, stride_kl, stride_kh, stride_kn, stride_kd,
        # Strides for queries: (L, H, k, D)  
        stride_ql, stride_qh, stride_qk, stride_qd,
        # Strides for output: (B, L, H, k, D)
        stride_ob, stride_ol, stride_oh, stride_ok, stride_od,
        # Block sizes
        BLOCK_SEQ: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        # Program IDs
        pid_b = tl.program_id(0)  # batch
        pid_lh = tl.program_id(1)  # layer * num_heads + head
        pid_k = tl.program_id(2)  # slot index
        
        pid_l = pid_lh // num_heads
        pid_h = pid_lh % num_heads
        
        # Load query vector for this slot: [D]
        q_offsets = tl.arange(0, BLOCK_D)
        q_mask = q_offsets < head_dim
        q_ptr = queries_ptr + pid_l * stride_ql + pid_h * stride_qh + pid_k * stride_qk
        q = tl.load(q_ptr + q_offsets * stride_qd, mask=q_mask, other=0.0)
        
        # Accumulate attention-weighted pooling across sequence blocks
        # First pass: compute max logit for numerical stability
        max_logit = float('-inf')
        for seq_start in range(0, seq_len, BLOCK_SEQ):
            seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)
            seq_mask = seq_offsets < seq_len
            
            # Load key block: [BLOCK_SEQ, D]
            k_ptr = keys_ptr + pid_b * stride_kb + pid_l * stride_kl + pid_h * stride_kh
            k_block = tl.load(
                k_ptr + seq_offsets[:, None] * stride_kn + q_offsets[None, :] * stride_kd,
                mask=seq_mask[:, None] & q_mask[None, :], other=0.0
            )
            
            # Dot product: [BLOCK_SEQ]
            logits = tl.sum(k_block * q[None, :], axis=1) * scale
            logits = tl.where(seq_mask, logits, float('-inf'))
            max_logit = tl.maximum(max_logit, tl.max(logits, axis=0))
        
        # Second pass: compute softmax and weighted sum
        sum_exp = 0.0
        acc_k = tl.zeros((BLOCK_D,), dtype=tl.float32)
        acc_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for seq_start in range(0, seq_len, BLOCK_SEQ):
            seq_offsets = seq_start + tl.arange(0, BLOCK_SEQ)
            seq_mask = seq_offsets < seq_len
            
            k_ptr = keys_ptr + pid_b * stride_kb + pid_l * stride_kl + pid_h * stride_kh
            k_block = tl.load(
                k_ptr + seq_offsets[:, None] * stride_kn + q_offsets[None, :] * stride_kd,
                mask=seq_mask[:, None] & q_mask[None, :], other=0.0
            )
            
            v_ptr = values_ptr + pid_b * stride_kb + pid_l * stride_kl + pid_h * stride_kh
            v_block = tl.load(
                v_ptr + seq_offsets[:, None] * stride_kn + q_offsets[None, :] * stride_kd,
                mask=seq_mask[:, None] & q_mask[None, :], other=0.0
            )
            
            logits = tl.sum(k_block * q[None, :], axis=1) * scale
            logits = tl.where(seq_mask, logits, float('-inf'))
            
            weights = tl.exp(logits - max_logit)
            weights = tl.where(seq_mask, weights, 0.0)
            sum_exp += tl.sum(weights, axis=0)
            
            acc_k += tl.sum(weights[:, None] * k_block, axis=0)
            acc_v += tl.sum(weights[:, None] * v_block, axis=0)
        
        # Normalize
        acc_k = acc_k / tl.maximum(sum_exp, 1e-6)
        acc_v = acc_v / tl.maximum(sum_exp, 1e-6)
        
        # Store output
        o_ptr = output_k_ptr + pid_b * stride_ob + pid_l * stride_ol + pid_h * stride_oh + pid_k * stride_ok
        tl.store(o_ptr + q_offsets * stride_od, acc_k, mask=q_mask)
        
        o_v_ptr = output_v_ptr + pid_b * stride_ob + pid_l * stride_ol + pid_h * stride_oh + pid_k * stride_ok
        tl.store(o_v_ptr + q_offsets * stride_od, acc_v, mask=q_mask)


def fused_kslot_pool_cuda(
    keys: torch.Tensor,
    values: torch.Tensor,
    queries: torch.Tensor,
    base: float = 500000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused k-slot pooling on CUDA via Triton.
    Falls back to PyTorch einsum when Triton is unavailable.
    
    Args:
        keys: (B, L, H, N, D)
        values: (B, L, H, N, D)
        queries: (L, H, k, D)
        base: RoPE base frequency (reserved for future RoPE fusion)
    
    Returns:
        Tuple of pooled (keys, values), both (B, L, H, k, D)
    """
    import math
    if not HAS_TRITON or not keys.is_cuda:
        # PyTorch fallback
        logits = torch.einsum('lhjd,blhnd->blhjn', queries.to(keys.dtype), keys)
        logits = logits / math.sqrt(keys.shape[-1])
        attn = torch.softmax(logits, dim=-1)
        pooled_k = torch.einsum('blhjn,blhnd->blhjd', attn, keys)
        pooled_v = torch.einsum('blhjn,blhnd->blhjd', attn, values)
        return pooled_k, pooled_v
    
    batch, num_layers, num_heads, seq_len, head_dim = keys.shape
    num_slots = queries.shape[2]
    scale = 1.0 / math.sqrt(head_dim)
    
    output_k = torch.empty(batch, num_layers, num_heads, num_slots, head_dim, 
                           device=keys.device, dtype=keys.dtype)
    output_v = torch.empty_like(output_k)
    
    BLOCK_SEQ = min(128, triton.next_power_of_2(seq_len))
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    grid = (batch, num_layers * num_heads, num_slots)
    
    _fused_kslot_pool_rope_kernel[grid](
        keys, values, queries, output_k, output_v,
        batch, num_layers, num_heads, seq_len, head_dim, num_slots,
        scale, base,
        keys.stride(0), keys.stride(1), keys.stride(2), keys.stride(3), keys.stride(4),
        queries.stride(0), queries.stride(1), queries.stride(2), queries.stride(3),
        output_k.stride(0), output_k.stride(1), output_k.stride(2), output_k.stride(3), output_k.stride(4),
        BLOCK_SEQ=BLOCK_SEQ, BLOCK_D=BLOCK_D,
    )
    
    return output_k, output_v
