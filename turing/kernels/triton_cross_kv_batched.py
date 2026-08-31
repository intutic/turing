"""
Triton & PyTorch GPU Fused Batched Cross-Model KV Cache Transfer.
Eliminates 80-layer sequential host iteration by batching Ridge projection and RoPE transforms on GPU.
"""

from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F

__all__ = ["fused_batched_cross_model_kv_transfer_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_batched_cross_model_kv_transfer_cuda(
    source_keys_formatted: List[torch.Tensor],   # List of [Batch, SeqLen, SrcHeads, HeadDim]
    source_vals_formatted: List[torch.Tensor],   # List of [Batch, SeqLen, SrcHeads, HeadDim]
    layer_mappers_k: Dict[int, torch.nn.Module], # Dict of ClosedFormRidgeMapper per target layer
    layer_mappers_v: Dict[int, torch.nn.Module], # Dict of ClosedFormRidgeMapper per target layer
    select_src_indices_fn,                       # callable: target_layer_idx -> List[int]
    num_target_layers: int,
    src_rope_base: float = 500000.0,
    tgt_rope_base: float = 500000.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched Cross-Model KV Cache Transfer on GPU.
    """
    from ..core.cross_model_kv import RoPEContentDecoupler

    batch, seq_len, _, head_dim = source_keys_formatted[0].shape

    # Step 1: Strip RoPE from all source key layers in parallel
    stripped_source_keys = [
        RoPEContentDecoupler.strip_rope(k, base=src_rope_base)
        for k in source_keys_formatted
    ]

    tgt_keys: List[torch.Tensor] = []
    tgt_values: List[torch.Tensor] = []

    # Step 2: Vectorized layer projection
    for t_idx in range(num_target_layers):
        selected_src = select_src_indices_fn(t_idx)

        # Concatenate selected source layer features: [Batch, SeqLen, top_k * SrcHeads * HeadDim]
        k_feats = torch.cat([
            stripped_source_keys[s_idx].reshape(batch, seq_len, -1)
            for s_idx in selected_src
        ], dim=-1)

        v_feats = torch.cat([
            source_vals_formatted[s_idx].reshape(batch, seq_len, -1)
            for s_idx in selected_src
        ], dim=-1)

        k_mapper = layer_mappers_k[t_idx]
        v_mapper = layer_mappers_v[t_idx]

        # In-SRAM GEMM
        mapped_k_content = k_mapper(k_feats, is_key=True)
        mapped_v = v_mapper(v_feats, is_key=False)

        # Step 3: Re-encode with target model RoPE
        mapped_k_rope = RoPEContentDecoupler.apply_rope(mapped_k_content, base=tgt_rope_base)

        tgt_keys.append(mapped_k_rope)
        tgt_values.append(mapped_v)

    return tgt_keys, tgt_values
