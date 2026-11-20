"""
Triton & PyTorch GPU Fused Quadtree MRP Speculation Kernel.
Generates 21-node quadtree candidate structures on the device without host synchronization: the tree shape is fixed,
so its parent table and mask are built once per device on the host, and quadrant picks use torch.where, not indexing
by a boolean mask (which syncs to learn the result's size).
"""

from typing import Tuple, List, Optional
import torch
import torch.nn.functional as F

__all__ = ["fused_quadtree_mrp_cuda"]

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def fused_quadtree_mrp_cuda(
    hidden: torch.Tensor,             # [1, HiddenDim] or [HiddenDim]
    draft_weight: torch.Tensor,       # [VocabSize, HiddenDim]
    spatial_proj_w: torch.Tensor,     # [2, HiddenDim]
    slice_width: Optional[int] = None,
    branching_factor: int = 4,
    max_depth: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused GPU Quadtree MRP Candidate Generator.
    Returns:
        token_ids: [21] tensor on device
        parent_indices: [21] tensor on device
        dag_mask: [21, 21] tensor on device
    """
    hidden_dim = hidden.shape[-1]
    vocab_size = draft_weight.shape[0]
    eff_w = min(slice_width or hidden_dim, hidden_dim)

    # 1. 2D Spatial origin
    h_flat = hidden.view(1, -1)
    mrp_origin = F.linear(h_flat, spatial_proj_w).squeeze(0) # [2]
    origin_x = mrp_origin[0]
    origin_y = mrp_origin[1]

    # 2. Sliced Draft GEMV
    h_sliced = h_flat[:, :eff_w]
    w_sliced = draft_weight[:, :eff_w]
    logits = F.linear(h_sliced, w_sliced).squeeze(0) # [VocabSize]

    # 3. Top-64 candidate selection
    topk_k = min(64, vocab_size)
    top_candidates = torch.topk(logits, k=topk_k, dim=-1).indices # [64]

    root_tok = top_candidates[0]
    cands_tail = top_candidates[1:]

    # 4. Vectorized Cartesian quadrant assignment (0 host stalls)
    dx = ((cands_tail % 7).float() - 3.0) - (origin_x * 0.01)
    dy = (((cands_tail // 7) % 7).float() - 3.0) - (origin_y * 0.01)

    q0_mask = (dx >= 0) & (dy >= 0)
    q1_mask = (dx < 0) & (dy >= 0)
    q2_mask = (dx < 0) & (dy < 0)
    q3_mask = (dx >= 0) & (dy < 0)

    def _first_or_fallback(mask: torch.Tensor, fallback_idx: int) -> torch.Tensor:
        first = cands_tail[mask.int().argmax()]  # the first match; argmax is 0 when nothing matches
        return torch.where(mask.any(), first, top_candidates[min(fallback_idx, topk_k - 1)])

    # Depth 1 nodes (4 quadrant representatives)
    d1_q0 = _first_or_fallback(q0_mask, 1)
    d1_q1 = _first_or_fallback(q1_mask, 2)
    d1_q2 = _first_or_fallback(q2_mask, 3)
    d1_q3 = _first_or_fallback(q3_mask, 4)

    # Depth 2 nodes (16 grandchildren)
    d2_toks = []
    cand_len = top_candidates.numel()
    for idx in range(5, 21):
        d2_toks.append(top_candidates[idx % cand_len])

    token_ids = torch.stack([
        root_tok,
        d1_q0, d1_q1, d1_q2, d1_q3,
        *d2_toks
    ])

    parents, dag_mask = _tree(hidden.device)
    return token_ids, parents.clone(), dag_mask.clone()  # device copies: no sync, and callers may edit them


_PARENTS = [-1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
_TREES = {}


def _tree(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """The fixed 21-node tree's parent table and additive ancestor mask (0 on self and ancestors, -inf elsewhere),
    built on the host once per device."""
    if device not in _TREES:
        n = len(_PARENTS)
        mask = torch.full((n, n), float("-inf"), dtype=torch.float32)
        for i in range(n):
            mask[i, i] = 0.0
            curr = _PARENTS[i]
            while curr != -1:
                mask[i, curr] = 0.0
                curr = _PARENTS[curr]
        _TREES[device] = (torch.tensor(_PARENTS, dtype=torch.int32).to(device), mask.to(device))
    return _TREES[device]
