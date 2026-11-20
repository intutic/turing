"""
Entropic Optimal Transport (OT) KV cache eviction in PyTorch tensor ops (no Triton kernel): it runs on whatever
device q and k are on, and turing/core/optimal_transport.py uses it for CUDA tensors and for the CPU fallback.
"""

import math
from typing import Tuple
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

def sinkhorn_ot_eviction_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    budget: int,
    epsilon: float = 0.05,
    num_iters: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Entropic OT KV eviction with top-k selection: (retained key indices, sorted, int32; key scores [N], sum 1).

    The balanced plan P = diag(u) K diag(v) has uniform column sums (the last v update sets every one to 1/N), so
    its marginals cannot rank keys. A key's score is the mass the queries send it under the final row scaling,
    before that column constraint: sum_i u_i K_ij (= 1 / (N v_j)). Keys the queries attend to strongly need the
    smallest v to be held to 1/N and score highest.
    """
    m_queries, head_dim = q.shape
    n_keys, _ = k.shape

    if n_keys <= budget:
        all_idx = torch.arange(n_keys, device=k.device, dtype=torch.int32)
        return all_idx, torch.ones(n_keys, device=k.device) / max(1, n_keys)

    scale = 1.0 / math.sqrt(head_dim)
    cost = -torch.matmul(q, k.t()) * scale
    cost_min = cost.min()
    gibbs = torch.exp(-(cost - cost_min) / epsilon)

    u = torch.ones(m_queries, device=q.device) / m_queries
    v = torch.ones(n_keys, device=k.device) / n_keys

    for _ in range(num_iters):
        kv = torch.matmul(gibbs, v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        u = (1.0 / m_queries) / kv

        ktu = torch.matmul(gibbs.t(), u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        v = (1.0 / n_keys) / ktu

    key_mass = torch.matmul(gibbs.t(), u.unsqueeze(-1)).squeeze(-1)
    key_mass = key_mass / key_mass.sum()

    _, keep_indices = torch.topk(key_mass, k=budget)
    sorted_indices, _ = torch.sort(keep_indices)

    return sorted_indices.to(torch.int32), key_mass
