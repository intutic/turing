"""
In-SRAM Sinkhorn-Knopp Entropic Optimal Transport (OT) KV Cache Eviction.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

from turing.kernels.sinkhorn_ot_cuda import sinkhorn_ot_eviction_cuda

def sinkhorn_knopp_eviction(
    q: torch.Tensor,
    k: torch.Tensor,
    budget: int,
    epsilon: float = 0.05,
    num_iters: int = 15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Entropic Optimal Transport KV pruning.
    q: [M_queries, HeadDim]
    k: [N_keys, HeadDim]
    budget: Target number of keys to retain
    epsilon: Entropic regularization parameter (0.05)
    Returns: (retained_k_indices, marginal_key_mass)
    """
    m_queries, head_dim = q.shape
    n_keys, _ = k.shape

    if n_keys <= budget:
        all_idx = torch.arange(n_keys, device=k.device)
        return all_idx, torch.ones(n_keys, device=k.device) / max(1, n_keys)

    if q.is_cuda and k.is_cuda:
        return sinkhorn_ot_eviction_cuda(q, k, budget, epsilon, num_iters)

    if HAS_CSRC:
        q_cpu = q.detach().to(torch.float32).cpu().contiguous().numpy()
        k_cpu = k.detach().to(torch.float32).cpu().contiguous().numpy()
        retained_np, mass_np = turing_csrc.sinkhorn_ot_eviction(q_cpu, k_cpu, budget, epsilon, num_iters)
        return (
            torch.from_numpy(retained_np).to(device=k.device, dtype=torch.int64),
            torch.from_numpy(mass_np).to(device=k.device, dtype=k.dtype)
        )

    scale = 1.0 / math.sqrt(head_dim) if "math" in globals() else 1.0 / (head_dim ** 0.5)

    # Cost Matrix: C_ij = - (q_i . k_j) / sqrt(D)
    cost = -torch.matmul(q, k.t()) * scale # [M, N]

    # Gibbs kernel: K = exp(- (C - min(C)) / epsilon)
    cost_min = cost.min()
    gibbs = torch.exp(-(cost - cost_min) / epsilon)

    # Sinkhorn-Knopp matrix scaling iterations
    # Uniform query prior u_0 = 1 / M
    u = torch.ones(m_queries, device=q.device) / m_queries
    v = torch.ones(n_keys, device=k.device) / n_keys

    for _ in range(num_iters):
        # Update u: u = mu / (K @ v)
        kv = torch.matmul(gibbs, v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        u = (1.0 / m_queries) / kv

        # Update v: v = 1 / (K.T @ u)
        ktu = torch.matmul(gibbs.t(), u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        v = (1.0 / n_keys) / ktu

    # Transport plan P = diag(u) @ K @ diag(v)
    # Marginal key mass: m_j = sum_i P_ij
    p = u.unsqueeze(-1) * gibbs * v.unsqueeze(0)
    key_mass = torch.sum(p, dim=0) # [N_keys]

    # Top-K selection by highest attention transport mass
    _, keep_indices = torch.topk(key_mass, k=budget)
    sorted_indices, _ = torch.sort(keep_indices)

    return sorted_indices, key_mass

class OptimalTransportEviction(nn.Module):
    """
    Applies Entropic Optimal Transport to prune KV caches during long-context generation.
    """
    def __init__(self, epsilon: float = 0.05, num_iters: int = 15):
        super().__init__()
        self.epsilon = epsilon
        self.num_iters = num_iters

    def prune_kv_cache(
        self,
        query: torch.Tensor, # [Batch, Heads, NumQueries, HeadDim]
        k_cache: torch.Tensor, # [Batch, Heads, NumKeys, HeadDim]
        v_cache: torch.Tensor, # [Batch, Heads, NumKeys, HeadDim]
        budget: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, heads, m_q, d = query.shape
        _, _, n_k, _ = k_cache.shape

        if n_k <= budget:
            return k_cache, v_cache

        pruned_k = []
        pruned_v = []

        for b in range(batch):
            b_k = []
            b_v = []
            for h in range(heads):
                q_head = query[b, h]
                k_head = k_cache[b, h]
                v_head = v_cache[b, h]

                keep_idx, _ = sinkhorn_knopp_eviction(
                    q_head, k_head, budget=budget, epsilon=self.epsilon, num_iters=self.num_iters
                )
                b_k.append(k_head[keep_idx])
                b_v.append(v_head[keep_idx])
            pruned_k.append(torch.stack(b_k, dim=0))
            pruned_v.append(torch.stack(b_v, dim=0))

        return torch.stack(pruned_k, dim=0), torch.stack(pruned_v, dim=0)
