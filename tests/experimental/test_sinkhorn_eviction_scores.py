"""
Sinkhorn OT KV eviction ranks keys by the mass the queries send them, not by the plan's column marginals (which
the final v update makes uniform). A planted input must keep its high-attention keys, in every implementation:
the PyTorch ops (CUDA and CPU fallback), the C++ kernel, and the sinkhorn_knopp_eviction dispatcher.
"""

import numpy as np
import pytest
import torch

from turing.experimental.optimal_transport import sinkhorn_knopp_eviction
from turing.experimental.kernels.sinkhorn_ot_cuda import sinkhorn_ot_eviction_cuda

try:
    from turing import turing_csrc
except ImportError:
    turing_csrc = None


def _planted(seed=0, m=8, n=64, d=64, budget=8):
    g = torch.Generator().manual_seed(seed)
    direction = torch.zeros(d)
    direction[0] = 1.0
    q = 2.0 * direction + 0.3 * torch.randn(m, d, generator=g)
    k = 0.3 * torch.randn(n, d, generator=g)
    planted = torch.randperm(n, generator=g)[:budget].sort().values
    k[planted] += 3.0 * direction
    return q, k, planted, budget


def _impls():
    impls = {"torch": lambda q, k, b: sinkhorn_ot_eviction_cuda(q, k, b, 0.05, 15),
             "dispatcher": lambda q, k, b: sinkhorn_knopp_eviction(q, k, budget=b, epsilon=0.05, num_iters=15)}
    if turing_csrc is not None:
        impls["cpp"] = lambda q, k, b: tuple(torch.from_numpy(x) for x in turing_csrc.sinkhorn_ot_eviction(
            q.numpy(), k.numpy(), b, epsilon=0.05, num_iters=15))
    if torch.cuda.is_available():
        impls["torch_cuda"] = lambda q, k, b: tuple(x.cpu() for x in sinkhorn_ot_eviction_cuda(q.cuda(), k.cuda(), b, 0.05, 15))
    return impls


@pytest.mark.parametrize("name", list(_impls()))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_planted_high_mass_keys_are_retained(name, seed):
    q, k, planted, budget = _planted(seed)
    retained, scores = _impls()[name](q, k, budget)
    assert retained.long().tolist() == planted.tolist()
    assert scores.shape == (k.shape[0],)
    assert abs(float(scores.sum()) - 1.0) < 1e-4
    others = torch.ones(k.shape[0], dtype=torch.bool)
    others[planted] = False
    assert scores[planted].min() > 10 * scores[others].max()  # not the uniform 1/N of the plan's marginals


@pytest.mark.skipif(turing_csrc is None, reason="turing_csrc not compiled")
@pytest.mark.parametrize("seed", [3, 4, 5])
def test_cpp_matches_torch(seed):
    rng = np.random.default_rng(seed)
    q = torch.from_numpy(rng.standard_normal((8, 64)).astype(np.float32))
    k = torch.from_numpy(rng.standard_normal((48, 64)).astype(np.float32))
    ref_idx, ref_scores = sinkhorn_ot_eviction_cuda(q, k, 12, 0.05, 15)
    idx, scores = turing_csrc.sinkhorn_ot_eviction(q.numpy(), k.numpy(), 12, epsilon=0.05, num_iters=15)
    np.testing.assert_allclose(scores, ref_scores.numpy(), rtol=1e-3, atol=1e-6)
    assert idx.tolist() == ref_idx.tolist()
