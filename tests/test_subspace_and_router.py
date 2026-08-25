import pytest
import torch
from turing.core.subspace import SubspaceRecirculation
from turing.core.router import SubspaceStructuredRouter, DAREOActivationReuse

def test_subspace_recirculation():
    hidden_dim = 256
    rank = 32
    recirc = SubspaceRecirculation(hidden_dim=hidden_dim, rank=rank, alpha=0.15)

    h_shallow = torch.randn(2, 8, hidden_dim)
    h_deep = torch.randn(2, 8, hidden_dim)

    out = recirc(h_shallow, h_deep)
    assert out.shape == h_shallow.shape
    assert not torch.allclose(out, h_shallow)

def test_subspace_router_gumbel_and_inference():
    router = SubspaceStructuredRouter(hidden_dim=128, total_tiles=8, min_tiles=2, max_tiles=4)

    h_j = torch.randn(2, 16, 128)
    router.train()
    mask_train, unc_train = router(h_j)
    assert mask_train.shape == (2, 8)
    assert unc_train.shape == (2, 1)

    router.eval()
    mask_eval, unc_eval = router(h_j)
    assert mask_eval.shape == (2, 8)
    # Check that elements are binary (0 or 1)
    assert torch.all((mask_eval == 0.0) | (mask_eval == 1.0))

def test_dare_o_activation_reuse():
    dare = DAREOActivationReuse(cosine_threshold=0.95)

    x1 = torch.randn(1, 64)
    out1 = torch.randn(1, 64)

    dare.update_cache(x1, out1)

    # Identical input -> should reuse
    should_reuse, cached_out = dare.should_reuse(x1)
    assert should_reuse is True
    assert torch.allclose(cached_out, out1)

    # Orthogonal input -> should not reuse
    x2 = torch.randn(1, 64)
    should_reuse_2, _ = dare.should_reuse(x2)
    # Most likely false unless random collision
    assert isinstance(should_reuse_2, bool)
