import pytest
import torch
from turing.core.subspace import SubspaceRecirculation

def test_subspace_recirculation():
    hidden_dim = 256
    rank = 32
    recirc = SubspaceRecirculation(hidden_dim=hidden_dim, rank=rank, alpha=0.15)

    h_shallow = torch.randn(2, 8, hidden_dim)
    h_deep = torch.randn(2, 8, hidden_dim)

    out = recirc(h_shallow, h_deep)
    assert out.shape == h_shallow.shape
    assert not torch.allclose(out, h_shallow)


