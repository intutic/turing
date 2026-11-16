"""
Unit Tests for Manifold-Constrained Hyper-Connections (mHC) (DeepSeek V4).
"""

import pytest
import torch

from turing.core.mhc import BirkhoffManifoldProjector, ManifoldHyperConnection

def test_birkhoff_manifold_projector_doubly_stochastic():
    """
    Tests that Sinkhorn-projected matrix is doubly stochastic (row sums = 1, col sums = 1, all >= 0).
    """
    n = 4
    raw_matrix = torch.randn(n, n)

    p = BirkhoffManifoldProjector.project(raw_matrix, num_iterations=30)

    # 1. Non-negativity
    assert (p >= 0).all().item(), "Matrix contains negative values"

    # 2. Row sums == 1.0
    row_sums = p.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), f"Row sums not 1: {row_sums}"

    # 3. Col sums == 1.0
    col_sums = p.sum(dim=-2)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4), f"Col sums not 1: {col_sums}"


def test_manifold_hyper_connection_forward_step():
    """
    Tests that mHC executes multi-stream residual routing stably.
    """
    batch_size = 2
    seq_len = 16
    hidden_dim = 128
    num_streams = 4

    mhc_block = ManifoldHyperConnection(hidden_dim=hidden_dim, num_streams=num_streams)

    streams = torch.randn(batch_size, seq_len, num_streams, hidden_dim)

    # Dummy layer function
    def dummy_layer(x):
        return torch.tanh(x) * 0.5

    new_streams, layer_out = mhc_block(streams, dummy_layer)

    assert new_streams.shape == (batch_size, seq_len, num_streams, hidden_dim)
    assert layer_out.shape == (batch_size, seq_len, hidden_dim)

    # Energy bound check: verify no explosion across 20 iterations
    curr_streams = streams
    for _ in range(20):
        curr_streams, _ = mhc_block(curr_streams, dummy_layer)

    assert not torch.isnan(curr_streams).any()
    assert not torch.isinf(curr_streams).any()


@pytest.mark.parametrize("native", [True, False])
def test_projection_is_doubly_stochastic_property(native, monkeypatch):
    """Over 300 random inputs (n = 2..16, entries ~ N(0, 1)): entries >= 0, every row and column sums to 1 within
    1e-4 after 50 Sinkhorn iterations, and the result never expands a vector (spectral norm <= 1)."""
    import turing.core.mhc as mhc
    if native and not mhc.HAS_CSRC:
        pytest.skip("native extension not built")
    monkeypatch.setattr(mhc, "HAS_CSRC", native)
    for seed in range(300):
        g = torch.Generator().manual_seed(seed)
        n = (2, 3, 4, 8, 16)[seed % 5]
        p = BirkhoffManifoldProjector.project(torch.randn(n, n, generator=g), num_iterations=50)
        assert (p >= 0).all()
        torch.testing.assert_close(p.sum(-1), torch.ones(n), atol=1e-4, rtol=0)
        torch.testing.assert_close(p.sum(-2), torch.ones(n), atol=1e-4, rtol=0)
        assert torch.linalg.matrix_norm(p, 2) <= 1 + 1e-4


@pytest.mark.parametrize("native", [True, False])
def test_hyper_connection_res_map_is_doubly_stochastic(native, monkeypatch):
    """The mixing matrix the block actually uses (default 20 iterations on its initial parameters)."""
    import turing.core.mhc as mhc
    if native and not mhc.HAS_CSRC:
        pytest.skip("native extension not built")
    monkeypatch.setattr(mhc, "HAS_CSRC", native)
    for seed in range(50):
        torch.manual_seed(seed)
        h = ManifoldHyperConnection(hidden_dim=8, num_streams=4).get_doubly_stochastic_res_map()
        assert (h >= 0).all()
        torch.testing.assert_close(h.sum(-1), torch.ones(4), atol=1e-5, rtol=0)
        torch.testing.assert_close(h.sum(-2), torch.ones(4), atol=1e-5, rtol=0)
