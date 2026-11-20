import pytest
import math
import torch

from turing.models.mesh_2d import Mesh2DCoordinator, Mesh2DParallelLinear

def test_mesh_2d_coordinator_and_linear():
    coord = Mesh2DCoordinator(rows=2, cols=2, rank=3)
    assert coord.row_idx == 1
    assert coord.col_idx == 1
    assert coord.get_row_ranks() == [2, 3]
    assert coord.get_col_ranks() == [1, 3]

    linear_2d = Mesh2DParallelLinear(in_features=128, out_features=256, coordinator=coord)
    assert linear_2d.local_in_features == 64
    assert linear_2d.local_out_features == 128

    x = torch.randn(2, 16, 64)
    out = linear_2d(x)
    assert out.shape == (2, 16, 128)


def test_halo_exchange_step_csrc():
    coord = Mesh2DCoordinator(rows=2, cols=2, rank=0)
    grid = torch.ones(8, 16)
    top_in = torch.ones(16) * 2.0
    bot_in = torch.ones(16) * 3.0

    next_g, top_out, bot_out = coord.halo_exchange(grid, top_in, bot_in, diffusion_alpha=0.25)
    assert next_g.shape == (8, 16)
    assert top_out.shape == (16,)
    assert bot_out.shape == (16,)
    assert not torch.isnan(next_g).any()


def test_nbody_belief_recirculate_csrc():
    from turing.experimental.demo.agent_system import MultiAgentCoordinator
    import numpy as np

    states = np.random.randn(4, 32).astype(np.float32)
    system = MultiAgentCoordinator(engine=None)

    recirc = system.recirculate_agent_beliefs(states, softening_sq=1e-4, step_size=0.05)
    assert recirc.shape == (4, 32)
    assert not np.isnan(recirc).any()

