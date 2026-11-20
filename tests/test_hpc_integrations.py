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


def test_native_csrc_hpc_kernels():
    try:
        from turing import turing_csrc
    except ImportError:
        import turing_csrc

    # 1. Native C++ Cooperative Shared 1D Conv
    x = torch.randn(2, 4, 16).numpy()
    w = torch.randn(8, 4, 3).numpy()
    b = torch.zeros(8).numpy()
    conv_out = turing_csrc.cooperative_shared_conv1d(x, w, b, 1, 1)
    assert conv_out.shape == (2, 8, 16)

    # 2. Native C++ Softened N-Body Attention
    q = torch.randn(2, 2, 8, 16).numpy()
    k = torch.randn(2, 2, 8, 16).numpy()
    v = torch.randn(2, 2, 8, 16).numpy()
    attn_out = turing_csrc.softened_nbody_attention(q, k, v, 1e-4, 0.25)
    assert attn_out.shape == (2, 2, 8, 16)

    # 5. Native C++ Fused Adam Step (Fused High-Performance Kernel)
    p = torch.ones(8).numpy()
    g = (torch.ones(8) * 0.5).numpy()
    m = torch.zeros(8).numpy()
    v = torch.zeros(8).numpy()
    turing_csrc.fused_adam_step(p, g, m, v, 0.01, 0.9, 0.999, 1e-8, 1)
    assert (p < 1.0).all()
    assert (m > 0.0).all()
    assert (v > 0.0).all()

    # 6. Native C++ Multi-Threaded CPU MoE GEMV (Fused High-Performance Kernel)
    in_acts = torch.randn(4, 16).numpy()
    exp_w = torch.randn(4, 32, 16).numpy()
    exp_idx = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 3]], dtype=torch.int32).numpy()
    gate_w = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.8, 0.2]], dtype=torch.float32).numpy()
    moe_out = turing_csrc.parallel_cpu_moe_gemv(in_acts, exp_w, exp_idx, gate_w)
    assert moe_out.shape == (4, 32)


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

def test_cooperative_conv2d_shared_csrc():
    try:
        from turing import turing_csrc
    except ImportError:
        import turing_csrc
    import numpy as np

    x = np.random.randn(4, 16, 16).astype(np.float32)
    w = np.random.randn(8, 4, 3, 3).astype(np.float32)
    b = np.zeros(8, dtype=np.float32)

    out = turing_csrc.cooperative_conv2d_shared(x, w, b, 1, 1)
    assert out.shape == (8, 16, 16)
    assert not np.isnan(out).any()

def test_nbody_belief_recirculate_csrc():
    from turing.experimental.demo.agent_system import MultiAgentCoordinator
    import numpy as np

    states = np.random.randn(4, 32).astype(np.float32)
    system = MultiAgentCoordinator(engine=None)

    recirc = system.recirculate_agent_beliefs(states, softening_sq=1e-4, step_size=0.05)
    assert recirc.shape == (4, 32)
    assert not np.isnan(recirc).any()

