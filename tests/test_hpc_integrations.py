import pytest
import math
import torch

from turing.models.mesh_2d import Mesh2DCoordinator, Mesh2DParallelLinear
from turing.core.swarm_opt import AsynchronousSwarmOptimizer
from turing.core.cca_fast import CooperativeSharedConv2D, FastCompressedConvolutionalAttention
from turing.kernels.softened_attention import SoftenedAttentionEngine
from turing.core.hex_quant import HexagonalSubspaceQuantizer

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

def test_asynchronous_swarm_optimizer_monotonic_convergence():
    # Test sphere function optimization: f(x) = sum(x^2), min at x = 0
    def sphere_obj(pos: torch.Tensor) -> float:
        return float(torch.sum(pos ** 2).item())

    optimizer = AsynchronousSwarmOptimizer(
        dim=4,
        population_size=20,
        bounds=(-5.0, 5.0),
        device=torch.device("cpu")
    )

    best_pos, best_fit, info = optimizer.optimize(sphere_obj, max_iterations=20)
    assert info["convergence_monotonic"] is True
    assert best_fit < 1.0
    assert best_pos.shape == (4,)

def test_cooperative_shared_cca_fast():
    cca_fast = FastCompressedConvolutionalAttention(
        hidden_dim=64,
        compression_ratio=4,
        kernel_size=3
    )

    kv_states = torch.randn(2, 16, 64)
    compressed_out = cca_fast(kv_states)
    assert compressed_out.shape == (2, 4, 64)

def test_softened_attention_engine():
    attn = SoftenedAttentionEngine(
        hidden_dim=64,
        num_heads=2,
        head_dim=32,
        softening_sq=1e-4
    )

    q = torch.randn(2, 2, 8, 32)
    k = torch.randn(2, 2, 8, 32)
    v = torch.randn(2, 2, 8, 32)

    out = attn(q, k, v)
    assert out.shape == (2, 2, 8, 32)
    assert not torch.isnan(out).any()

def test_hexagonal_subspace_quantizer():
    quant = HexagonalSubspaceQuantizer(
        codebook_dim=32,
        grid_width=4,
        grid_height=4,
        device=torch.device("cpu")
    )

    activations = torch.randn(8, 32)
    quantized, bmu_indices = quant.quantize_subspace(activations)
    assert quantized.shape == (8, 32)
    assert len(bmu_indices) == 8
    assert (bmu_indices >= 0).all() and (bmu_indices < 16).all()

    # Test hexagonal metric distance
    dist = quant.hex_neighborhood_distance(0, 1)
    assert dist > 0.0

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

    # 3. Native C++ Hexagonal BMU Search
    acts = torch.randn(4, 16).numpy()
    cb = torch.randn(8, 16).numpy()
    bmu_indices, bmu_dists = turing_csrc.hexagonal_bmu_search(acts, cb)
    assert len(bmu_indices) == 4
    assert len(bmu_dists) == 4

    # 4. Native C++ Hexagonal Distance
    hex_dist = turing_csrc.hexagonal_distance(0.0, 0.0, 1.0, 1.0)
    assert hex_dist > 0.0

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

def test_autonomic_threshold_tuner():
    from turing.core.router import AutonomicThresholdTuner

    tuner = AutonomicThresholdTuner(target_latency_ms=6.5, initial_threshold=0.5, initial_tau=1.0)
    init_thresh = tuner.current_threshold
    init_tau = tuner.current_tau

    # Simulate high observed latency -> threshold should adjust
    for _ in range(5):
        tuner.update_from_latency_observation(observed_latency_ms=12.0)

    assert tuner.timestep == 5
    assert 0.1 <= tuner.current_threshold <= 0.9
    assert 0.2 <= tuner.current_tau <= 2.0

def test_heterogeneous_moe_cpu_fast_path():
    from turing.config import ModelConfig
    from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider, HostExpertBank, HeterogeneousMoERunner

    cfg = ModelConfig(
        name="test-moe-cpu",
        hidden_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        vocab_size=100
    )
    bank = HostExpertBank(num_layers=2, num_experts=4, hidden_dim=32, ffn_dim=64, active_subspace_dim=16)
    decider = BandwidthAdaptiveDecider(torch.device("cpu"), pcie_bandwidth_gb_s=0.1, cpu_throughput_gflops=1000.0, gpu_throughput_gflops=10.0)
    runner = HeterogeneousMoERunner(cfg, bank, decider, torch.device("cpu"))

    hidden = torch.randn(2, 4, 32)
    logits = torch.randn(2, 4, 4)
    out, stats = runner.route_and_execute(hidden, logits, layer_idx=0, top_k=2)

    assert out.shape == (2, 4, 32)
    assert stats["total_active_experts"] > 0

def test_asynch_task_scheduler_csrc():
    from turing.core.asynch_scheduler import AsynchronousWorkerPool
    import numpy as np

    pool = AsynchronousWorkerPool(num_workers=4)
    tokens = torch.randn(16, 64)
    scaled = pool.parallel_scale_tokens(tokens, scale=2.5)

    assert scaled.shape == (16, 64)
    assert torch.allclose(scaled, tokens * 2.5, atol=1e-5)

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
    from turing.demo.agent_system import MultiAgentCoordinator
    import numpy as np

    states = np.random.randn(4, 32).astype(np.float32)
    system = MultiAgentCoordinator(engine=None)

    recirc = system.recirculate_agent_beliefs(states, softening_sq=1e-4, step_size=0.05)
    assert recirc.shape == (4, 32)
    assert not np.isnan(recirc).any()

def test_pso_optimize_hyperparams_csrc():
    opt_params = AsynchronousSwarmOptimizer.optimize_csrc(
        num_particles=20,
        num_dims=4,
        num_iterations=30,
        lower_bounds=[-2.0, -2.0, -2.0, -2.0],
        upper_bounds=[2.0, 2.0, 2.0, 2.0]
    )
    assert len(opt_params) == 4
    # All dimensions should have converged close to 0 (sphere minimum)
    for p in opt_params:
        assert abs(p) < 1.0

def test_hex_quantize_activations_csrc():
    quant = HexagonalSubspaceQuantizer(
        codebook_dim=16,
        grid_width=4,
        grid_height=4,
        device=torch.device("cpu")
    )
    acts = torch.randn(8, 16)
    quantized, bmu = quant.quantize_subspace(acts)
    assert quantized.shape == (8, 16)
    assert len(bmu) == 8
    assert not torch.isnan(quantized).any()






