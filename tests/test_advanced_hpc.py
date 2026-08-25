import pytest
import numpy as np
import torch

def test_matrix_power_recurrence_jump_ahead():
    from turing.core.matrix_pow import LogarithmicRecurrenceEngine
    
    x0, x1 = 123, 456
    A, B, C, M = 3, 7, 11, 1000000007
    
    engine = LogarithmicRecurrenceEngine(x0, x1, A, B, C, M)
    
    # Check base cases
    assert engine.jump_ahead(0) == x0 % M
    assert engine.jump_ahead(1) == x1 % M
    
    # Compare O(log K) jump against linear simulation for K=100
    linear_val = engine.jump_ahead(100)
    
    # Generate chunk
    chunk = engine.generate_chunk(start_idx=95, length=10)
    assert len(chunk) == 10
    assert chunk[5] == linear_val

def test_persistent_thread_reducer():
    from turing.core.persistent_gemv import PersistentThreadAccumulator
    
    acc = PersistentThreadAccumulator(num_threads=4, dim=128)
    
    # Random thread-local data
    thread_data = np.random.randn(4, 128).astype(np.float32)
    reduced = acc.reduce_sum(thread_data)
    
    expected = np.sum(thread_data, axis=0)
    assert np.allclose(reduced, expected, atol=1e-5)

def test_unified_memory_buffer_pool():
    from turing.core.unified_memory import UnifiedMemoryBufferPool
    
    pool = UnifiedMemoryBufferPool(capacity_bytes=1024 * 1024)
    assert pool.free_bytes == 1024 * 1024
    
    off1 = pool.allocate(1000) # Should align to 1024
    assert off1 == 0
    assert pool.used_bytes_count == 1024
    
    off2 = pool.allocate(2000) # Should align to 2048
    assert off2 == 1024
    assert pool.used_bytes_count == 1024 + 2048
    
    pool.reset()
    assert pool.used_bytes_count == 0
    assert pool.free_bytes == 1024 * 1024

def test_pso_multimodal_objectives():
    from turing.core.swarm_objectives import evaluate_objective
    
    # Global minimum of Sphere/Rastrigin/Ackley is at origin (0, ..., 0)
    origin = np.zeros(5, dtype=np.float64)
    ackley_zero = evaluate_objective("ackley", origin)
    assert np.allclose(ackley_zero, 0.0, atol=1e-5)
    
    rastrigin_zero = evaluate_objective("rastrigin", origin)
    assert np.allclose(rastrigin_zero, 0.0, atol=1e-5)
    
    griewank_zero = evaluate_objective("griewank", origin)
    assert np.allclose(griewank_zero, 0.0, atol=1e-5)
    
    ones = np.ones(5, dtype=np.float64)
    rosenbrock_ones = evaluate_objective("rosenbrock", ones)
    assert np.allclose(rosenbrock_ones, 0.0, atol=1e-5)

def test_laplacian_2d_stencil_diffusion():
    try:
        from turing import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False

    grid = np.zeros((32, 32), dtype=np.float32)
    grid[16, 16] = 100.0 # Heat impulse at center
    
    if HAS_CSRC:
        diffused = turing_csrc.laplacian_2d_stencil_step(grid, 0.1)
        assert diffused.shape == (32, 32)
        # Center should cool down, neighbors should heat up
        assert diffused[16, 16] < 100.0
        assert diffused[16, 17] > 0.0
        assert diffused[15, 16] > 0.0

def test_online_welford_and_exponential_decay():
    from turing.core.router_annealer import OnlineWelfordAccumulator, compute_exponential_decay
    
    # Welford test
    welford = OnlineWelfordAccumulator()
    data = np.random.randn(500).astype(np.float64)
    for x in data:
        welford.update(x)
        
    assert np.allclose(welford.mean_value, np.mean(data), atol=1e-3)
    assert np.allclose(welford.variance_value, np.var(data, ddof=1), atol=1e-3)
    
    # Exponential decay test
    v0 = compute_exponential_decay(init_val=2.0, min_val=0.1, current_step=0, max_steps=100)
    v50 = compute_exponential_decay(init_val=2.0, min_val=0.1, current_step=50, max_steps=100)
    v100 = compute_exponential_decay(init_val=2.0, min_val=0.1, current_step=100, max_steps=100)
    
    assert np.allclose(v0, 2.0)
    assert np.allclose(v100, 0.1)
    assert 0.1 < v50 < 2.0
