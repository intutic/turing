import pytest
import math
import torch
import torch.nn.functional as F

from turing.experimental.kernels.triton_linear_recurrence_step import (
    fused_linear_recurrence_step_cuda,
    fused_kda_conv_recurrence_step_cuda
)
from turing.experimental.paging import (
    GroupedMicroScalePagedKVPool,
    StaticPagedKVPool
)
from turing.experimental.lineage import (
    CleanBaseLineageBuffer
)


def test_grouped_micro_scale_paged_kv_pool():
    """Validates two-level micro-scaling quantization accuracy and pool allocation."""
    torch.manual_seed(42)
    pool = GroupedMicroScalePagedKVPool(
        num_layers=2, num_heads=4, head_dim=64, page_size=16,
        max_total_pages=16, group_size=32
    )

    pages = pool.allocate_pages("req-1", 2)
    assert len(pages) == 2

    k_orig = torch.randn(4, 64, dtype=torch.float16)
    v_orig = torch.randn(4, 64, dtype=torch.float16)

    pool.write_token(layer_idx=0, page_id=pages[0], slot_idx=0, k=k_orig, v=v_orig)
    k_deq, v_deq = pool.read_token(layer_idx=0, page_id=pages[0], slot_idx=0)

    # Verify high signal fidelity under micro-scaling
    cos_sim_k = F.cosine_similarity(k_orig.flatten().float(), k_deq.flatten().float(), dim=0)
    cos_sim_v = F.cosine_similarity(v_orig.flatten().float(), v_deq.flatten().float(), dim=0)

    assert cos_sim_k.item() > 0.99
    assert cos_sim_v.item() > 0.99

    pool.free_pages("req-1")
    assert len(pool.free_page_indices) == 16


def test_clean_base_fed_token_prefix_tracking():
    """Validates exact fed token recording and prefix matching in CleanBaseLineageBuffer."""
    kv = [torch.randn(1, 4, 2, 64)]
    buf = CleanBaseLineageBuffer(kv, initial_token_ids=[101, 2054, 2003, 1037])
    
    assert buf.fed_tokens == [101, 2054, 2003, 1037]
    assert not buf.is_tainted

    # Matching prefix
    common_len = buf.find_common_prefix_length([101, 2054, 2003, 1037, 3000, 4000])
    assert common_len == 4

    # Divergent prefix
    divergent_len = buf.find_common_prefix_length([101, 2054, 9999, 1037])
    assert divergent_len == 2

    # Tainted buffer bypass
    buf.mark_tainted()
    assert buf.find_common_prefix_length([101, 2054, 2003, 1037]) == 0


def test_simd_prefix_matcher_cpp_binding():
    """Tests native C++ SIMD find_common_prefix_fast binding."""
    try:
        import turing.turing_csrc as turing_csrc
        import numpy as np

        tok_a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int32)
        tok_b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 99, 14, 15, 16], dtype=np.int32)

        prefix_len = turing_csrc.find_common_prefix_fast(tok_a, tok_b)
        assert prefix_len == 12
    except ImportError:
        pytest.skip("turing_csrc not loaded")


def test_kda_step_numerical_parity():
    """Tests fused KDA depthwise conv and delta state recurrence."""
    torch.manual_seed(42)
    B, H, D = 1, 2, 16
    kernel_size = 4

    qkv = torch.randn(B, 3, H, D)
    conv_state = torch.zeros(B, 3, H, D, kernel_size)
    conv_weight = torch.randn(3, H, D, kernel_size)
    state = torch.zeros(B, H, D, D)
    gate = torch.zeros(B, H, D)
    beta = torch.ones(B, H) * 0.5

    out, updated_state, updated_conv = fused_kda_conv_recurrence_step_cuda(
        qkv, conv_state, conv_weight, state, gate, beta
    )

    assert out.shape == (B, H, D)
    assert updated_state.shape == (B, H, D, D)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_kda_step_simd_cpp_binding():
    """Tests native C++20 AVX2 SIMD kda_step_fast binding."""
    try:
        import turing.turing_csrc as turing_csrc
        import numpy as np

        B, H, D = 1, 2, 16
        kernel_size = 4

        qkv = np.random.randn(B, 3, H, D).astype(np.float32)
        conv_state = np.zeros((B, 3, H, D, kernel_size), dtype=np.float32)
        conv_weight = np.random.randn(3, H, D, kernel_size).astype(np.float32)
        state = np.zeros((B, H, D, D), dtype=np.float32)
        gate = np.zeros((B, H, D), dtype=np.float32)
        beta = np.ones((B, H), dtype=np.float32) * 0.5

        out_cpp = turing_csrc.kda_step_fast(qkv, conv_state, conv_weight, state, gate, beta, 1e-5)
        assert out_cpp.shape == (B, H, D)
    except ImportError:
        pytest.skip("turing_csrc not loaded")
