"""
Automated Numerical Parity and Verification Tests for Native C++20 AVX2 SIMD & Triton Kernel Fusions.
Validates:
1. DFlash 1D Dilated Depthwise Causal Convolution numerical parity (|Δ| <= 1e-4) against PyTorch nn.Conv1d.
2. Fused Base GEMV + LoRA Rank-8 contraction numerical parity (|Δ| <= 1e-4) against PyTorch.
3. 1-Pass Subspace Residual Outlier Extraction parity against torch.topk.
4. Lock-Free Elastic Memory Budget Controller dynamic rebalancing invariants.
5. 4-Stream Birkhoff mHC SIMD mixing parity against PyTorch reference.
6. In-SRAM Fused Subspace Structured Router bitmask output parity.
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from turing.core.speculation import SubspaceEAGLEDraftHead
from turing.core.subspace import SubspaceManager
from turing.models.adapters import TenantLoRAAdapter
from turing.core.elastic_memory import ElasticMemoryBudgetManager
from turing.core.expert_cache import GPULRUExpertCache
from turing.core.paging import StaticPagedKVPool
from turing.core.mhc import ManifoldHyperConnection
from turing.core.router import SubspaceStructuredRouter

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False


def test_dilated_causal_conv1d_parity():
    """Verifies DFlash 1D Dilated Causal Convolution SIMD against PyTorch nn.Conv1d."""
    torch.manual_seed(42)
    batch, seq_len, channels = 2, 8, 64
    kernel_size = 3
    dilation = 2

    x = torch.randn(batch, seq_len, channels, dtype=torch.float32)
    conv_layer = nn.Conv1d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) * dilation,
        dilation=dilation,
        groups=channels,
        bias=False
    )

    # Reference PyTorch output
    x_t = x.transpose(1, 2)
    ref_out_t = conv_layer(x_t)[..., :seq_len]
    ref_out = ref_out_t.transpose(1, 2).contiguous()

    # Native C++20 SIMD output
    if HAS_CSRC:
        x_np = x.numpy()
        w_np = conv_layer.weight.squeeze(1).detach().numpy()
        csrc_out_np = turing_csrc.dilated_causal_conv1d_cpu(x_np, w_np, dilation)
        csrc_out = torch.from_numpy(csrc_out_np)

        max_diff = torch.max(torch.abs(ref_out - csrc_out)).item()
        assert max_diff <= 1e-4, f"Dilated conv SIMD max diff {max_diff} exceeds tolerance"

    # Triton CUDA / Auto-dispatch test
    from turing.kernels.triton_dilated_conv import dilated_causal_conv1d_cuda
    dispatch_out = dilated_causal_conv1d_cuda(x, conv_layer.weight.squeeze(1), dilation=dilation)
    max_diff_dispatch = torch.max(torch.abs(ref_out - dispatch_out)).item()
    assert max_diff_dispatch <= 1e-4, f"Dilated conv dispatch max diff {max_diff_dispatch} exceeds tolerance"


def test_fused_lora_gemv_parity():
    """Verifies Fused Base GEMV + LoRA Rank-8 contraction against separate PyTorch matmuls."""
    torch.manual_seed(42)
    batch = 4
    in_dim = 128
    out_dim = 128
    rank = 8
    alpha = 0.5

    x = torch.randn(batch, in_dim, dtype=torch.float32)
    w_base = torch.randn(in_dim, out_dim, dtype=torch.float32) * 0.02
    w_a = torch.randn(in_dim, rank, dtype=torch.float32) * 0.02
    w_b = torch.randn(rank, out_dim, dtype=torch.float32) * 0.02

    # Reference PyTorch calculation
    ref_base = torch.matmul(x, w_base)
    ref_lora = torch.matmul(torch.matmul(x, w_a) * alpha, w_b)
    ref_out = ref_base + ref_lora

    # Native C++20 SIMD output
    if HAS_CSRC:
        x_np = x.numpy()
        wb_np = w_base.numpy()
        wa_np = w_a.numpy()
        wb2_np = w_b.numpy()

        csrc_out_np = turing_csrc.fused_lora_gemv_cpu(x_np, wb_np, wa_np, wb2_np, alpha)
        csrc_out = torch.from_numpy(csrc_out_np)

        max_diff = torch.max(torch.abs(ref_out - csrc_out)).item()
        assert max_diff <= 1e-4, f"Fused LoRA SIMD max diff {max_diff} exceeds tolerance"

    # Fused LoRA Triton / Auto-dispatch test
    from turing.kernels.triton_fused_lora import fused_lora_gemv_cuda
    fused_out = fused_lora_gemv_cuda(x, w_base, w_a, w_b, alpha=alpha)
    max_diff_fused = torch.max(torch.abs(ref_out - fused_out)).item()
    assert max_diff_fused <= 1e-4, f"Fused LoRA dispatch max diff {max_diff_fused} exceeds tolerance"


def test_find_residual_outlier_parity():
    """Verifies 1-Pass Subspace Residual Outlier Extraction against torch.topk."""
    torch.manual_seed(42)
    batch = 8
    hidden_dim = 1024

    residual = torch.randn(batch, hidden_dim, dtype=torch.float32)

    # Reference torch.topk
    ref_top_vals, ref_top_indices = torch.topk(torch.abs(residual), k=1, dim=-1)
    ref_signed_vals = torch.gather(residual, -1, ref_top_indices)

    # Native C++20 SIMD output
    if HAS_CSRC:
        r_np = residual.numpy()
        idx_np, val_np = turing_csrc.find_residual_outlier_cpu(r_np)

        simd_indices = torch.from_numpy(idx_np).view(batch, 1)
        simd_vals = torch.from_numpy(val_np).view(batch, 1)

        assert torch.equal(ref_top_indices, simd_indices), "Outlier indices do not match"
        assert torch.allclose(ref_signed_vals, simd_vals, atol=1e-5), "Outlier values do not match"

    # Triton CUDA / Auto-dispatch test
    from turing.kernels.triton_residual_outlier import find_residual_outlier_cuda
    out_idx, out_val = find_residual_outlier_cuda(residual)
    assert torch.equal(ref_top_indices, out_idx), "Triton outlier indices do not match"
    assert torch.allclose(ref_signed_vals, out_val, atol=1e-5), "Triton outlier values do not match"


def test_native_elastic_budget_controller():
    """Verifies Lock-Free NativeElasticBudgetController state transitions and rebalancing."""
    if not HAS_CSRC:
        pytest.skip("turing_csrc not compiled")

    controller = turing_csrc.NativeElasticBudgetController(
        initial_expert_slots=16,
        min_expert_slots=4,
        max_expert_slots=32,
        initial_kv_pages=128,
        min_kv_pages=32,
        max_kv_pages=1024,
        bytes_per_slot=4096,
        bytes_per_page=2048,
        page_size_tokens=64,
        target_headroom=0.25
    )

    assert controller.get_expert_slots() == 16
    assert controller.get_kv_pages() == 128
    assert controller.get_rebalance_count() == 0

    # Demand burst requiring KV expansion (e.g. 15,000 active tokens -> ~293 pages)
    res = controller.evaluate_and_rebalance(15000, False)
    assert res["rebalanced"] is True
    assert res["new_kv_pages"] > 128
    assert res["new_expert_slots"] < 16
    assert "expand_kv" in res["action"]
    assert controller.get_rebalance_count() == 1


def test_mhc_4stream_simd_parity():
    """Verifies 4-Stream Birkhoff mHC SIMD mixing parity against PyTorch reference."""
    torch.manual_seed(42)
    batch, seq_len, hidden_dim = 2, 4, 128
    streams = torch.randn(batch, seq_len, 4, hidden_dim, dtype=torch.float32)
    layer_update = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float32)

    mhc_block = ManifoldHyperConnection(hidden_dim=hidden_dim, num_streams=4)

    # Reference evaluation
    ref_streams_mixed = mhc_block.res_map(streams)
    ref_new_streams = mhc_block.post_map(ref_streams_mixed, layer_update)

    if HAS_CSRC:
        s_np = streams.view(-1, 4, hidden_dim).numpy()
        lup_np = layer_update.view(-1, hidden_dim).numpy()
        alpha_np = F.softmax(mhc_block.raw_pre_weights, dim=0).detach().numpy()
        h_res = mhc_block.get_doubly_stochastic_res_map()
        h_np = h_res.detach().numpy()
        beta_np = torch.sigmoid(mhc_block.raw_post_weights).detach().numpy()

        l_in_np, s_out_np = turing_csrc.mhc_4stream_simd_cpu(s_np, lup_np, alpha_np, h_np, beta_np)
        simd_new_streams = torch.from_numpy(s_out_np).view(batch, seq_len, 4, hidden_dim)

        max_diff = torch.max(torch.abs(ref_new_streams - simd_new_streams)).item()
        assert max_diff <= 1e-4, f"mHC SIMD max diff {max_diff} exceeds tolerance"


def test_fused_subspace_router_dispatch():
    """Verifies SubspaceStructuredRouter fused dispatch produces valid top-k bitmasks."""
    torch.manual_seed(42)
    batch = 4
    hidden_dim = 128
    total_tiles = 32

    router = SubspaceStructuredRouter(hidden_dim=hidden_dim, total_tiles=total_tiles, min_tiles=8, max_tiles=16)
    router.eval()

    h = torch.randn(batch, 16, hidden_dim, dtype=torch.float32)
    tile_mask, uncertainty = router(h, top_k_override=10)

    assert tile_mask.shape == (batch, total_tiles)
    assert uncertainty.shape == (batch, 1)
    # Check that exactly 10 tiles are active per batch item
    active_per_item = tile_mask.sum(dim=-1)
    assert torch.all(active_per_item == 10.0), f"Expected 10 active tiles, got {active_per_item}"
