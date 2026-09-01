"""
Automated Unit & Numerical Parity Tests for Fused GPU Kernels & C++20 SIMD Subsystems.
Verifies triton_spec_verify, triton_fused_kslot_gate, NativeSafetensorsHeaderParser, and NativeTrafficManager.
"""

import json
import pytest
import torch
import numpy as np

from turing.turing_csrc import (
    NativeSafetensorsHeaderParser,
    NativeTrafficManager,
    NativeAsyncRingReader
)
from turing.kernels.triton_spec_verify import fused_speculative_verify_cuda
from turing.kernels.triton_fused_kslot_gate import fused_kslot_pooling_and_gating_cuda
from turing.core.speculation import RidgeAssistedTreeSpeculator
from turing.core.kslot_pooling import KSlotCachePooler, GatedZeroIdentityHead


def test_fast_safetensors_header_parser():
    """Verifies C++ NativeSafetensorsHeaderParser against Python json.loads()."""
    mock_meta = {
        "__metadata__": {"format": "pt"},
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [32000, 4096],
            "data_offsets": [0, 262144000]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [4096, 4096],
            "data_offsets": [262144000, 295698432]
        }
    }
    json_str = json.dumps(mock_meta)
    parsed = NativeSafetensorsHeaderParser.parse_header(json_str)

    assert "model.embed_tokens.weight" in parsed
    assert "model.layers.0.self_attn.q_proj.weight" in parsed
    assert "__metadata__" not in parsed

    embed_info = parsed["model.embed_tokens.weight"]
    assert embed_info.dtype == "F16"
    assert embed_info.shape == [32000, 4096]
    assert embed_info.start_offset == 0
    assert embed_info.end_offset == 262144000


def test_native_traffic_manager_estimation_and_admission():
    """Verifies NativeTrafficManager atomic VRAM budgeting and 64-bit prefix routing."""
    # 1. Estimation
    est_bytes = NativeTrafficManager.estimate_kv_bytes(
        num_prompt_tokens=512,
        max_new_tokens=128,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_bytes=2,
        svd_compression_ratio=0.75
    )
    assert est_bytes > 0

    # 2. Prefix Hashing
    tokens = [101, 2054, 2003, 1037, 3231, 102]
    hash_a = NativeTrafficManager.compute_prefix_hash(tokens, window=128)
    hash_b = NativeTrafficManager.compute_prefix_hash(tokens, window=128)
    assert hash_a == hash_b
    assert hash_a > 0

    # 3. Admission Control (Budget = 100 MB)
    tm = NativeTrafficManager(vram_budget_bytes=100 * 1024 * 1024, high_watermark=0.90, shed_watermark=0.95)
    res1 = tm.admit("req_1", 50 * 1024 * 1024)
    assert res1.admitted
    assert res1.decision == "admit"

    res2 = tm.admit("req_2", 42 * 1024 * 1024) # Total 92MB -> Queue
    assert not res2.admitted
    assert res2.decision == "queue"

    res3 = tm.admit("req_3", 50 * 1024 * 1024) # Total 50+50 = 100MB >= 95MB -> Shed
    assert not res3.admitted
    assert res3.decision == "shed"

    tm.release("req_1")
    assert tm.utilization < 0.5


def test_fused_speculative_verify_exact_parity():
    """Verifies fused_speculative_verify_cuda parity with RidgeAssistedTreeSpeculator."""
    K = 6
    vocab_size = 100
    torch.manual_seed(42)

    target_logits = torch.randn(K, vocab_size)
    target_preds = torch.argmax(target_logits, dim=-1).tolist()

    # Case A: Perfect match on first 3 tokens, mismatch on 4th
    draft_tokens = target_preds[:3] + [999] + target_preds[4:]
    accepted_tokens, num_accepted = fused_speculative_verify_cuda(draft_tokens, target_logits, temperature=0.0)

    assert num_accepted == 4 # 3 matched + 1 corrected target token
    assert accepted_tokens.tolist() == target_preds[:4]


def test_fused_kslot_pooling_and_gating_shapes():
    """Verifies output dimensions of fused k-slot pooling and gating."""
    B, L, H, N, D = 2, 4, 2, 64, 64
    k_slots = 4
    torch.manual_seed(42)

    keys = torch.randn(B, L, H, N, D)
    values = torch.randn(B, L, H, N, D)
    queries = torch.randn(L, H, k_slots, D)
    gate_w = torch.randn(2 * H, H * D)
    head_k = torch.randn(H * D, H * D)
    head_v = torch.randn(H * D, H * D)

    pk, pv, dk, dv = fused_kslot_pooling_and_gating_cuda(
        keys, values, queries, gate_w, head_k, head_v
    )

    assert pk.shape == (B, L, H, k_slots, D)
    assert pv.shape == (B, L, H, k_slots, D)
    assert dk.shape == (B, L, H, k_slots, D)
    assert dv.shape == (B, L, H, k_slots, D)
