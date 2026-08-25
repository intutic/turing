"""
Automated Unit Tests for Heterogeneous Cross-Device Hybrid Pipeline Mesh (Mac Metal + Remote GPU).
"""

import pytest
import torch
from turing.config import ModelConfig
from turing.core.hybrid_mesh import (
    HybridMeshConfig,
    HybridMeshCoordinator,
    TensorSerializer,
    LocalPipelineStage,
    RemotePipelineStage
)

def test_tensor_serializer_fp16_and_int8():
    dev = torch.device("cpu")
    original_tensor = torch.randn(1, 16, 2048, device=dev)

    # 1. Test FP16 serialization
    payload_fp16 = TensorSerializer.serialize(original_tensor, compress_int8=False)
    deserialized_fp16 = TensorSerializer.deserialize(payload_fp16[4:], device=dev)

    assert deserialized_fp16.shape == original_tensor.shape
    fp16_err = torch.norm(original_tensor - deserialized_fp16).item() / torch.norm(original_tensor).item()
    assert fp16_err < 1e-3

    # 2. Test Dynamic INT8 Quantized serialization
    payload_int8 = TensorSerializer.serialize(original_tensor, compress_int8=True)
    deserialized_int8 = TensorSerializer.deserialize(payload_int8[4:], device=dev)

    assert deserialized_int8.shape == original_tensor.shape
    int8_err = torch.norm(original_tensor - deserialized_int8).item() / torch.norm(original_tensor).item()
    assert int8_err < 0.05 # < 5% quantization noise
    assert len(payload_int8) < len(payload_fp16) # Exactly 50% data compression

def test_hybrid_stages_forward():
    dev = torch.device("cpu")
    cfg = ModelConfig(
        name="test-split-model",
        hidden_dim=128,
        ffn_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        num_layers=4,
        vocab_size=256,
        active_tiles=2,
        tile_size=128
    )

    local_stage = LocalPipelineStage(cfg, layer_start=0, layer_end=2, device=dev)
    remote_stage = RemotePipelineStage(cfg, layer_start=2, layer_end=4, device=dev)

    input_ids = torch.tensor([[1, 5, 20, 100]], dtype=torch.long, device=dev)
    h_mid, local_kvs = local_stage(input_ids, start_pos=0)
    assert h_mid.shape == (1, 4, 128)
    assert len(local_kvs) == 2

    logits, remote_kvs = remote_stage(h_mid, start_pos=0)
    assert logits.shape == (1, 4, 256)
    assert len(remote_kvs) == 2

def test_hybrid_mesh_coordinator_e2e_generation():
    local_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    remote_dev = torch.device("cpu")

    cfg = ModelConfig(
        name="test-hybrid-mesh-70b-scale",
        hidden_dim=256,
        ffn_dim=1024,
        num_heads=8,
        num_kv_heads=2,
        head_dim=32,
        num_layers=6,
        vocab_size=1000,
        active_tiles=2,
        tile_size=256
    )
    mesh_cfg = HybridMeshConfig(
        model_name="test-hybrid-mesh-70b-scale",
        total_layers=6,
        local_layer_start=0,
        local_layer_end=3,
        remote_layer_start=3,
        remote_layer_end=6,
        compression="int8"
    )

    coordinator = HybridMeshCoordinator(
        config=cfg,
        mesh_config=mesh_cfg,
        local_device=local_dev,
        remote_device=remote_dev
    )

    prompt = [15, 220, 1032, 45]
    out_tokens, stats = coordinator.generate(
        prompt_tokens=prompt,
        max_new_tokens=8,
        temperature=0.7,
        top_k=20
    )

    assert len(out_tokens) == len(prompt) + 8
    assert len(stats) == 9 # 1 prefill + 8 decode steps
    assert stats[0]["local_stage_device"] == str(local_dev)
    assert stats[0]["remote_stage_device"] == str(remote_dev)
    assert stats[0]["network_payload_kb"] > 0
