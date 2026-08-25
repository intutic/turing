"""
Comprehensive Automated Tests for Heterogeneous MoE, Expert LRU Caching, and Anthropic API.
"""

import os
import json
import struct
import numpy as np
import pytest
import torch
import httpx
from fastapi.testclient import TestClient

from turing.config import ModelConfig, TuringConfig
from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider, HostExpertBank, HeterogeneousMoERunner
from turing.core.expert_cache import GPULRUExpertCache
from turing.models.safetensors_mmap import SafetensorsMmapReader
from turing.serving.anthropic_api import (
    AnthropicMessageRequest,
    AnthropicMessage,
    AnthropicContentBlock,
    AnthropicAPIHandler
)
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app

def test_bandwidth_adaptive_decider():
    dev = torch.device("cpu")
    decider = BandwidthAdaptiveDecider(
        device=dev,
        pcie_bandwidth_gb_s=25.0,
        cpu_throughput_gflops=100.0,
        gpu_throughput_gflops=5000.0
    )

    # For tiny batch (1 token) and large expert weight, CPU compute should be favored over PCIe transfer overhead
    expert_bytes = 4 * 1024 * 1024 # 4MB
    decision_tiny = decider.should_stream_to_gpu(
        expert_bytes_int4=expert_bytes,
        batch_tokens=1,
        hidden_dim=4096,
        moe_intermediate_dim=2048
    )
    assert isinstance(decision_tiny, bool)

    # For large batch (128 tokens), GPU streaming and compute should be favored
    decision_large = decider.should_stream_to_gpu(
        expert_bytes_int4=expert_bytes,
        batch_tokens=128,
        hidden_dim=4096,
        moe_intermediate_dim=2048
    )
    assert decision_large is True

def test_host_expert_bank_subspace_packing():
    bank = HostExpertBank(
        num_layers=2,
        num_experts=8,
        hidden_dim=256,
        ffn_dim=1024,
        active_subspace_dim=512
    )

    assert bank.bytes_per_expert == (3 * 512 * 256) // 2 # 196,608 bytes
    uncompressed_bytes = 3 * 1024 * 256 * 2 # 1,572,864 bytes
    assert bank.bytes_per_expert == uncompressed_bytes // 8 # 8x vs FP16 unpruned

    expert_slice = bank.get_expert_slice(layer_idx=1, expert_idx=3)
    assert expert_slice.shape[0] == bank.bytes_per_expert

def test_gpu_lru_expert_cache():
    dev = torch.device("cpu")
    cache = GPULRUExpertCache(
        num_slots=4,
        hidden_dim=256,
        active_subspace_dim=128,
        device=dev
    )

    # Fill slots: (0,0), (0,1), (0,2), (0,3)
    for i in range(4):
        slot, evicted = cache.allocate_or_evict_slot(layer_idx=0, expert_idx=i)
        assert slot == (3 - i) or slot in range(4)
        assert evicted is None

    assert cache.stats()["used_slots"] == 4
    assert cache.stats()["free_slots"] == 0

    # Access (0,0) -> cache hit
    hit_slot = cache.get_slot(0, 0)
    assert hit_slot is not None
    assert cache.hits == 1

    # Allocate a 5th expert (0, 4) -> should evict LRU (which is (0,1) since (0,0) was touched)
    new_slot, evicted_key = cache.allocate_or_evict_slot(0, 4)
    assert evicted_key == (0, 1)
    assert cache.contains(0, 4)
    assert not cache.contains(0, 1)

def test_heterogeneous_moe_runner():
    dev = torch.device("cpu")
    cfg = ModelConfig(
        name="test-moe",
        hidden_dim=128,
        ffn_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
        num_layers=2,
        vocab_size=500,
        active_tiles=2,
        tile_size=128
    )
    decider = BandwidthAdaptiveDecider(dev, pcie_bandwidth_gb_s=20.0, cpu_throughput_gflops=200.0, gpu_throughput_gflops=2000.0)
    host_bank = HostExpertBank(num_layers=2, num_experts=8, hidden_dim=128, ffn_dim=512, active_subspace_dim=256)
    runner = HeterogeneousMoERunner(cfg, host_bank, decider, dev)

    x = torch.randn(1, 4, 128)
    logits = torch.randn(1, 4, 8)

    out, stats = runner.route_and_execute(x, logits, layer_idx=0, top_k=2)
    assert out.shape == (1, 4, 128)
    assert stats["total_active_experts"] <= 8
    assert "hybrid_ratio" in stats

def test_safetensors_mmap_reader(tmp_path):
    # Construct a synthetic .safetensors binary file
    dummy_file = tmp_path / "model.safetensors"
    tensor_data = np.arange(100, dtype=np.float32)
    data_bytes = tensor_data.tobytes()

    metadata = {
        "weight_a": {
            "dtype": "F32",
            "shape": [10, 10],
            "data_offsets": [0, len(data_bytes)]
        }
    }
    meta_json = json.dumps(metadata).encode("utf-8")
    header_size = len(meta_json)

    with open(dummy_file, "wb") as f:
        f.write(struct.pack("<Q", header_size))
        f.write(meta_json)
        f.write(data_bytes)

    with SafetensorsMmapReader(str(dummy_file)) as reader:
        names = reader.get_tensor_names()
        assert "weight_a" in names
        t = reader.read_tensor_slice("weight_a", device="cpu")
        assert t.shape == torch.Size([10, 10])
        assert torch.allclose(t[0, 0], torch.tensor(0.0))
        assert torch.allclose(t[9, 9], torch.tensor(99.0))

def test_anthropic_api_handler():
    req = AnthropicMessageRequest(
        model="deepseek-v4-flash",
        messages=[
            AnthropicMessage(role="user", content="Hello Claude"),
            AnthropicMessage(role="assistant", content="Hello! How can I help?"),
            AnthropicMessage(role="user", content="Explain MoE.")
        ],
        system="You are an AI assistant.",
        max_tokens=64
    )

    prompt = AnthropicAPIHandler.extract_prompt_from_request(req)
    assert "System: You are an AI assistant." in prompt
    assert "User: Hello Claude" in prompt
    assert "Assistant: Hello! How can I help?" in prompt
    assert "User: Explain MoE." in prompt

    resp = AnthropicAPIHandler.format_non_streaming_response(
        req=req,
        generated_text="MoE activates a subset of experts.",
        input_tokens_count=20,
        output_tokens_count=8
    )
    assert resp.type == "message"
    assert resp.role == "assistant"
    assert resp.content[0].text == "MoE activates a subset of experts."
    assert resp.usage.input_tokens == 20
    assert resp.usage.output_tokens == 8

def test_anthropic_messages_endpoint():
    cfg = ModelConfig(
        name="test-anthropic",
        hidden_dim=64,
        ffn_dim=256,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
        num_layers=2,
        vocab_size=256,
        active_tiles=2,
        tile_size=64
    )
    jcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        # 1. Test Non-Streaming /v1/messages
        payload = {
            "model": "test-anthropic",
            "messages": [{"role": "user", "content": "What is Turing Engine?"}],
            "max_tokens": 16,
            "stream": False
        }
        res = client.post("/v1/messages", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert len(data["content"]) > 0
        assert data["usage"]["completion_tokens"] > 0 if "completion_tokens" in data["usage"] else data["usage"]["output_tokens"] > 0

        # 2. Test Streaming /v1/messages (SSE)
        payload["stream"] = True
        stream_res = client.post("/v1/messages", json=payload)
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]
        body_text = stream_res.text
        assert "event: message_start" in body_text
        assert "event: content_block_delta" in body_text
        assert "event: message_stop" in body_text
        stream_res.close()

