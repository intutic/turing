import pytest
from fastapi.testclient import TestClient
from turing.config import ModelConfig, TuringConfig
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app

@pytest.fixture
def test_client():
    cfg = ModelConfig(
        name="test-turing-model",
        hidden_dim=256,
        ffn_dim=512,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        num_layers=2,
        vocab_size=1000,
        tile_size=64,
        active_tiles=4
    )
    turing_cfg = TuringConfig(device="cpu", max_batch_size=8)
    engine = ContinuousBatchEngine(cfg, turing_cfg)
    app = create_app(engine)
    with TestClient(app) as client:
        yield client

def test_chat_completions_with_turing_headers(test_client):
    payload = {
        "model": "test-turing-model",
        "messages": [{"role": "user", "content": "Hello Turing"}],
        "max_tokens": 8,
        "temperature": 0.5
    }
    headers = {
        "X-Turing-Sparsity": "0.450",
        "X-Turing-SVD-KV": "1",
        "X-Turing-Draft-Tokens": "4"
    }
    resp = test_client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert "X-Turing-Sparsity" in resp.headers
    assert resp.headers["X-Turing-Sparsity"] == "0.450"
    assert resp.headers["X-Turing-SVD-KV"] == "1"
    assert resp.headers["X-Turing-Model"] == "test-turing-model"
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0

def test_chat_completions_with_body_parameters(test_client):
    payload = {
        "model": "test-turing-model",
        "messages": [{"role": "user", "content": "Benchmark test"}],
        "max_tokens": 4,
        "sparsity_ratio": 0.625,
        "use_svd_kv": False,
        "draft_tokens": 2
    }
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers["X-Turing-Sparsity"] == "0.625"
    assert resp.headers["X-Turing-SVD-KV"] == "0"

def test_anthropic_messages_with_turing_headers(test_client):
    payload = {
        "model": "test-turing-model",
        "messages": [{"role": "user", "content": "Anthropic prompt"}],
        "max_tokens": 6,
        "sparsity_ratio": 0.500
    }
    headers = {
        "X-Turing-Sparsity": "0.500",
        "X-Turing-SVD-KV": "1"
    }
    resp = test_client.post("/v1/messages", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.headers["X-Turing-Sparsity"] == "0.500"
    assert resp.headers["X-Turing-SVD-KV"] == "1"
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
