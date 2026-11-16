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
    """X-Turing-Draft-Tokens is honoured; the response headers report what the server really did."""
    payload = {
        "model": "test-turing-model",
        "messages": [{"role": "user", "content": "Hello Turing"}],
        "max_tokens": 8,
        "temperature": 0.5
    }
    resp = test_client.post("/v1/chat/completions", json=payload, headers={"X-Turing-Draft-Tokens": "4"})
    assert resp.status_code == 200
    assert resp.headers["X-Turing-Draft-Tokens"] == "4"
    assert resp.headers["X-Turing-KV-Cache-Dtype"] == "native"
    assert resp.headers["X-Turing-Model"] == "test-turing-model"
    assert int(resp.headers["X-Turing-Cached-Tokens"]) >= 0
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


def test_removed_per_request_knobs_are_rejected(test_client):
    """Sparsity and SVD KV were never applied per request: they are load-time settings, and saying so beats
    echoing a value that changed nothing."""
    payload = {"model": "test-turing-model", "messages": [{"role": "user", "content": "Benchmark test"}],
               "max_tokens": 4, "draft_tokens": 2}
    for body, headers, flag in [({"sparsity_ratio": 0.625}, {}, "--sparsity"),
                                ({"use_svd_kv": False}, {}, "--kv-cache-dtype"),
                                ({}, {"X-Turing-Sparsity": "0.450"}, "--sparsity"),
                                ({}, {"X-Turing-SVD-KV": "1"}, "--kv-cache-dtype")]:
        resp = test_client.post("/v1/chat/completions", json={**payload, **body}, headers=headers)
        assert resp.status_code == 400 and flag in resp.json()["error"]["message"]
    resp = test_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200 and resp.headers["X-Turing-Draft-Tokens"] == "2"


def test_anthropic_messages_with_turing_headers(test_client):
    payload = {
        "model": "test-turing-model",
        "messages": [{"role": "user", "content": "Anthropic prompt"}],
        "max_tokens": 6,
    }
    resp = test_client.post("/v1/messages", json=payload, headers={"X-Turing-Lane": "batch"})
    assert resp.status_code == 200
    assert resp.headers["X-Turing-Lane"] == "batch"
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    resp = test_client.post("/v1/messages", json=payload, headers={"X-Turing-Sparsity": "0.500"})
    assert resp.status_code == 400 and resp.json()["error"]["type"] == "invalid_request_error"
