"""
Unit tests for /v1/completions and /render tokenization endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app


def test_render_endpoints():
    cfg = get_model_config("test-tiny")
    tcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, tcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        # 1. /v1/completions/render with string
        resp = client.post("/v1/completions/render", json={"prompt": "Hello world!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data
        assert "count" in data
        assert data["count"] == len(data["tokens"])
        assert data["count"] > 0

        # 2. /v1/completions/render with token list
        resp_toks = client.post("/v1/completions/render", json={"prompt": [10, 20, 30]})
        assert resp_toks.status_code == 200
        assert resp_toks.json()["tokens"] == [10, 20, 30]
        assert resp_toks.json()["count"] == 3

        # 3. /v1/chat/completions/render with messages
        resp_chat = client.post(
            "/v1/chat/completions/render",
            json={"messages": [{"role": "user", "content": "Tell me a story"}]},
        )
        assert resp_chat.status_code == 200
        data_chat = resp_chat.json()
        assert "tokens" in data_chat
        assert data_chat["count"] > 0


def test_v1_completions_endpoint():
    cfg = get_model_config("test-tiny")
    tcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, tcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        # Non-streaming text completion
        resp = client.post(
            "/v1/completions",
            json={"model": cfg.name, "prompt": "Once upon a time", "max_tokens": 4, "stream": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) > 0
        assert "text" in data["choices"][0]
        assert data["usage"]["completion_tokens"] == 4

        # Streaming text completion
        resp_stream = client.post(
            "/v1/completions",
            json={"model": cfg.name, "prompt": "Once upon a time", "max_tokens": 4, "stream": True},
        )
        assert resp_stream.status_code == 200
        assert "text/event-stream" in resp_stream.headers["content-type"]
        assert "[DONE]" in resp_stream.text
