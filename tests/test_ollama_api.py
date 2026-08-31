"""
Unit tests for Native Ollama REST API compatibility layer (/api/*).
"""

import json
import pytest
from fastapi.testclient import TestClient
from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app
from turing.serving.ollama_api import OllamaAPIHandler, OllamaGenerateRequest, OllamaChatRequest


@pytest.fixture
def test_app():
    cfg = get_model_config("test-tiny")
    jcfg = TuringConfig(device="cpu", max_batch_size=4)
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)
    return app


def test_ollama_tags_endpoint(test_app):
    with TestClient(test_app) as client:
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 1
        m = data["models"][0]
        assert "tiny" in m["name"].lower()
        assert m["details"]["format"] == "subspace"


def test_ollama_version_endpoint(test_app):
    with TestClient(test_app) as client:
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.7.0"


def test_ollama_ps_endpoint(test_app):
    with TestClient(test_app) as client:
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 1
        assert "tiny" in data["models"][0]["name"].lower()
        assert data["models"][0]["size_vram"] > 0


def test_ollama_show_endpoint(test_app):
    with TestClient(test_app) as client:
        resp = client.post("/api/show", json={"model": "test-tiny"})
        assert resp.status_code == 200
        data = resp.json()
        assert "modelfile" in data
        assert "parameters" in data
        assert "model_info" in data
        assert data["model_info"]["general.architecture"] == "subspace_causal_lm"


def test_ollama_generate_non_streaming(test_app):
    with TestClient(test_app) as client:
        payload = {
            "model": "test-tiny",
            "prompt": "Explain quantum computing in one sentence.",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 16
            }
        }
        resp = client.post("/api/generate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test-tiny"
        assert data["done"] is True
        assert data["done_reason"] == "stop"
        assert "response" in data
        assert "context" in data
        assert data["prompt_eval_count"] > 0
        assert data["eval_count"] > 0


def test_ollama_generate_streaming(test_app):
    with TestClient(test_app) as client:
        payload = {
            "model": "test-tiny",
            "prompt": "Hello",
            "stream": True,
            "options": {
                "num_predict": 8
            }
        }
        with client.stream("POST", "/api/generate", json=payload) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines() if line]
            assert len(lines) >= 2
            # Parse first chunk
            first_chunk = json.loads(lines[0])
            assert first_chunk["model"] == "test-tiny"
            # Parse final chunk
            final_chunk = json.loads(lines[-1])
            assert final_chunk["done"] is True
            assert final_chunk["done_reason"] == "stop"


def test_ollama_chat_non_streaming(test_app):
    with TestClient(test_app) as client:
        payload = {
            "model": "test-tiny",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "stream": False,
            "options": {
                "num_predict": 12
            }
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "test-tiny"
        assert data["done"] is True
        assert data["message"]["role"] == "assistant"
        assert len(data["message"]["content"]) > 0


def test_ollama_chat_streaming(test_app):
    with TestClient(test_app) as client:
        payload = {
            "model": "test-tiny",
            "messages": [
                {"role": "user", "content": "Hi"}
            ],
            "stream": True,
            "options": {
                "num_predict": 6
            }
        }
        with client.stream("POST", "/api/chat", json=payload) as resp:
            assert resp.status_code == 200
            lines = [line for line in resp.iter_lines() if line]
            assert len(lines) >= 2
            final_chunk = json.loads(lines[-1])
            assert final_chunk["done"] is True


def test_ollama_embed_endpoint(test_app):
    with TestClient(test_app) as client:
        resp = client.post("/api/embeddings", json={"prompt": "Embed this text representation"})
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding" in data
        assert len(data["embedding"]) == 32
