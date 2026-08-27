"""
Unit tests for llm-d compatible metrics exported by Turing Engine.
"""

import pytest
from fastapi.testclient import TestClient
from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app


def test_engine_llmd_metrics_methods():
    cfg = get_model_config("test-tiny")
    tcfg = TuringConfig(device="cpu", max_batch_size=4)
    engine = ContinuousBatchEngine(cfg, tcfg)

    util = engine.get_kv_cache_utilization()
    assert isinstance(util, float)
    assert 0.0 <= util <= 1.0

    metrics = engine.get_llmd_metrics()
    assert "num_requests_waiting" in metrics
    assert "num_requests_running" in metrics
    assert "kv_cache_usage_perc" in metrics
    assert "block_size" in metrics
    assert "num_gpu_blocks" in metrics
    assert "total_tokens_generated" in metrics

    assert metrics["num_requests_waiting"] == 0
    assert metrics["num_requests_running"] == 0
    assert metrics["block_size"] == 16
    assert metrics["num_gpu_blocks"] == 256


def test_prometheus_llmd_metrics_exposition():
    cfg = get_model_config("test-tiny")
    tcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, tcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        # 1. Prometheus text exposition
        resp = client.get("/metrics", headers={"Accept": "text/plain"})
        assert resp.status_code == 200
        text = resp.text

        assert "turing_num_requests_waiting" in text
        assert "turing_num_requests_running" in text
        assert "turing_kv_cache_usage_perc" in text
        assert "turing_cache_config_info" in text
        assert 'block_size="16"' in text

        # 2. JSON metrics payload
        json_resp = client.get("/metrics", headers={"Accept": "application/json"})
        assert json_resp.status_code == 200
        data = json_resp.json()
        assert "llmd_metrics" in data
        assert data["llmd_metrics"]["num_requests_waiting"] == 0
        assert data["llmd_metrics"]["num_requests_running"] == 0
