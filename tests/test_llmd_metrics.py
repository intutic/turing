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
    assert metrics["num_gpu_blocks"] == engine.kv_pool.page_limit > 0  # the real paged pool


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
        json_resp = client.get("/metrics?format=json")
        assert json_resp.status_code == 200
        data = json_resp.json()
        assert "llmd_metrics" in data
        assert data["llmd_metrics"]["num_requests_waiting"] == 0
        assert data["llmd_metrics"]["num_requests_running"] == 0


def test_metrics_default_to_prometheus_text_with_prefix_cache_and_speculation():
    """GET /metrics with no Accept header is Prometheus text that prometheus_client's parser reads; after two
    requests sharing a 32-token prefix it reports the prefix cache's real hit rate."""
    from prometheus_client.parser import text_string_to_metric_families

    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))
    with TestClient(create_app(engine)) as client:
        shared = list(range(100, 132))
        for tail in ([7] * 8, [9] * 8):
            r = client.post("/v1/completions", json={"model": "m", "prompt": shared + tail, "max_tokens": 2,
                                                     "temperature": 0.0})
            assert r.status_code == 200
        resp = client.get("/metrics")
        assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/plain")
        samples = {s.name: s for f in text_string_to_metric_families(resp.text) for s in f.samples}

    pc = engine.get_telemetry()["prefix_cache"]
    assert pc["hit_tokens"] == 32  # the second request reused the shared prefix's two pages
    assert samples["turing_prefix_cache_hit_rate"].value == pytest.approx(pc["hit_rate"])
    assert samples["turing_prefix_cache_hit_tokens_total"].value == 32
    assert samples["turing_prefix_cache_pages"].value == pc["cached_pages"] > 0
    assert samples["turing_spec_drafted_tokens_total"].value == engine.spec_drafted
    assert samples["turing_kv_cache_usage_perc"].value == pytest.approx(engine.get_kv_cache_utilization())
    assert samples["turing_cache_config_info"].labels == {"block_size": "16",
                                                          "num_gpu_blocks": str(engine.kv_pool.page_limit)}
