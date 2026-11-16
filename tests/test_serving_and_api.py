import pytest
import asyncio
from fastapi.testclient import TestClient

from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.server import create_app

def test_continuous_batch_engine_streaming():
    async def _run():
        cfg = get_model_config("test-tiny")
        jcfg = TuringConfig(device="cpu", max_batch_size=4)
        engine = ContinuousBatchEngine(cfg, jcfg)

        await engine.start()
        try:
            tokens = []
            stream = engine.stream_generate(prompt_tokens=[5, 10, 15], max_new_tokens=4, temperature=0.0)
            async for tok in stream:
                tokens.append(tok)
            assert len(tokens) == 4
        finally:
            await engine.stop()

    asyncio.run(_run())

def test_fastapi_server_endpoints():
    cfg = get_model_config("test-tiny")
    jcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)

    with TestClient(app) as client:
        # Health check
        res_health = client.get("/health")
        assert res_health.status_code == 200
        data_health = res_health.json()
        assert data_health["status"] == "healthy"

        # Models list
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200

        # Chat completions non-streaming
        payload = {
            "model": cfg.name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 4,
            "stream": False
        }
        res_chat = client.post("/v1/chat/completions", json=payload)
        assert res_chat.status_code == 200
        data_chat = res_chat.json()
        assert "choices" in data_chat
        assert len(data_chat["choices"]) > 0

        # Prometheus text/plain metrics
        res_metrics_txt = client.get("/metrics", headers={"Accept": "text/plain"})
        assert res_metrics_txt.status_code == 200
        assert "turing_serving_throughput_tok_per_sec" in res_metrics_txt.text
        assert "turing_total_tokens_generated" in res_metrics_txt.text

        # JSON metrics
        res_metrics_json = client.get("/metrics?format=json")
        assert res_metrics_json.status_code == 200
        data_m = res_metrics_json.json()
        assert "telemetry" in data_m
        assert data_m["telemetry"]["total_tokens_generated"] >= 4


def _overloaded_app(**pool):
    import dataclasses
    cfg = dataclasses.replace(get_model_config("test-tiny"), **pool)  # the registry's object is shared
    engine = ContinuousBatchEngine(cfg, TuringConfig(device="cpu", max_batch_size=4))
    return engine


def test_rejections_return_400_429_503_with_retry_after():
    """Pool full -> 503, admission QUEUE -> 429, never-fits -> 400; streaming requests are rejected before any 200."""
    from turing.serving.traffic import AdmissionController
    engine = _overloaded_app(kv_pool_gib=0.002)
    engine.max_waiting = 0  # no waiting room: a full pool answers 503 at once
    pool = engine.kv_pool
    with TestClient(create_app(engine)) as client:
        too_big = pool.page_limit * 16 + 1
        r = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": too_big, "stream": True})
        assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"

        held = pool.alloc_pages(pool.page_limit)  # another tenant holds every page
        try:
            for body, path in [({"model": "m", "prompt": "hi", "max_tokens": 4}, "/v1/completions"),
                               ({"model": "m", "prompt": "hi", "max_tokens": 4, "stream": True}, "/v1/completions"),
                               ({"model": "m", "max_tokens": 4, "stream": True,
                                 "messages": [{"role": "user", "content": "hi"}]}, "/v1/messages"),
                               ({"model": "m", "prompt": "hi", "stream": True}, "/api/generate")]:
                r = client.post(path, json=body)
                assert r.status_code == 503, (path, r.status_code, r.text)
                assert r.headers["retry-after"] == "1"
            assert client.post("/v1/messages", json={"model": "m", "max_tokens": 4, "messages": [
                {"role": "user", "content": "hi"}]}).json()["error"]["type"] == "overloaded_error"
        finally:
            pool.decref(held)
        assert pool.used_pages == 0

    engine = _overloaded_app()
    app = create_app(engine, admission_controller=AdmissionController(1, high_watermark=0.0, shed_watermark=float("inf")))
    with TestClient(app) as client:
        r = client.post("/v1/completions", json={"model": "m", "prompt": "hi", "max_tokens": 4})
        assert r.status_code == 429 and r.headers["retry-after"] == "2"
        assert r.json()["error"]["type"] == "rate_limit_exceeded"
