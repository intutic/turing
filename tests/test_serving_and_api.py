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
        res_metrics_json = client.get("/metrics", headers={"Accept": "application/json"})
        assert res_metrics_json.status_code == 200
        data_m = res_metrics_json.json()
        assert "telemetry" in data_m
        assert data_m["telemetry"]["total_tokens_generated"] >= 4
