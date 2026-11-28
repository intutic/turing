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


def test_serve_lists_the_served_model_name(monkeypatch):
    """`turing serve --served-model-name`: the id /v1/models lists and /health reports. KServe passes the
    InferenceService's name; without it a model loaded from a directory was listed as its path ('HF-/mnt/models')."""
    import sys
    import turing.cli as cli
    apps = []
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kw: apps.append(app))
    monkeypatch.setattr(sys, "argv", ["turing", "serve", "--model", "test-tiny", "--device", "cpu",
                                      "--served-model-name", "my-model"])
    cli.main()
    with TestClient(apps[0]) as client:
        assert [m["id"] for m in client.get("/v1/models").json()["data"]] == ["my-model"]
        assert client.get("/health").json()["model"] == "my-model"
    assert get_model_config("test-tiny").name != "my-model"  # the preset is a shared object


def test_engine_warm_up_leaves_no_trace():
    """warm_up() sends the requests a fresh process would otherwise make its first clients wait for (first kernels,
    each tuning bucket of the paged prefill kernel, each batch bucket, a decode row riding in a prefill step) and
    leaves the counters, the prefix cache and the KV pool as on a new engine."""
    import dataclasses

    async def _run(context):
        cfg = dataclasses.replace(get_model_config("test-tiny"), max_position_embeddings=context)
        engine = ContinuousBatchEngine(cfg, TuringConfig(device="cpu", max_batch_size=8))
        lengths, add = [], engine.add_request

        async def spy(prompt_tokens, **kw):
            lengths.append(len(prompt_tokens))
            return await add(prompt_tokens=prompt_tokens, **kw)
        engine.add_request = spy
        await engine.start()
        try:
            assert await engine.warm_up() > 0
            assert lengths.count(24) >= 3 + 2 + 4 + 8  # alone three times, then each batch bucket
            assert all(lengths.count(n) == 3 for n in range(56, 257, 32))  # each short bucket's graph
            for n in (300, 1000, 2000):  # a chunk in each tuning bucket, after two cached pages
                assert 32 + n in lengths
            if context > 4096:  # the prompt whose second step carries a decode row
                assert max(lengths) > engine.prefill_chunk_size
            assert max(lengths) + 8 <= context  # nothing the model's context cannot hold
            assert engine.total_tokens_generated == engine.total_requests_completed == 0
            assert not engine.recent_ttft and not engine.recent_itl
            assert engine.prefix_cache.stats() == {"cached_pages": 0, "hit_tokens": 0, "queried_tokens": 0,
                                                   "hit_rate": None}
            assert engine.kv_pool.get_stats()["used_pages"] == 0
            tokens = [t async for t in engine.stream_generate([5, 10, 15], max_new_tokens=4, temperature=0.0)]
            assert len(tokens) == 4 and engine.total_requests_completed == 1
        finally:
            await engine.stop()

    asyncio.run(_run(8192))
    asyncio.run(_run(2048))


def test_serve_warms_up_before_it_listens(monkeypatch):
    """create_app(warmup=True) runs the engine's warm-up in the app's startup, before a request is served."""
    cfg = get_model_config("test-tiny")
    engine = ContinuousBatchEngine(cfg, TuringConfig(device="cpu"))
    calls = []

    async def warm_up():
        calls.append(engine.is_running)
        return 0.0
    monkeypatch.setattr(engine, "warm_up", warm_up)
    with TestClient(create_app(engine, warmup=True)) as client:
        assert calls == [True]
        assert client.get("/health").status_code == 200
    with TestClient(create_app(engine)):
        assert calls == [True]  # off unless asked for


def test_the_app_reads_the_vocabulary_at_startup_not_in_a_request():
    """create_app asks for the thinking tags once, so the tokenizer's vocabulary dict (0.1 s to build for a
    150K-token vocabulary) is not built inside the first chat request, where every running request would wait."""
    class Tok:
        eos_token_id = None
        calls = 0

        def get_vocab(self):
            Tok.calls += 1
            return {"a": 1}

    tok = Tok()
    engine = ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"), tokenizer=tok)
    create_app(engine)
    assert Tok.calls == 1
    from turing.serving.reasoning import think_token_ids
    assert think_token_ids(tok) is None and Tok.calls == 1


def test_image_parts_become_media_markers():
    """With a vision projector (turing serve --backend libllama --mmproj), each image part becomes the media marker in
    its message's text, as llama-server does, and its bytes come back in order; only base64 data: URLs are taken (the
    server never fetches a URL a request names)."""
    import base64
    from turing.serving.engine import RequestRejected
    from turing.serving.generation import openai_media_messages

    def img(data: bytes, mime="image/png"):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}}

    msgs, images = openai_media_messages(
        [{"role": "system", "content": "Be brief."},
         {"role": "user", "content": [img(b"first"), {"type": "text", "text": "Compare"},
                                      {"type": "input_image", "image_url": "data:image/jpeg;base64," +
                                       base64.b64encode(b"second").decode()}]}], "<M>")
    assert images == [b"first", b"second"]
    assert msgs[1] == {"role": "user", "content": "<M>\nCompare\n<M>"} and msgs[0]["content"] == "Be brief."
    for bad in [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png,notbase64"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,***"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}]:
        with pytest.raises(RequestRejected) as e:
            openai_media_messages([{"role": "user", "content": [bad]}], "<M>")
        assert e.value.status_code == 400
