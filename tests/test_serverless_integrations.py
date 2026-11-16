"""
RunPod, Ray Serve and Modal recipes (integrations/) on a real engine. Ray, RunPod and Modal are not
needed: each recipe's request path is called directly.
"""

import asyncio
import json

import pytest
from starlette.requests import Request

from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine

from integrations.common import InProcessTuring, load_engine
from integrations.modal import modal_turing
from integrations.ray.ray_serve_turing import forward
from integrations.runpod.runpod_handler import RunPodTuringWorker


def _tiny_engine():
    return ContinuousBatchEngine(get_model_config("test-tiny"), TuringConfig(device="cpu"))


def test_runpod_handler_prompt_and_messages():
    worker = RunPodTuringWorker(engine=_tiny_engine())

    async def run():
        a = await worker.handler({"input": {"prompt": "Test prompt", "max_tokens": 8}})
        b = await worker.handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4}})
        return a, b

    a, b = asyncio.run(run())
    assert a["object"] == "chat.completion" and isinstance(a["choices"][0]["message"]["content"], str)
    assert a["usage"]["completion_tokens"] == 8 and b["usage"]["completion_tokens"] == 4


def _request(method, path, body=b""):
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}
    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(b"content-type", b"application/json")]}
    return Request(scope, receive)


def test_ray_forward_serves_the_http_api():
    turing = InProcessTuring(_tiny_engine())

    async def run():
        health = await forward(turing, _request("GET", "/health"))
        body = json.dumps({"model": "test-tiny", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4})
        chat = await forward(turing, _request("POST", "/v1/chat/completions", body.encode()))
        return health, chat

    health, chat = asyncio.run(run())
    assert health.status_code == 200
    assert chat.status_code == 200 and json.loads(chat.body)["usage"]["completion_tokens"] == 4


def test_modal_recipe_imports():
    assert modal_turing.modal is None or hasattr(modal_turing, "app")


def test_load_engine_real_weights():
    """load_engine on real weights (SmolLM2-135M, when it is in the local HuggingFace cache)."""
    from huggingface_hub import try_to_load_from_cache
    repo = "HuggingFaceTB/SmolLM2-135M"
    if not isinstance(try_to_load_from_cache(repo, "config.json"), str):
        pytest.skip(f"{repo} is not in the local HuggingFace cache")
    engine = load_engine(repo, device="cpu")
    assert engine.tokenizer is not None and engine.model_config.num_layers == 30
    out = asyncio.run(RunPodTuringWorker(engine=engine).handler(
        {"input": {"prompt": "The capital of France is", "max_tokens": 4, "temperature": 0.0}}))
    assert out["choices"][0]["message"]["content"].strip()
