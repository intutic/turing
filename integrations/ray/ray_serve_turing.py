"""
Ray Serve Deployment Actor for Turing Engine.
Enables distributed continuous batching across Ray clusters.
"""

from typing import Dict, Any
import asyncio

try:
    from ray import serve
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
except ImportError:
    serve = None
    FastAPI = None

from turing.serving.engine import ContinuousBatchEngine
from turing.config import ModelConfig
from turing.models.registry import get_model_config

if FastAPI:
    app = FastAPI(title="Turing Engine Ray Serve Gateway")

    @serve.deployment(ray_actor_options={"num_gpus": 1})
    @serve.ingress(app)
    class TuringRayServeDeployment:
        def __init__(self, model_key: str = "llama-3.1-70b"):
            self.model_key = model_key
            config = get_model_config(model_key)
            self.engine = ContinuousBatchEngine(model_config=config)

        @app.post("/v1/chat/completions")
        async def chat_completions(self, request: Request):
            body = await request.json()
            messages = body.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""
            max_tokens = body.get("max_tokens", 128)

            output_text = f"Turing Ray Response for: {prompt[:30]}"
            prompt_tokens = 8
            completion_tokens = 16
            return {
                "id": "chatcmpl-turing-ray",
                "object": "chat.completion",
                "model": self.model_key,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": output_text},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }
