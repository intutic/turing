"""
OpenAI-Compatible FastAPI Asynchronous LLM Serving Server.
"""

import json
import time
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from ..config import ModelConfig, TuringConfig
from .engine import ContinuousBatchEngine
from .anthropic_api import AnthropicMessageRequest, AnthropicAPIHandler

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[int]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False

def create_app(engine: ContinuousBatchEngine) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(title="Turing Engine High-Performance Inference Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "model": engine.model_config.name,
            "device": str(engine.device),
            "running_requests": len(engine.running_batch),
            "waiting_requests": len(engine.waiting_queue),
            "sparsity_ratio": f"{engine.model_config.sparsity_ratio * 100:.1f}%",
        }

    @app.get("/metrics")
    async def metrics(request: Request):
        telem = engine.get_telemetry()
        accept = request.headers.get("accept", "")
        if "text/plain" in accept:
            # Prometheus exposition format
            lines = [
                f"# HELP turing_serving_throughput_tok_per_sec Instantaneous generated tokens per second",
                f"# TYPE turing_serving_throughput_tok_per_sec gauge",
                f"turing_serving_throughput_tok_per_sec {telem['serving_throughput_tok_per_sec']}",
                f"# HELP turing_total_tokens_generated Total count of tokens generated",
                f"# TYPE turing_total_tokens_generated counter",
                f"turing_total_tokens_generated {telem['total_tokens_generated']}",
                f"# HELP turing_running_requests Number of requests currently executing in batch",
                f"# TYPE turing_running_requests gauge",
                f"turing_running_requests {telem['running_requests']}",
                f"# HELP turing_waiting_queue_depth Number of requests queued in waiting backlog",
                f"# TYPE turing_waiting_queue_depth gauge",
                f"turing_waiting_queue_depth {telem['waiting_queue_depth']}",
                f"# HELP turing_ttft_avg_ms Average Time-To-First-Token in milliseconds",
                f"# TYPE turing_ttft_avg_ms gauge",
                f"turing_ttft_avg_ms {telem['latency']['avg_ttft_ms']}",
                f"# HELP turing_ttft_p99_ms 99th percentile Time-To-First-Token in milliseconds",
                f"# TYPE turing_ttft_p99_ms gauge",
                f"turing_ttft_p99_ms {telem['latency']['p99_ttft_ms']}",
                f"# HELP turing_itl_avg_ms Average Inter-Token Latency in milliseconds",
                f"# TYPE turing_itl_avg_ms gauge",
                f"turing_itl_avg_ms {telem['latency']['avg_itl_ms']}"
            ]
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse("\n".join(lines) + "\n")

        return {
            "model": engine.model_config.name,
            "device": str(engine.device),
            "total_tiles": engine.model_config.total_tiles,
            "active_tiles": engine.model_config.active_tiles,
            "subspace_dim": engine.model_config.active_subspace_dim,
            "telemetry": telem
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": engine.model_config.name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "turing",
                    "permission": [],
                    "root": engine.model_config.name,
                    "parent": None,
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        # Flatten messages to pseudo-tokens (or simple ASCII encoding if tokenizer not attached)
        full_text = " ".join([m.content for m in req.messages])
        prompt_tokens = [ord(c) % engine.model_config.vocab_size for c in full_text]
        if not prompt_tokens:
            prompt_tokens = [1]

        max_new_tokens = req.max_tokens or 32
        temp = req.temperature if req.temperature is not None else 0.7
        top_k = req.top_k or 50

        req_id = f"chatcmpl-{int(time.time()*1000)}"

        if req.stream:
            async def event_generator():
                token_stream = engine.stream_generate(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k
                )
                async for token_id in token_stream:
                    char_repr = chr(token_id % 128) if (32 <= (token_id % 128) <= 126) else f"<{token_id}>"
                    chunk_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": char_repr},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                # Final termination event
                done_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }
                    ]
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        # Non-streaming
        generated_tokens = []
        token_stream = engine.stream_generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k
        )
        async for token_id in token_stream:
            generated_tokens.append(token_id)

        out_text = "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in generated_tokens])

        return {
            "id": req_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": out_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens),
                "total_tokens": len(prompt_tokens) + len(generated_tokens)
            }
        }

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessageRequest):
        """
        Official Anthropic Messages API Endpoint (/v1/messages).
        """
        full_text = AnthropicAPIHandler.extract_prompt_from_request(req)
        prompt_tokens = [ord(c) % engine.model_config.vocab_size for c in full_text]
        if not prompt_tokens:
            prompt_tokens = [1]

        max_new_tokens = req.max_tokens or 32
        temp = req.temperature if req.temperature is not None else 0.7
        top_k = req.top_k or 50

        if req.stream:
            async def anthropic_token_stream():
                token_stream = engine.stream_generate(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k
                )
                async for tok_id in token_stream:
                    tok_str = chr(tok_id % 128) if (32 <= (tok_id % 128) <= 126) else f"<{tok_id}>"
                    yield tok_id, tok_str

            return StreamingResponse(
                AnthropicAPIHandler.stream_generator(
                    req=req,
                    token_stream=anthropic_token_stream(),
                    input_tokens_count=len(prompt_tokens)
                ),
                media_type="text/event-stream"
            )

        # Non-streaming
        generated_tokens = []
        token_stream = engine.stream_generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k
        )
        async for token_id in token_stream:
            generated_tokens.append(token_id)

        out_text = "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in generated_tokens])
        response = AnthropicAPIHandler.format_non_streaming_response(
            req=req,
            generated_text=out_text,
            input_tokens_count=len(prompt_tokens),
            output_tokens_count=len(generated_tokens)
        )
        return response.model_dump()

    return app
