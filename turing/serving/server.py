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
from .kv_events import KVBlockEventPublisher

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
    sparsity_ratio: Optional[float] = Field(default=None, description="Custom subspace sparsity ratio (0.0 to 0.9)")
    use_svd_kv: Optional[bool] = Field(default=None, description="Enable calibrated SVD INT8 KV cache paging")
    draft_tokens: Optional[int] = Field(default=None, description="Number of speculative candidate draft tokens")

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[int]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False
    sparsity_ratio: Optional[float] = Field(default=None, description="Custom subspace sparsity ratio (0.0 to 0.9)")
    use_svd_kv: Optional[bool] = Field(default=None, description="Enable calibrated SVD INT8 KV cache paging")
    draft_tokens: Optional[int] = Field(default=None, description="Number of speculative candidate draft tokens")

def create_app(engine: ContinuousBatchEngine, kv_publisher: Optional[KVBlockEventPublisher] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        if kv_publisher is not None:
            kv_publisher.start()
        try:
            yield
        finally:
            if kv_publisher is not None:
                kv_publisher.stop()
            await engine.stop()

    app = FastAPI(title="Turing Engine High-Performance Inference Server", version="0.3.0", lifespan=lifespan)




    def _extract_turing_controls(req_obj: Any, raw_req: Request) -> Dict[str, Any]:
        """
        Extracts dynamic per-request sparsity, SVD KV paging, and draft speculation knobs
        from HTTP headers (X-Turing-*) or JSON request payload with fallback to model defaults.
        """
        headers = raw_req.headers

        # 1. Sparsity Ratio
        sparsity = getattr(req_obj, "sparsity_ratio", None)
        if sparsity is None:
            raw_hdr = headers.get("x-turing-sparsity") or headers.get("X-Turing-Sparsity")
            if raw_hdr:
                try:
                    sparsity = float(raw_hdr)
                except ValueError:
                    pass
        if sparsity is None:
            sparsity = engine.model_config.sparsity_ratio

        # 2. SVD KV Paging
        svd_kv = getattr(req_obj, "use_svd_kv", None)
        if svd_kv is None:
            raw_hdr = headers.get("x-turing-svd-kv") or headers.get("X-Turing-SVD-KV")
            if raw_hdr:
                svd_kv = raw_hdr.strip().lower() in ("1", "true", "yes")
        if svd_kv is None:
            svd_kv = True

        # 3. Draft Speculation Tokens
        draft_toks = getattr(req_obj, "draft_tokens", None)
        if draft_toks is None:
            raw_hdr = headers.get("x-turing-draft-tokens") or headers.get("X-Turing-Draft-Tokens")
            if raw_hdr:
                try:
                    draft_toks = int(raw_hdr)
                except ValueError:
                    pass

        return {
            "sparsity_ratio": float(sparsity),
            "use_svd_kv": bool(svd_kv),
            "draft_tokens": draft_toks
        }

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
            llmd = engine.get_llmd_metrics()
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
                f"turing_itl_avg_ms {telem['latency']['avg_itl_ms']}",
                f"# HELP turing_num_requests_waiting Number of requests queued in waiting backlog (llm-d TotalQueuedRequests)",
                f"# TYPE turing_num_requests_waiting gauge",
                f"turing_num_requests_waiting {llmd['num_requests_waiting']}",
                f"# HELP turing_num_requests_running Number of requests currently executing in batch (llm-d TotalRunningRequests)",
                f"# TYPE turing_num_requests_running gauge",
                f"turing_num_requests_running {llmd['num_requests_running']}",
                f"# HELP turing_kv_cache_usage_perc Fraction of KV cache memory pool in use (llm-d KVCacheUtilization)",
                f"# TYPE turing_kv_cache_usage_perc gauge",
                f"turing_kv_cache_usage_perc {llmd['kv_cache_usage_perc']}",
                f"# HELP turing_cache_config_info KV cache configuration info (llm-d BlockSize and NumGPUBlocks)",
                f"# TYPE turing_cache_config_info gauge",
                f'turing_cache_config_info{{block_size="{llmd["block_size"]}",num_gpu_blocks="{llmd["num_gpu_blocks"]}"}} 1.0',
            ]
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse("\n".join(lines) + "\n")

        return {
            "model": engine.model_config.name,
            "device": str(engine.device),
            "total_tiles": engine.model_config.total_tiles,
            "active_tiles": engine.model_config.active_tiles,
            "subspace_dim": engine.model_config.active_subspace_dim,
            "telemetry": telem,
            "llmd_metrics": engine.get_llmd_metrics(),
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

    @app.post("/v1/completions/render")
    async def render_completions(req: Request):
        """Tokenizes prompt for llm-d EPP token-producer prefix indexing."""
        body = await req.json()
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            tokens = [int(t) for t in prompt]
        else:
            tokens = engine.encode_prompt(str(prompt))
        if not tokens:
            tokens = [1]
        return JSONResponse(content={"tokens": tokens, "count": len(tokens)})

    @app.post("/v1/chat/completions/render")
    async def render_chat_completions(req: Request):
        """Tokenizes chat messages for llm-d EPP token-producer prefix indexing."""
        body = await req.json()
        messages = body.get("messages", [])
        full_text = " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
        tokens = engine.encode_prompt(full_text)
        if not tokens:
            tokens = [1]
        return JSONResponse(content={"tokens": tokens, "count": len(tokens)})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_req: Request):
        controls = _extract_turing_controls(req, raw_req)
        resp_headers = {
            "X-Turing-Sparsity": f"{controls['sparsity_ratio']:.3f}",
            "X-Turing-SVD-KV": "1" if controls["use_svd_kv"] else "0",
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device),
        }

        if isinstance(req.prompt, list):
            prompt_tokens = [int(t) for t in req.prompt]
        else:
            prompt_tokens = engine.encode_prompt(req.prompt or "")

        if not prompt_tokens:
            prompt_tokens = [1]

        max_new_tokens = req.max_tokens or 32
        temp = req.temperature if req.temperature is not None else 0.7
        top_k = 50

        req_id = f"cmpl-{int(time.time()*1000)}"

        if req.stream:
            async def event_generator():
                token_stream = engine.stream_generate(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                    top_k=top_k,
                    sparsity_ratio=controls["sparsity_ratio"],
                    use_svd_kv=controls["use_svd_kv"],
                    draft_tokens=controls["draft_tokens"],
                )
                async for token_id in token_stream:
                    char_repr = engine.decode_tokens([token_id])
                    chunk_data = {
                        "id": req_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "text": char_repr,
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                done_chunk = {
                    "id": req_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": req.model,
                    "choices": [
                        {
                            "text": "",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream", headers=resp_headers)

        # Non-streaming
        generated_tokens = []
        token_stream = engine.stream_generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            sparsity_ratio=controls["sparsity_ratio"],
            use_svd_kv=controls["use_svd_kv"],
            draft_tokens=controls["draft_tokens"],
        )
        async for token_id in token_stream:
            generated_tokens.append(token_id)

        out_text = engine.decode_tokens(generated_tokens)

        return JSONResponse(
            content={
                "id": req_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "text": out_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(generated_tokens),
                    "total_tokens": len(prompt_tokens) + len(generated_tokens),
                },
            },
            headers=resp_headers,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw_req: Request):
        controls = _extract_turing_controls(req, raw_req)
        resp_headers = {
            "X-Turing-Sparsity": f"{controls['sparsity_ratio']:.3f}",
            "X-Turing-SVD-KV": "1" if controls["use_svd_kv"] else "0",
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device)
        }

        # Encode messages using real tokenizer with ASCII fallback
        full_text = " ".join([m.content for m in req.messages])
        prompt_tokens = engine.encode_prompt(full_text)
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
                    top_k=top_k,
                    sparsity_ratio=controls["sparsity_ratio"],
                    use_svd_kv=controls["use_svd_kv"],
                    draft_tokens=controls["draft_tokens"]
                )
                async for token_id in token_stream:
                    char_repr = engine.decode_tokens([token_id])
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

            return StreamingResponse(event_generator(), media_type="text/event-stream", headers=resp_headers)

        # Non-streaming
        generated_tokens = []
        token_stream = engine.stream_generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            sparsity_ratio=controls["sparsity_ratio"],
            use_svd_kv=controls["use_svd_kv"],
            draft_tokens=controls["draft_tokens"]
        )
        async for token_id in token_stream:
            generated_tokens.append(token_id)

        out_text = engine.decode_tokens(generated_tokens)

        return JSONResponse(
            content={
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
            },
            headers=resp_headers
        )

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessageRequest, raw_req: Request):
        """
        Official Anthropic Messages API Endpoint (/v1/messages).
        """
        controls = _extract_turing_controls(req, raw_req)
        resp_headers = {
            "X-Turing-Sparsity": f"{controls['sparsity_ratio']:.3f}",
            "X-Turing-SVD-KV": "1" if controls["use_svd_kv"] else "0",
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device)
        }

        full_text = AnthropicAPIHandler.extract_prompt_from_request(req)
        prompt_tokens = engine.encode_prompt(full_text)
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
                    top_k=top_k,
                    sparsity_ratio=controls["sparsity_ratio"],
                    use_svd_kv=controls["use_svd_kv"],
                    draft_tokens=controls["draft_tokens"]
                )
                async for tok_id in token_stream:
                    tok_str = engine.decode_tokens([tok_id])
                    yield tok_id, tok_str

            return StreamingResponse(
                AnthropicAPIHandler.stream_generator(
                    req=req,
                    token_stream=anthropic_token_stream(),
                    input_tokens_count=len(prompt_tokens)
                ),
                media_type="text/event-stream",
                headers=resp_headers
            )

        # Non-streaming
        generated_tokens = []
        token_stream = engine.stream_generate(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=top_k,
            sparsity_ratio=controls["sparsity_ratio"],
            use_svd_kv=controls["use_svd_kv"],
            draft_tokens=controls["draft_tokens"]
        )
        async for token_id in token_stream:
            generated_tokens.append(token_id)

        out_text = engine.decode_tokens(generated_tokens)
        response = AnthropicAPIHandler.format_non_streaming_response(
            req=req,
            generated_text=out_text,
            input_tokens_count=len(prompt_tokens),
            output_tokens_count=len(generated_tokens)
        )
        return JSONResponse(content=response.model_dump(), headers=resp_headers)

    return app
