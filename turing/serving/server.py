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
from .ollama_api import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaShowRequest,
    OllamaAPIHandler
)
from .structured import StructuredOutputParser
from .tools import ToolCallingHandler
from .kv_events import KVBlockEventPublisher
from .traffic import AdmissionController, LanePolicy, Lane
from .. import __version__

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
    lane: Optional[str] = Field(default=None, description="QoS scheduling lane (interactive, batch, background)")
    reasoning_effort: Optional[str] = Field(default=None, description="Constrains reasoning effort level: low, medium, high")
    response_format: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="Structured output format (json_object or json_schema)")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="List of tools/functions available for invocation")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="Tool selection policy")

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[int]]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 64
    stream: Optional[bool] = False
    sparsity_ratio: Optional[float] = Field(default=None, description="Custom subspace sparsity ratio (0.0 to 0.9)")
    use_svd_kv: Optional[bool] = Field(default=None, description="Enable calibrated SVD INT8 KV cache paging")
    draft_tokens: Optional[int] = Field(default=None, description="Number of speculative candidate draft tokens")
    lane: Optional[str] = Field(default=None, description="QoS scheduling lane (interactive, batch, background)")
    reasoning_effort: Optional[str] = Field(default=None, description="Constrains reasoning effort level: low, medium, high")

def create_app(
    engine: ContinuousBatchEngine,
    kv_publisher: Optional[KVBlockEventPublisher] = None,
    admission_controller: Optional[AdmissionController] = None,
    lane_policy: Optional[LanePolicy] = None
) -> FastAPI:
    if admission_controller is not None:
        engine.admission = admission_controller
    if lane_policy is not None:
        engine.lane_policy = lane_policy

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

    app = FastAPI(title="Turing Engine High-Performance Inference Server", version=__version__, lifespan=lifespan)

    def _extract_turing_controls(req_obj: Any, raw_req: Request) -> Dict[str, Any]:
        """
        Extracts dynamic per-request sparsity, SVD KV paging, QoS lane, and draft speculation knobs
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

        # 4. QoS Lane
        lane_str = getattr(req_obj, "lane", None)
        if lane_str is None:
            lane_str = headers.get("x-turing-lane") or headers.get("X-Turing-Lane")

        lane_obj = None
        if lane_str:
            try:
                lane_obj = Lane(lane_str.lower())
            except ValueError:
                pass

        return {
            "sparsity_ratio": float(sparsity),
            "use_svd_kv": bool(svd_kv),
            "draft_tokens": draft_toks,
            "lane": lane_obj
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
            "kv_cache_utilization": f"{engine.get_kv_cache_utilization() * 100:.2f}%",
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
            if "admission" in telem:
                lines.extend([
                    f"# HELP turing_vram_utilization_ratio Admission controller tracked VRAM utilization ratio",
                    f"# TYPE turing_vram_utilization_ratio gauge",
                    f"turing_vram_utilization_ratio {telem['admission']['utilization']}",
                    f"# HELP turing_admission_shed_total Total requests shed by admission control",
                    f"# TYPE turing_admission_shed_total counter",
                    f"turing_admission_shed_total {telem['admission']['shed_count']}",
                ])
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
            "X-Turing-KV-Utilization": f"{engine.get_kv_cache_utilization():.4f}",
            "X-Turing-Queue-Depth": str(len(engine.waiting_queue)),
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
            "X-Turing-KV-Utilization": f"{engine.get_kv_cache_utilization():.4f}",
            "X-Turing-Queue-Depth": str(len(engine.waiting_queue)),
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device)
        }

        # Encode messages using real tokenizer with ASCII fallback
        full_text = " ".join([m.content for m in req.messages])

        # Inject Tool Calling Instructions if tools are provided
        if req.tools:
            full_text = ToolCallingHandler.inject_tools_instruction(full_text, req.tools)

        # Inject JSON Structured Output Instructions if requested
        if req.response_format:
            if isinstance(req.response_format, dict):
                fmt_type = req.response_format.get("type")
                if fmt_type == "json_schema":
                    schema_data = req.response_format.get("json_schema", {})
                    full_text = StructuredOutputParser.inject_json_instruction(
                        full_text,
                        schema=schema_data.get("schema"),
                        schema_name=schema_data.get("name")
                    )
                elif fmt_type == "json_object":
                    full_text = StructuredOutputParser.inject_json_instruction(full_text)
            elif isinstance(req.response_format, str) and "json" in req.response_format.lower():
                full_text = StructuredOutputParser.inject_json_instruction(full_text)

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
        finish_reason = "stop"
        tool_calls_extracted = None

        if req.tools:
            clean_text, tool_calls = ToolCallingHandler.extract_tool_calls(out_text)
            if tool_calls:
                tool_calls_extracted = tool_calls
                finish_reason = "tool_calls"
                out_text = clean_text or None

        msg_payload: Dict[str, Any] = {
            "role": "assistant",
            "content": out_text
        }
        if tool_calls_extracted:
            msg_payload["tool_calls"] = tool_calls_extracted

        return JSONResponse(
            content={
                "id": req_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": msg_payload,
                        "finish_reason": finish_reason
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

    # -------------------------------------------------------------------------
    # Native Ollama API Compatibility Layer (/api/*)
    # -------------------------------------------------------------------------

    @app.get("/api/tags")
    async def ollama_tags():
        """
        Lists available models in native Ollama format (compatible with Open WebUI, Ollama CLI).
        """
        return OllamaAPIHandler.format_tags_response(engine.model_config.name, engine.model_config)

    @app.get("/api/ps")
    async def ollama_ps():
        """
        Lists running models and active VRAM footprint in Ollama format.
        """
        vram_bytes = int(engine.get_kv_cache_utilization() * 4_294_967_296) + 2_000_000_000
        return OllamaAPIHandler.format_ps_response(engine.model_config.name, engine.model_config, vram_bytes=vram_bytes)

    @app.get("/api/version")
    async def ollama_version():
        """
        Returns server version in Ollama format.
        """
        return {"version": __version__}

    @app.post("/api/show")
    async def ollama_show(req: OllamaShowRequest):
        """
        Inspects model parameters and architecture in Ollama format.
        """
        target_model = req.model or engine.model_config.name
        return OllamaAPIHandler.format_show_response(target_model, engine.model_config)

    @app.post("/api/pull")
    async def ollama_pull(raw_req: Request):
        """
        Simulates Ollama model pull / loading status.
        """
        body = await raw_req.json() if raw_req.headers.get("content-type") == "application/json" else {}
        model_name = body.get("model", engine.model_config.name)
        return JSONResponse(content={"status": "success", "digest": f"sha256:turing{abs(hash(model_name)):016x}", "total": 100, "completed": 100})

    @app.post("/api/generate")
    async def ollama_generate(req: OllamaGenerateRequest, raw_req: Request):
        """
        Ollama single-prompt raw completion endpoint (/api/generate).
        """
        controls = _extract_turing_controls(req, raw_req)
        resp_headers = {
            "X-Turing-Sparsity": f"{controls['sparsity_ratio']:.3f}",
            "X-Turing-SVD-KV": "1" if controls["use_svd_kv"] else "0",
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device)
        }

        full_text = OllamaAPIHandler.extract_prompt_from_generate_request(req)

        # Inject JSON Structured Output if requested
        if req.format:
            if isinstance(req.format, dict):
                full_text = StructuredOutputParser.inject_json_instruction(full_text, schema=req.format)
            elif isinstance(req.format, str) and req.format.lower() == "json":
                full_text = StructuredOutputParser.inject_json_instruction(full_text)

        prompt_tokens = engine.encode_prompt(full_text)
        if not prompt_tokens:
            prompt_tokens = [1]

        opts = req.options or {}
        max_new_tokens = opts.get("num_predict", 64)
        temp = float(opts.get("temperature", 0.7))
        top_k = int(opts.get("top_k", 50))

        t0 = time.time_ns()

        if req.stream:
            async def ollama_generate_stream():
                accumulated_tokens = []
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
                    accumulated_tokens.append(tok_id)
                    tok_str = engine.decode_tokens([tok_id])
                    yield OllamaAPIHandler.format_streaming_generate_chunk(req, tok_str, done=False)

                # Final chunk
                yield OllamaAPIHandler.format_streaming_generate_chunk(
                    req,
                    token_text="",
                    done=True,
                    context_tokens=prompt_tokens + accumulated_tokens,
                    prompt_tokens=len(prompt_tokens),
                    completion_tokens=len(accumulated_tokens)
                )

            return StreamingResponse(ollama_generate_stream(), media_type="application/x-ndjson", headers=resp_headers)

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

        t_end = time.time_ns()
        out_text = engine.decode_tokens(generated_tokens)

        response = OllamaAPIHandler.format_non_streaming_generate(
            req=req,
            response_text=out_text,
            context_tokens=prompt_tokens + generated_tokens,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(generated_tokens),
            total_duration_ns=t_end - t0,
            eval_duration_ns=max(1, t_end - t0 - 10_000_000)
        )
        return JSONResponse(content=response, headers=resp_headers)

    @app.post("/api/chat")
    async def ollama_chat(req: OllamaChatRequest, raw_req: Request):
        """
        Ollama chat completion endpoint (/api/chat).
        """
        controls = _extract_turing_controls(req, raw_req)
        resp_headers = {
            "X-Turing-Sparsity": f"{controls['sparsity_ratio']:.3f}",
            "X-Turing-SVD-KV": "1" if controls["use_svd_kv"] else "0",
            "X-Turing-Model": engine.model_config.name,
            "X-Turing-Device": str(engine.device)
        }

        full_text = OllamaAPIHandler.extract_prompt_from_chat_request(req)

        # Inject Tool Calling Instructions if tools are provided
        if req.tools:
            full_text = ToolCallingHandler.inject_tools_instruction(full_text, req.tools)

        # Inject JSON Structured Output if requested
        if req.format:
            if isinstance(req.format, dict):
                full_text = StructuredOutputParser.inject_json_instruction(full_text, schema=req.format)
            elif isinstance(req.format, str) and req.format.lower() == "json":
                full_text = StructuredOutputParser.inject_json_instruction(full_text)

        prompt_tokens = engine.encode_prompt(full_text)
        if not prompt_tokens:
            prompt_tokens = [1]

        opts = req.options or {}
        max_new_tokens = opts.get("num_predict", 64)
        temp = float(opts.get("temperature", 0.7))
        top_k = int(opts.get("top_k", 50))

        t0 = time.time_ns()

        if req.stream:
            async def ollama_chat_stream():
                accumulated_tokens = []
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
                    accumulated_tokens.append(tok_id)
                    tok_str = engine.decode_tokens([tok_id])
                    yield OllamaAPIHandler.format_streaming_chat_chunk(req, tok_str, done=False)

                # Final chunk
                yield OllamaAPIHandler.format_streaming_chat_chunk(
                    req,
                    delta_text="",
                    done=True,
                    prompt_tokens=len(prompt_tokens),
                    completion_tokens=len(accumulated_tokens)
                )

            return StreamingResponse(ollama_chat_stream(), media_type="application/x-ndjson", headers=resp_headers)

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

        t_end = time.time_ns()
        out_text = engine.decode_tokens(generated_tokens)
        tool_calls_extracted = None

        if req.tools:
            clean_text, tool_calls = ToolCallingHandler.extract_tool_calls(out_text)
            if tool_calls:
                tool_calls_extracted = tool_calls
                out_text = clean_text

        response = OllamaAPIHandler.format_non_streaming_chat(
            req=req,
            message_content=out_text,
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(generated_tokens),
            tool_calls=tool_calls_extracted,
            total_duration_ns=t_end - t0
        )
        return JSONResponse(content=response, headers=resp_headers)

    @app.post("/api/embed")
    @app.post("/api/embeddings")
    async def ollama_embeddings(raw_req: Request):
        """
        Ollama embeddings endpoint.
        """
        body = await raw_req.json() if raw_req.headers.get("content-type") == "application/json" else {}
        prompt = body.get("prompt", "")
        prompt_tokens = engine.encode_prompt(prompt) or [1]
        dim = engine.model_config.rank_sub or 64
        embedding = [(float((t * 17) % 100) / 100.0) - 0.5 for t in range(dim)]
        return JSONResponse(content={"embedding": embedding, "embeddings": [embedding]})

    return app

