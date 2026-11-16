"""
OpenAI-Compatible FastAPI Asynchronous LLM Serving Server.
"""

import asyncio
import json
import math
import time

import torch
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ..config import ModelConfig, TuringConfig
from .engine import ContinuousBatchEngine, RequestRejected
from .anthropic_api import AnthropicMessageRequest, AnthropicAPIHandler
from .ollama_api import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaShowRequest,
    OllamaAPIHandler
)
from .structured import StructuredOutputParser
from .tools import ToolCallingHandler
from .kv_events import KVBlockEventPublisher, attach_kv_events
from .traffic import AdmissionController, LanePolicy, Lane, AdaptiveComputeBatcher
from .embeddings import HFEmbedder, HFReranker, llm_embed, llm_yes_scores, shape_embeddings
from .extract import ExtractionError, extract_spans
from .pull import ndjson, pull_plan, pull_progress, repo_and_revision
from .. import __version__

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    model: Optional[str] = None
    top_n: Optional[int] = None
    return_documents: bool = True

class ExtractRequest(BaseModel):
    text: str
    labels: List[str]
    model: Optional[str] = None
    score_threshold: Optional[float] = None
    max_tokens: int = 512

class OllamaEmbedRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    dimensions: Optional[int] = None

class OllamaLegacyEmbedRequest(BaseModel):
    model: Optional[str] = None
    prompt: str

class OllamaPullRequest(BaseModel):
    model: Optional[str] = None
    name: Optional[str] = None  # older Ollama clients
    stream: bool = True
    insecure: Optional[bool] = None

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
    lane_policy: Optional[LanePolicy] = None,
    embedding_model: Optional[str] = None,
    rerank_model: Optional[str] = None,
) -> FastAPI:
    """embedding_model / rerank_model: HuggingFace ids or local paths of a BERT-style embedding model and a
    cross-encoder for /v1/embeddings, /api/embed and /v1/rerank; without them the served LLM does both. A
    kv_publisher receives the engine's prefix-cache events (llm-d) and is started and stopped with the app."""
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

    @app.exception_handler(RequestRejected)
    async def request_rejected(request: Request, exc: RequestRejected):
        headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after)))} if exc.retry_after else None
        path = request.url.path
        if path.startswith("/v1/messages"):  # Anthropic error shape
            kind = {400: "invalid_request_error", 429: "rate_limit_error"}.get(exc.status_code, "overloaded_error")
            body = {"type": "error", "error": {"type": kind, "message": str(exc)}}
        elif path.startswith("/api/"):  # Ollama error shape
            body = {"error": str(exc)}
        else:  # OpenAI error shape
            kind = {400: "invalid_request_error", 429: "rate_limit_exceeded"}.get(exc.status_code, "server_overloaded")
            body = {"error": {"message": str(exc), "type": kind, "code": exc.status_code}}
        return JSONResponse(status_code=exc.status_code, content=body, headers=headers)
    embedder = HFEmbedder(embedding_model, engine.device) if embedding_model else None
    reranker = HFReranker(rerank_model, engine.device) if rerank_model else None
    if kv_publisher is not None:
        attach_kv_events(engine, kv_publisher)

    def _embed(texts: List[str]):
        """(L2-normalised vectors [N, D], tokens read, model name): the embedding model's, else the served LLM's
        mean-pooled final hidden states."""
        if not texts or any(not t for t in texts):
            raise HTTPException(status_code=400, detail="input must be a non-empty string or list of non-empty strings")
        if embedder is not None:
            vecs, n = embedder(texts)
            return vecs, n, embedder.name
        toks = [engine.encode_prompt(t) for t in texts]
        limit = engine.model_config.max_position_embeddings
        if max(map(len, toks)) > limit:
            raise HTTPException(status_code=400, detail=f"an input is longer than the model's {limit} tokens")
        return llm_embed(engine.model, toks), sum(map(len, toks)), engine.model_config.name

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

    def _prom(v):
        return "NaN" if v is None else v  # Prometheus' value for "no samples yet"

    @app.get("/metrics")
    async def metrics(request: Request):
        """Prometheus text exposition (format 0.0.4); ?format=json returns the engine telemetry as JSON."""
        telem = engine.get_telemetry()
        llmd = engine.get_llmd_metrics()
        if request.query_params.get("format") == "json":
            return {
                "model": engine.model_config.name,
                "device": str(engine.device),
                "total_tiles": engine.model_config.total_tiles,
                "active_tiles": engine.model_config.active_tiles,
                "subspace_dim": engine.model_config.active_subspace_dim,
                "telemetry": telem,
                "llmd_metrics": llmd,
            }

        lines = []

        def metric(name, kind, help_text, value, labels=""):
            lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", f"{name}{labels} {_prom(value)}"])

        metric("turing_serving_throughput_tok_per_sec", "gauge", "Generated tokens per second since start",
               telem["serving_throughput_tok_per_sec"])
        metric("turing_total_tokens_generated", "counter", "Total count of tokens generated",
               telem["total_tokens_generated"])
        metric("turing_running_requests", "gauge", "Number of requests currently executing in batch",
               telem["running_requests"])
        metric("turing_waiting_queue_depth", "gauge", "Number of requests queued in waiting backlog",
               telem["waiting_queue_depth"])
        metric("turing_ttft_avg_ms", "gauge", "Average Time-To-First-Token in milliseconds",
               telem["latency"]["avg_ttft_ms"])
        metric("turing_ttft_p99_ms", "gauge", "99th percentile Time-To-First-Token in milliseconds",
               telem["latency"]["p99_ttft_ms"])
        metric("turing_itl_avg_ms", "gauge", "Average Inter-Token Latency in milliseconds",
               telem["latency"]["avg_itl_ms"])
        metric("turing_num_requests_waiting", "gauge",
               "Number of requests queued in waiting backlog (llm-d TotalQueuedRequests)", llmd["num_requests_waiting"])
        metric("turing_num_requests_running", "gauge",
               "Number of requests currently executing in batch (llm-d TotalRunningRequests)",
               llmd["num_requests_running"])
        metric("turing_kv_cache_usage_perc", "gauge",
               "Fraction of KV cache memory pool in use (llm-d KVCacheUtilization)", llmd["kv_cache_usage_perc"])
        metric("turing_cache_config_info", "gauge", "KV cache configuration info (llm-d BlockSize and NumGPUBlocks)",
               1.0, f'{{block_size="{llmd["block_size"]}",num_gpu_blocks="{llmd["num_gpu_blocks"]}"}}')
        pc = telem.get("prefix_cache")
        if pc is not None:
            metric("turing_prefix_cache_hit_rate", "gauge",
                   "Prompt tokens served from the prefix cache over prompt tokens looked up", pc["hit_rate"])
            metric("turing_prefix_cache_hit_tokens_total", "counter", "Prompt tokens served from the prefix cache",
                   pc["hit_tokens"])
            metric("turing_prefix_cache_queried_tokens_total", "counter", "Prompt tokens looked up in the prefix cache",
                   pc["queried_tokens"])
            metric("turing_prefix_cache_pages", "gauge", "KV pages held by the prefix cache", pc["cached_pages"])
        spec = telem["speculation"]
        metric("turing_spec_drafted_tokens_total", "counter", "Speculative draft tokens verified",
               spec["drafted_tokens"])
        metric("turing_spec_accepted_tokens_total", "counter", "Speculative draft tokens accepted",
               spec["accepted_tokens"])
        metric("turing_spec_acceptance_rate", "gauge", "Accepted over drafted speculative tokens",
               spec["acceptance_rate"])
        if "admission" in telem:
            metric("turing_vram_utilization_ratio", "gauge", "Admission controller tracked VRAM utilization ratio",
                   telem["admission"]["utilization"])
            metric("turing_admission_shed_total", "counter", "Total requests shed by admission control",
                   telem["admission"]["shed_count"])
        return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

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

    @app.post("/v1/embeddings")
    async def create_embeddings(req: EmbeddingRequest):
        """OpenAI-compatible embeddings: the embedding model's (--embedding-model), else the served LLM's
        mean-pooled final hidden states; L2-normalised, `dimensions` truncates (Matryoshka) and renormalises."""
        inputs = [req.input] if isinstance(req.input, str) else req.input
        vecs, n_tokens, name = _embed(inputs)
        try:
            vectors = shape_embeddings(vecs, req.dimensions, req.encoding_format or "float")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {
            "object": "list",
            "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
            "model": name,
            "usage": {"prompt_tokens": n_tokens, "total_tokens": n_tokens},
        }

    @app.post("/v1/rerank")
    async def create_rerank(req: RerankRequest):
        """Cohere / Jina-compatible reranking: a cross-encoder's sigmoid score (--rerank-model), else the served
        LLM's probability of answering "yes" to whether the document is relevant. Results sorted best first."""
        if not req.documents:
            raise HTTPException(status_code=400, detail="documents must not be empty")
        if reranker is not None:
            scores, n_tokens = reranker(req.query, req.documents)
            name, scorer = reranker.name, "cross-encoder: sigmoid of the relevance logit"
        else:
            scores, n_tokens = llm_yes_scores(engine, req.query, req.documents)
            name = engine.model_config.name
            scorer = ("llm-yes-probability: P(yes | yes or no) of the served model's next token after asking "
                      "whether the document is relevant to the query")
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        if req.top_n is not None and req.top_n > 0:
            order = order[:req.top_n]
        results = [{"index": i, "relevance_score": scores[i],
                    **({"document": {"text": req.documents[i]}} if req.return_documents else {})} for i in order]
        return {
            "id": f"rerank-{int(time.time() * 1000)}",
            "model": name,
            "results": results,
            "meta": {"scorer": scorer, "total_documents": len(req.documents)},
            "usage": {"total_tokens": n_tokens},
        }

    @app.post("/v1/extract")
    async def create_extract(req: ExtractRequest):
        """Spans of the requested labels, extracted by the served LLM (greedy, JSON output). Every span is located
        in the text (start/end index it); spans the text does not contain are listed in `unmatched`, not returned.
        Output that is not JSON is a 422 error carrying the model's raw output."""
        if not req.text or not req.labels:
            raise HTTPException(status_code=400, detail="text and labels must not be empty")
        try:
            r = await extract_spans(engine, req.text, req.labels, max_tokens=req.max_tokens)
        except ExtractionError as e:
            return JSONResponse(status_code=422, content={"error": {
                "message": f"{e}; the served model could not extract spans", "type": "invalid_model_output",
                "raw_output": e.raw_output}})
        spans = [s for s in r["spans"] if req.score_threshold is None or (s["score"] or 0.0) >= req.score_threshold]
        return {
            "model": engine.model_config.name,
            "data": spans,
            "count": len(spans),
            "unmatched": r["unmatched"],
            "raw_output": r["raw_output"],
            "usage": {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                      "total_tokens": r["prompt_tokens"] + r["completion_tokens"]},
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
            # Admit before the response starts, so a rejection is a 429/503 rather than a broken 200 stream
            admitted = await engine.add_request(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                draft_tokens=controls["draft_tokens"],
            )

            async def event_generator():
                token_stream = engine.stream_request(admitted)
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
            # Admit before the response starts, so a rejection is a 429/503 rather than a broken 200 stream
            admitted = await engine.add_request(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                draft_tokens=controls["draft_tokens"]
            )

            async def event_generator():
                token_stream = engine.stream_request(admitted)
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
            # Admit before the response starts, so a rejection is a 429/503 rather than a broken 200 stream
            admitted = await engine.add_request(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                draft_tokens=controls["draft_tokens"]
            )

            async def anthropic_token_stream():
                token_stream = engine.stream_request(admitted)
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
        return OllamaAPIHandler.format_tags_response(engine.model_config.name, engine.model_config, engine.model)

    @app.get("/api/ps")
    async def ollama_ps():
        """
        Lists running models and active VRAM footprint in Ollama format.
        """
        # weights plus the KV pool as allocated (on CUDA, what the allocator really holds)
        if engine.device.type == "cuda":
            vram_bytes = torch.cuda.memory_allocated(engine.device)
        else:
            vram_bytes = sum(t.numel() * t.element_size() for t in list(engine.model.parameters()) + list(engine.model.buffers()))
            st = engine.kv_pool.get_stats()
            vram_bytes += st["max_physical_pages"] * st["bytes_per_page"]
        return OllamaAPIHandler.format_ps_response(engine.model_config.name, engine.model_config, vram_bytes=vram_bytes,
                                                   model=engine.model)

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
        return OllamaAPIHandler.format_show_response(target_model, engine.model_config, engine.model,
                                                     getattr(engine.tokenizer, "chat_template", None))

    @app.post("/api/pull")
    async def ollama_pull(req: OllamaPullRequest):
        """
        Ollama model pull for a HuggingFace repo ("org/name", "hf.co/org/name[:revision]"): downloads its files into
        the local HuggingFace cache, streaming Ollama progress lines (status, digest, total, completed per file).
        """
        from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
        if not (req.model or req.name):
            return JSONResponse(status_code=400, content={"error": "model is required"})
        repo, revision = repo_and_revision(req.model or req.name)
        try:
            files = await asyncio.to_thread(pull_plan, repo, revision)
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            return JSONResponse(status_code=404, content={"error": f"pull model manifest: {e}"})
        except Exception as e:  # no network, hub errors
            return JSONResponse(status_code=500, content={"error": f"pull model manifest: {e}"})
        progress = pull_progress(repo, revision, files)
        if not req.stream:
            last = {}
            async for last in progress:
                pass
            return JSONResponse(status_code=500 if "error" in last else 200, content=last)
        return StreamingResponse(ndjson(progress), media_type="application/x-ndjson")

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
            # Admit before the response starts, so a rejection is a 429/503 rather than a broken 200 stream
            admitted = await engine.add_request(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                draft_tokens=controls["draft_tokens"]
            )

            async def ollama_generate_stream():
                accumulated_tokens = []
                token_stream = engine.stream_request(admitted)
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
            # Admit before the response starts, so a rejection is a 429/503 rather than a broken 200 stream
            admitted = await engine.add_request(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temp,
                top_k=top_k,
                draft_tokens=controls["draft_tokens"]
            )

            async def ollama_chat_stream():
                accumulated_tokens = []
                token_stream = engine.stream_request(admitted)
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
    async def ollama_embed(req: OllamaEmbedRequest):
        """
        Ollama embeddings (/api/embed): `input` a string or a list of strings -> `embeddings`, L2-normalised.
        """
        t0 = time.time_ns()
        inputs = [req.input] if isinstance(req.input, str) else req.input
        vecs, n_tokens, name = _embed(inputs)
        try:
            vectors = shape_embeddings(vecs, req.dimensions, "float")
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        return {"model": req.model or name, "embeddings": vectors, "total_duration": time.time_ns() - t0,
                "load_duration": 0, "prompt_eval_count": n_tokens}

    @app.post("/api/embeddings")
    async def ollama_embeddings(req: OllamaLegacyEmbedRequest):
        """
        Ollama's legacy embeddings endpoint: `prompt` -> `embedding`.
        """
        vecs, _, _ = _embed([req.prompt])
        return {"embedding": shape_embeddings(vecs, None, "float")[0]}

    return app

