"""
OpenAI-, Anthropic- and Ollama-compatible FastAPI inference server. Every generation endpoint goes through the shared
core in generation.py: chat templates, constrained decoding, reasoning, tool calls, stop strings and real usage.
"""

import logging
import asyncio
import hmac
import json
import math
import time
import uuid

import torch
from typing import List, Optional, Dict, Any, Union
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError

from ..core.kv_cache import PAGE
from .engine import ContinuousBatchEngine, RequestRejected
from .anthropic_api import AnthropicMessageRequest, AnthropicAPIHandler
from .ollama_api import (
    OllamaGenerateRequest,
    OllamaChatRequest,
    OllamaShowRequest,
    OllamaAPIHandler,
    ollama_format,
    ollama_options,
    render_go_template,
)
from . import generation
from .generation import (GenerationParams, REMOVED_FIELDS, REMOVED_HEADERS, bad_request, effort_budget,
                         normalize_tool_choice, openai_messages, render_chat, render_text, thinking_kwargs)
from .tools import normalize_tools
from .kv_events import KVBlockEventPublisher, attach_kv_events
from .traffic import AdmissionController, LanePolicy, Lane
from .embeddings import HFEmbedder, HFReranker, embedding_shape_error, llm_embed, llm_yes_scores, shape_embeddings
from .extract import extract_spans
from .pull import ndjson, pull_plan, pull_progress, repo_and_revision
from .. import __version__

logger = logging.getLogger(__name__)

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

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    draft_tokens: Optional[int] = Field(default=None, description="n-gram speculative draft tokens per step")
    lane: Optional[str] = Field(default=None, description="QoS scheduling lane (interactive, batch, background)")
    reasoning_effort: Optional[str] = Field(default=None, description="Thinking budget: none, low, medium, high")
    response_format: Optional[Union[Dict[str, Any], str]] = Field(default=None, description="json_object or json_schema")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tools the model may call")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(default=None, description="auto, none, required or a named tool")

class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[int]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    max_tokens: Optional[int] = 64
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    response_format: Optional[Union[Dict[str, Any], str]] = None
    draft_tokens: Optional[int] = Field(default=None, description="n-gram speculative draft tokens per step")
    lane: Optional[str] = Field(default=None, description="QoS scheduling lane (interactive, batch, background)")


def _error_body(path: str, status: int, message: str) -> Dict[str, Any]:
    """An error in the shape of the API the path belongs to."""
    if path.startswith("/v1/messages"):  # Anthropic
        kind = {400: "invalid_request_error", 401: "authentication_error", 429: "rate_limit_error"}.get(
            status, "overloaded_error")
        return {"type": "error", "error": {"type": kind, "message": message}}
    if path.startswith("/api/"):  # Ollama
        return {"error": message}
    kind = {400: "invalid_request_error", 401: "invalid_request_error", 429: "rate_limit_exceeded"}.get(
        status, "server_overloaded")
    return {"error": {"message": message, "type": kind, "code": "invalid_api_key" if status == 401 else status}}


def _response_schema(fmt: Any) -> Optional[Dict[str, Any]]:
    """OpenAI response_format as the JSON schema to constrain to (None for plain text)."""
    if fmt is None:
        return None
    if isinstance(fmt, str):
        fmt = {"type": fmt}
    kind = fmt.get("type")
    if kind in (None, "text"):
        return None
    if kind == "json_object":
        return {"type": "object"}
    if kind == "json_schema":
        schema = (fmt.get("json_schema") or {}).get("schema")
        if not isinstance(schema, dict):
            raise bad_request("response_format json_schema needs json_schema.schema (an object)")
        return schema
    raise bad_request(f"response_format type must be text, json_object or json_schema, not '{kind}'")


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _usage(gen) -> Dict[str, Any]:
    return {"prompt_tokens": gen.prompt_tokens, "completion_tokens": gen.completion_tokens,
            "total_tokens": gen.prompt_tokens + gen.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": gen.cached_tokens}}


def create_app(
    engine: ContinuousBatchEngine,
    kv_publisher: Optional[KVBlockEventPublisher] = None,
    admission_controller: Optional[AdmissionController] = None,
    lane_policy: Optional[LanePolicy] = None,
    embedding_model: Optional[str] = None,
    rerank_model: Optional[str] = None,
    api_key: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    admission: bool = True,
    lanes: bool = True,
) -> FastAPI:
    """embedding_model / rerank_model: HuggingFace ids or local paths of a BERT-style embedding model and a
    cross-encoder for /v1/embeddings, /api/embed and /v1/rerank; without them the served LLM does both. A
    kv_publisher receives the engine's prefix-cache events (llm-d) and is started and stopped with the app.
    api_key: when set, every endpoint but /health needs `Authorization: Bearer <key>` or `x-api-key: <key>`.
    reasoning_effort: the thinking budget for requests that do not set one (none, low, medium, high).
    admission / lanes: build an AdmissionController sized to the engine's KV pool and the 3-lane LanePolicy when
    the engine has none and none is passed (False leaves them off)."""
    if admission_controller is not None:
        engine.admission = admission_controller
    elif admission and engine.admission is None:
        st = engine.kv_pool.get_stats()
        engine.admission = AdmissionController(st["total_pages"] * st["bytes_per_page"])
    if lane_policy is not None:
        engine.lane_policy = lane_policy
    elif lanes and engine.lane_policy is None:
        engine.lane_policy = LanePolicy()
    if reasoning_effort is not None:
        effort_budget(reasoning_effort)  # refuse a bad default at startup, not on every request

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
        return JSONResponse(status_code=exc.status_code, content=_error_body(request.url.path, exc.status_code,
                                                                             str(exc)), headers=headers)

    if api_key:
        key = api_key.encode()

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            if request.url.path != "/health":
                auth = request.headers.get("authorization", "")
                bearer = auth[7:].strip() if auth[:7].lower() == "bearer " else ""
                given = request.headers.get("x-api-key", "")
                if not (hmac.compare_digest(bearer.encode(), key) or hmac.compare_digest(given.encode(), key)):
                    return JSONResponse(status_code=401, headers={"WWW-Authenticate": "Bearer"},
                                        content=_error_body(request.url.path, 401, "invalid or missing API key"))
            return await call_next(request)

    cfg = engine.model_config
    lane_stats: Dict[str, Dict[str, int]] = {lane.value: {"admitted": 0, "finished": 0} for lane in Lane}
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

    def _controls(body: Dict[str, Any], raw_req: Request) -> Dict[str, Any]:
        """Per-request engine controls from the body or X-Turing-* headers: draft tokens, lane, LoRA adapter."""
        for k, why in REMOVED_FIELDS.items():
            if k in body:
                raise bad_request(f"'{k}' is not a request parameter: {why}")
        for h, why in REMOVED_HEADERS.items():
            if h in raw_req.headers:
                raise bad_request(f"header {h} is not supported: {why}")
        headers = raw_req.headers
        draft = body.get("draft_tokens")
        if draft is None and headers.get("x-turing-draft-tokens") is not None:
            try:
                draft = int(headers["x-turing-draft-tokens"])
            except ValueError:
                raise bad_request("X-Turing-Draft-Tokens must be an integer")
        if draft is not None and (not isinstance(draft, int) or draft < 0):
            raise bad_request("draft_tokens must be a non-negative integer")
        lane_str = body.get("lane") or headers.get("x-turing-lane")
        lane = None
        if lane_str:
            try:
                lane = Lane(str(lane_str).lower())
            except ValueError:
                raise bad_request(f"lane must be one of {', '.join(l.value for l in Lane)}, not '{lane_str}'")
        adapter = headers.get("x-turing-adapter")
        model = body.get("model") or ""
        if adapter is None and isinstance(model, str) and ":" in model:
            suffix = model.rsplit(":", 1)[1]  # "base:adapter" names a LoRA adapter the server holds
            if suffix in getattr(engine.model, "adapter_names", ()):
                adapter = suffix
        return {"draft_tokens": draft, "lane": lane, "adapter": adapter}

    async def _start(prompt_ids: List[int], params: GenerationParams, prompt_text: str = ""):
        gen = await generation.start(engine, prompt_ids, params, prompt_text=prompt_text)
        stats = lane_stats[gen.req.lane.value] if gen.req.lane is not None else None
        if stats is not None:
            stats["admitted"] += 1
            gen.stats = stats
        return gen

    def _headers(gen) -> Dict[str, str]:
        """What the server really did for this request."""
        h = {
            "X-Turing-Model": cfg.name,
            "X-Turing-Device": str(engine.device),
            "X-Turing-KV-Cache-Dtype": cfg.kv_cache_dtype,
            "X-Turing-Cached-Tokens": str(gen.cached_tokens),
            "X-Turing-Draft-Tokens": str(gen.draft_tokens),
            "X-Turing-KV-Utilization": f"{engine.get_kv_cache_utilization():.4f}",
            "X-Turing-Queue-Depth": str(len(engine.waiting_queue)),
        }
        if gen.req.lane is not None:
            h["X-Turing-Lane"] = gen.req.lane.value
        if gen.params.adapter:
            h["X-Turing-Adapter"] = gen.params.adapter
        return h

    def _max_tokens(requested: Optional[int], prompt_ids: List[int], default: int) -> int:
        """requested, the default when unset, or (negative, as Ollama's num_predict -1) as many as fit."""
        if requested is None:
            return default
        if requested < 0:
            pool = engine.kv_pool
            return max(1, min(pool.table_width, pool.page_limit) * PAGE - len(prompt_ids))
        return requested

    def _chat_prompt(req: ChatCompletionRequest):
        """(token ids, text) the chat endpoints feed the model: the chat template with tools and thinking kwargs."""
        if (req.n or 1) != 1:
            raise bad_request("n > 1 is not supported")
        try:
            tools = normalize_tools(req.tools)
        except ValueError as e:
            raise bad_request(str(e))
        choice = normalize_tool_choice(req.tool_choice)
        enabled, budget = effort_budget(req.reasoning_effort or reasoning_effort)
        ids, text = render_chat(engine, openai_messages(req.messages), tools if choice != "none" else None,
                                **thinking_kwargs(engine, enabled))
        return ids, text, tools, choice, budget

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

    def _lane_metrics() -> Dict[str, Dict[str, int]]:
        """Per lane: requests waiting and running now, admitted and finished since start."""
        out = {}
        for lane in Lane:
            out[lane.value] = {"waiting": sum(r.lane == lane for r in engine.waiting_queue),
                               "running": sum(r.lane == lane for r in engine.running_batch), **lane_stats[lane.value]}
        return out

    @app.get("/metrics")
    async def metrics(request: Request):
        """Prometheus text exposition (format 0.0.4); ?format=json returns the engine telemetry as JSON."""
        telem = engine.get_telemetry()
        telem["lanes"] = _lane_metrics()
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
        for kind, help_text in (("waiting", "Requests waiting"), ("running", "Requests running"),
                                ("admitted", "Requests admitted (total)"), ("finished", "Requests finished (total)")):
            name = f"turing_lane_requests_{kind}" + ("_total" if kind in ("admitted", "finished") else "")
            lines += [f"# HELP {name} {help_text} per QoS lane",
                      f"# TYPE {name} {'counter' if name.endswith('_total') else 'gauge'}"]
            lines += [f'{name}{{lane="{lane}"}} {v[kind]}' for lane, v in telem["lanes"].items()]
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
        if (error := embedding_shape_error(vecs.shape[-1], req.dimensions, req.encoding_format or "float")) is not None:
            raise HTTPException(status_code=400, detail=error)
        vectors = shape_embeddings(vecs, req.dimensions, req.encoding_format or "float")
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
        r = await extract_spans(engine, req.text, req.labels, max_tokens=req.max_tokens)
        if r.get("error"):
            return JSONResponse(status_code=422, content={"error": {
                "message": "the model's output is not a JSON array of {label, text} objects; the served model could "
                           "not extract spans", "type": "invalid_model_output", "raw_output": r["raw_output"]}})
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
    async def render_completions(raw_req: Request):
        """The token ids /v1/completions feeds the model for this prompt (llm-d EPP prefix indexing)."""
        body = await raw_req.json()
        tokens = render_text(engine, body.get("prompt", ""))
        return JSONResponse(content={"tokens": tokens, "count": len(tokens)})

    @app.post("/v1/chat/completions/render")
    async def render_chat_completions(raw_req: Request):
        """The token ids /v1/chat/completions feeds the model for these messages (chat template, tools, thinking)."""
        try:
            req = ChatCompletionRequest.model_validate(await raw_req.json())
        except ValidationError as e:
            raise bad_request(str(e))
        tokens = _chat_prompt(req)[0]
        return JSONResponse(content={"tokens": tokens, "count": len(tokens)})

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw_req: Request):
        ctl = _controls(await raw_req.json(), raw_req)
        if (req.n or 1) != 1:
            raise bad_request("n > 1 is not supported")
        prompt_ids = render_text(engine, req.prompt)
        params = GenerationParams(
            max_tokens=_max_tokens(req.max_tokens, prompt_ids, 64), temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, stop=req.stop, schema=_response_schema(req.response_format),
            reasoning=False, **ctl)
        gen = await _start(prompt_ids, params)  # admitted before the response starts: a rejection is a 4xx/503
        rid, created, model = f"cmpl-{uuid.uuid4().hex[:24]}", int(time.time()), req.model or cfg.name

        def body(text: str, finish: Optional[str]) -> Dict[str, Any]:
            choice = {"text": text, "index": 0, "logprobs": None, "finish_reason": finish}
            if finish and gen.schema_error:
                choice["schema_error"] = gen.schema_error
            return {"id": rid, "object": "text_completion", "created": created, "model": model, "choices": [choice]}

        if req.stream:
            async def event_generator():
                async for _, text in gen.events():
                    yield _sse(body(text, None))
                yield _sse(body("", gen.finish_reason))
                if (req.stream_options or {}).get("include_usage"):
                    yield _sse({**body("", None), "choices": [], "usage": _usage(gen)})
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_headers(gen))

        await gen.collect()
        return JSONResponse(content={**body(gen.content, gen.finish_reason), "usage": _usage(gen)},
                            headers=_headers(gen))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw_req: Request):
        ctl = _controls(await raw_req.json(), raw_req)
        prompt_ids, prompt_text, tools, choice, budget = _chat_prompt(req)
        params = GenerationParams(
            max_tokens=_max_tokens(req.max_completion_tokens or req.max_tokens, prompt_ids, 64),
            temperature=req.temperature, top_p=req.top_p, top_k=req.top_k, stop=req.stop, tools=tools,
            tool_choice=choice, schema=_response_schema(req.response_format), thinking_budget=budget, **ctl)
        gen = await _start(prompt_ids, params, prompt_text)
        rid, created, model = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time()), req.model or cfg.name

        def chunk(delta: Dict[str, Any], finish: Optional[str] = None) -> Dict[str, Any]:
            c = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
            if finish and gen.schema_error:
                c["schema_error"] = gen.schema_error
            return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [c]}

        if req.stream:
            async def event_generator():
                yield _sse(chunk({"role": "assistant", "content": ""}))
                async for kind, text in gen.events():
                    yield _sse(chunk({"reasoning_content": text} if kind == "reasoning" else {"content": text}))
                if gen.tool_calls:
                    yield _sse(chunk({"tool_calls": [{"index": i, **tc} for i, tc in enumerate(gen.tool_calls)]}))
                yield _sse(chunk({}, gen.finish_reason))
                if (req.stream_options or {}).get("include_usage"):
                    yield _sse({**chunk({}), "choices": [], "usage": _usage(gen)})
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream", headers=_headers(gen))

        await gen.collect()
        message: Dict[str, Any] = {"role": "assistant", "content": gen.content or (None if gen.tool_calls else "")}
        if gen.reasoning_text:
            message["reasoning_content"] = gen.reasoning_text
        if gen.tool_calls:
            message["tool_calls"] = gen.tool_calls
        out = {"index": 0, "message": message, "logprobs": None, "finish_reason": gen.finish_reason}
        if gen.schema_error:
            out["schema_error"] = gen.schema_error
        return JSONResponse(content={"id": rid, "object": "chat.completion", "created": created, "model": model,
                                     "choices": [out], "usage": _usage(gen)}, headers=_headers(gen))

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessageRequest, raw_req: Request):
        """
        Anthropic Messages API Endpoint (/v1/messages).
        """
        ctl = _controls(await raw_req.json(), raw_req)
        try:
            tools = normalize_tools(req.tools)
        except ValueError as e:
            raise bad_request(str(e))
        choice = normalize_tool_choice(req.tool_choice)
        enabled, budget = AnthropicAPIHandler.thinking(req)
        if enabled is None:
            enabled, budget = effort_budget(reasoning_effort)
        prompt_ids, prompt_text = render_chat(engine, AnthropicAPIHandler.to_chat_messages(req),
                                              tools if choice != "none" else None, **thinking_kwargs(engine, enabled))
        params = GenerationParams(
            max_tokens=_max_tokens(req.max_tokens, prompt_ids, 1024), temperature=req.temperature, top_p=req.top_p,
            top_k=req.top_k, stop=req.stop_sequences, tools=tools, tool_choice=choice, thinking_budget=budget, **ctl)
        gen = await _start(prompt_ids, params, prompt_text)

        if req.stream:
            return StreamingResponse(AnthropicAPIHandler.stream_generator(req, gen), media_type="text/event-stream",
                                     headers=_headers(gen))

        await gen.collect()
        response = AnthropicAPIHandler.format_non_streaming_response(
            req=req, generated_text=gen.content, input_tokens_count=gen.prompt_tokens,
            output_tokens_count=gen.completion_tokens, reasoning=gen.reasoning_text, tool_calls=gen.tool_calls,
            finish_reason=gen.finish_reason, stop_sequence=gen.stop_sequence, cached_tokens=gen.cached_tokens)
        return JSONResponse(content=response.model_dump(exclude_none=True), headers=_headers(gen))

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
        except (RepositoryNotFoundError, RevisionNotFoundError):
            where = f"{repo}@{revision}" if revision else repo
            return JSONResponse(status_code=404, content={"error": f"pull model manifest: {where} not found"})
        except Exception:  # no network, hub errors: details go to the server log, not the client
            logger.exception("pull %s: listing the repository failed", repo)
            return JSONResponse(status_code=500, content={"error": f"pull model manifest: could not list {repo} "
                                                                   "on the Hugging Face Hub"})
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
        Ollama single-prompt completion endpoint (/api/generate): raw prompts, a `template` override, `system`,
        `context` from an earlier response, `format`, `think`.
        """
        ctl = _controls(await raw_req.json(), raw_req)
        t0 = time.time()
        if req.images:
            raise bad_request("images are not supported by this server (text only)")
        if req.suffix:
            raise bad_request("suffix (fill-in-the-middle) is not supported by this server")
        if not req.prompt:  # Ollama loads the model and answers done
            return JSONResponse(content={"model": req.model, "response": "", "done": True, "done_reason": "load",
                                         "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        context = list(req.context or [])
        if req.raw:
            prompt_text = req.prompt
            prompt_ids = context + engine.encode_prompt(prompt_text)
        elif req.template:
            prompt_text = render_go_template(req.template, {"System": req.system or "", "Prompt": req.prompt,
                                                            "Response": ""})
            prompt_ids = context + engine.encode_prompt(prompt_text)
        else:
            messages = ([{"role": "system", "content": req.system}] if req.system else []) + \
                [{"role": "user", "content": req.prompt}]
            ids, prompt_text = render_chat(engine, messages, **thinking_kwargs(engine, req.think))
            prompt_ids = context + ids
        opts = ollama_options(req.options)
        opts["max_tokens"] = _max_tokens(opts["max_tokens"], prompt_ids, 128)
        params = GenerationParams(**opts, schema=ollama_format(req.format), reasoning=not req.raw,
                                  thinking_budget=0 if req.think is False else None, **ctl)
        gen = await _start(prompt_ids, params, prompt_text)

        if req.stream:
            async def ollama_generate_stream():
                async for kind, text in gen.events():
                    piece = {"thinking": text} if kind == "reasoning" else {"text": text}
                    yield json.dumps(OllamaAPIHandler.generate_chunk(req, **piece)) + "\n"
                yield json.dumps(OllamaAPIHandler.generate_chunk(
                    req, done=True, gen=gen, t0=t0, context=prompt_ids + gen.output_ids)) + "\n"

            return StreamingResponse(ollama_generate_stream(), media_type="application/x-ndjson",
                                     headers=_headers(gen))

        await gen.collect()
        return JSONResponse(content=OllamaAPIHandler.generate_chunk(
            req, text=gen.content, thinking=gen.reasoning_text, done=True, gen=gen, t0=t0,
            context=prompt_ids + gen.output_ids), headers=_headers(gen))

    @app.post("/api/chat")
    async def ollama_chat(req: OllamaChatRequest, raw_req: Request):
        """
        Ollama chat completion endpoint (/api/chat).
        """
        ctl = _controls(await raw_req.json(), raw_req)
        t0 = time.time()
        try:
            tools = normalize_tools(req.tools)
        except ValueError as e:
            raise bad_request(str(e))
        prompt_ids, prompt_text = render_chat(engine, OllamaAPIHandler.to_chat_messages(req), tools,
                                              **thinking_kwargs(engine, req.think))
        opts = ollama_options(req.options)
        opts["max_tokens"] = _max_tokens(opts["max_tokens"], prompt_ids, 128)
        params = GenerationParams(**opts, tools=tools, schema=ollama_format(req.format),
                                  thinking_budget=0 if req.think is False else None, **ctl)
        gen = await _start(prompt_ids, params, prompt_text)

        if req.stream:
            async def ollama_chat_stream():
                async for kind, text in gen.events():
                    piece = {"thinking": text} if kind == "reasoning" else {"text": text}
                    yield json.dumps(OllamaAPIHandler.chat_chunk(req, **piece)) + "\n"
                yield json.dumps(OllamaAPIHandler.chat_chunk(req, done=True, gen=gen, t0=t0)) + "\n"

            return StreamingResponse(ollama_chat_stream(), media_type="application/x-ndjson", headers=_headers(gen))

        await gen.collect()
        return JSONResponse(content=OllamaAPIHandler.chat_chunk(
            req, text=gen.content, thinking=gen.reasoning_text, done=True, gen=gen, t0=t0), headers=_headers(gen))

    @app.post("/api/embed")
    async def ollama_embed(req: OllamaEmbedRequest):
        """
        Ollama embeddings (/api/embed): `input` a string or a list of strings -> `embeddings`, L2-normalised.
        """
        t0 = time.time_ns()
        inputs = [req.input] if isinstance(req.input, str) else req.input
        vecs, n_tokens, name = _embed(inputs)
        if (error := embedding_shape_error(vecs.shape[-1], req.dimensions, "float")) is not None:
            return JSONResponse(status_code=400, content={"error": error})
        vectors = shape_embeddings(vecs, req.dimensions, "float")
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

