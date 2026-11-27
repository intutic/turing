"""The serving engine for `turing serve --backend libllama`: one libllama context (turing.turing_llama, a GGUF file,
CPU kernels from the pinned third_party/llama.cpp) behind the same request interface as ContinuousBatchEngine, so the
server's OpenAI / Anthropic / Ollama endpoints, chat templates, tools, reasoning and grammar constraints run on it.

Requests are served one at a time in arrival order (batch 1: decision CF2 in bmad-output/cpu-first/decision-log.md);
the context's KV cache is kept between them, so a request that shares a prompt prefix with the last one only evaluates
the rest. A request without a logits processor runs libllama's own sampling loop; one with a processor (grammar,
tools, a thinking budget) gets its logits in Python each step and the engine's own sampler, as ContinuousBatchEngine
samples."""
import asyncio
import time
import uuid
from types import SimpleNamespace
from typing import AsyncGenerator, List, Optional

import torch

from turing.core.kv_cache import PAGE
from turing.serving.engine import ContinuousBatchEngine, _eos_ids


class _Request:
    def __init__(self, prompt_tokens, max_new_tokens, temperature, top_k, top_p, logits_processor):
        self.request_id = uuid.uuid4().hex[:8]
        self.prompt_tokens = list(prompt_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.logits_processor = logits_processor
        self.generated_tokens: List[int] = []
        self.draft_tokens = 0
        self.lane = None  # no QoS lanes: one request at a time
        self.adapter = None
        self.cached_prompt_tokens = 0
        self.finish_reason: Optional[str] = None
        self.error: Optional[BaseException] = None
        self.cancelled = False
        self.arrival = time.time()
        self.first_token_time: Optional[float] = None
        self.output_queue: asyncio.Queue = asyncio.Queue()


class _ContextPool:
    """What the server reads of a KV pool, for one libllama context of n_ctx tokens."""

    def __init__(self, llm):
        self.llm = llm
        self.page_size = PAGE
        self.table_width = self.page_limit = llm.n_ctx // PAGE

    def get_stats(self) -> dict:
        pages = self.llm.n_ctx // PAGE
        return {"total_pages": pages, "max_physical_pages": pages, "used_pages": -(-self.llm.n_past // PAGE),
                "page_size": PAGE, "bytes_per_page": 0, "native_layers": None}


class _NoModule(torch.nn.Module):
    """The weights live in libllama, not in torch: no parameters, no LoRA adapters."""
    adapter_names = ()


class LlamaCppEngine:
    # the engine's own sampling and logits-processor semantics (they read only the request)
    _probs = ContinuousBatchEngine._probs
    _processed = ContinuousBatchEngine._processed

    def __init__(self, path: str, tokenizer, name: str, metadata: Optional[dict] = None, n_ctx: int = 4096,
                 n_threads: int = 0, seed: int = 0, draft_tokens: int = 0, **llama_kwargs):
        """metadata: the GGUF's key/values (GGUFReader(path, willneed=False).metadata), for the model's shape.
        draft_tokens: prompt-lookup drafts per step for greedy requests without a logits processor (0 = off). Greedy in
        exact arithmetic; libllama's batched kernels round differently from its one-token kernels (fp16 KV, 4-bit
        weights), so a near-tie can flip a token (results_cpu_waves123_c4a.json)."""
        from turing import turing_llama  # built only with TURING_BUILD_LLAMA=1
        self.llm = turing_llama.Llama(path, n_ctx=n_ctx, n_threads=n_threads, **llama_kwargs)
        self.tokenizer = tokenizer
        meta = metadata or {}
        arch = meta.get("general.architecture", "llama")

        def shape(key, default=0):
            v = meta.get(f"{arch}.{key}", default)
            return max(v) if isinstance(v, (list, tuple)) else v  # per-layer arrays (some MoE / hybrid models)

        heads = shape("attention.head_count")
        self.model_config = SimpleNamespace(
            name=name, arch=arch, vocab_size=self.llm.n_vocab, max_position_embeddings=self.llm.n_ctx,
            hidden_dim=shape("embedding_length"), num_layers=shape("block_count"), num_heads=heads,
            num_kv_heads=shape("attention.head_count_kv", heads), ffn_dim=shape("feed_forward_length"),
            kv_cache_dtype="f32" if llama_kwargs.get("exact") else "f16",  # libllama's KV storage
            quantization=None, sparsity_ratio=0.0, total_tiles=1, active_tiles=1, active_subspace_dim=0)
        self.device = torch.device("cpu")
        self.model = _NoModule()
        self.kv_pool = _ContextPool(self.llm)
        self.eos_token_ids = _eos_ids(tokenizer) | ({self.llm.eos} if self.llm.eos >= 0 else set())
        self.default_draft_tokens = draft_tokens
        self.admission = None
        self.lane_policy = None
        self.running_batch: List[_Request] = []
        self.waiting_queue: List[_Request] = []
        self.is_running = False
        self._seed = seed
        self._wake: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._started = time.time()
        self._tokens = 0
        self._completed = 0
        self._ttft: List[float] = []
        self._itl: List[float] = []

    # ------------------------------------------------------------------ text
    encode_prompt = ContinuousBatchEngine.encode_prompt
    decode_tokens = ContinuousBatchEngine.decode_tokens

    # ------------------------------------------------------------------ lifecycle
    async def start(self):
        if not self.is_running:
            self.is_running = True
            self._wake = asyncio.Event()
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self.is_running = False
        if self._task is not None:
            self._wake.set()
            await self._task
            self._task = None

    async def warm_up(self) -> float:
        t0 = time.perf_counter()
        await asyncio.to_thread(self.llm.prefill, [max(self.llm.bos, 0)])
        return time.perf_counter() - t0

    # ------------------------------------------------------------------ requests
    async def add_request(self, prompt_tokens: List[int], max_new_tokens: int, temperature: float = 0.0, top_k: int = 0,
                          top_p: float = 1.0, draft_tokens: Optional[int] = None, lane=None, prefix_kv=None,
                          logits_processor=None, adapter=None, **_) -> _Request:
        from turing.serving.generation import bad_request
        if prefix_kv is not None or adapter:
            raise bad_request("the libllama backend serves neither a caller's KV cache nor LoRA adapters")
        if len(prompt_tokens) + max_new_tokens > self.llm.n_ctx:
            raise bad_request(f"prompt plus max_tokens is {len(prompt_tokens) + max_new_tokens} tokens, but this "
                              f"server holds at most {self.llm.n_ctx} (--ctx)")
        req = _Request(prompt_tokens, max_new_tokens, temperature, top_k, top_p, logits_processor)
        req.draft_tokens = self.default_draft_tokens if draft_tokens is None else draft_tokens
        self.waiting_queue.append(req)
        if self._wake is not None:
            self._wake.set()
        return req

    async def stream_request(self, req: _Request) -> AsyncGenerator[int, None]:
        try:
            while True:
                token = await req.output_queue.get()
                if token is None:
                    if req.error is not None:
                        raise RuntimeError(f"request {req.request_id} failed in the engine: {req.error!r}") from req.error
                    break
                yield token
        finally:
            self.cancel(req)

    def cancel(self, req: _Request) -> None:
        req.cancelled = True

    async def _loop(self):
        while self.is_running:
            if not self.waiting_queue:
                self._wake.clear()
                await self._wake.wait()
                continue
            req = self.waiting_queue.pop(0)
            if req.cancelled:
                req.output_queue.put_nowait(None)
                continue
            self.running_batch.append(req)
            loop = asyncio.get_running_loop()
            try:
                await asyncio.to_thread(self._run, req, loop)
            except BaseException as e:  # reported to the request's stream, the engine keeps serving
                req.error = e
            finally:
                self.running_batch.remove(req)
                self._completed += 1
                loop.call_soon(req.output_queue.put_nowait, None)

    def _run(self, req: _Request, loop: asyncio.AbstractEventLoop) -> None:
        """One request on the engine's worker thread: tokens go to its queue on the event loop as they come."""
        t0 = time.perf_counter()
        last = [t0]

        def emit(tok: int) -> bool:
            """False ends the generation: an end-of-sequence token (not emitted), or the client went away."""
            if req.cancelled:
                return False
            if tok in self.eos_token_ids:
                req.finish_reason = "stop"
                return False
            now = time.perf_counter()
            if req.first_token_time is None:
                req.first_token_time = now
                self._ttft.append((now - t0) * 1000)
            else:
                self._itl.append((now - last[0]) * 1000)
            last[0] = now
            req.generated_tokens.append(tok)
            self._tokens += 1
            loop.call_soon_threadsafe(req.output_queue.put_nowait, tok)
            return True

        if req.logits_processor is None:
            self.llm.generate(req.prompt_tokens, req.max_new_tokens, temperature=max(req.temperature, 0.0),
                              top_p=req.top_p, top_k=req.top_k, seed=self._seed, on_token=emit,
                              n_draft=req.draft_tokens if req.temperature <= 1e-5 else 0)
            req.cached_prompt_tokens = self.llm.cached_prompt_tokens
        else:
            logits = torch.from_numpy(self.llm.prefill(req.prompt_tokens))
            req.cached_prompt_tokens = self.llm.cached_prompt_tokens
            gen = torch.Generator().manual_seed(self._seed)
            for i in range(req.max_new_tokens):
                row = self._processed(req, logits)
                p = self._probs(req, row)
                tok = int(row.argmax()) if p is None else int(torch.multinomial(p, 1, generator=gen))
                if self.llm.is_eog(tok) or not emit(tok):
                    break
                if i + 1 < req.max_new_tokens:
                    logits = torch.from_numpy(self.llm.eval([tok]))
        if req.finish_reason is None:
            req.finish_reason = "length" if len(req.generated_tokens) >= req.max_new_tokens else "stop"

    # ------------------------------------------------------------------ metrics
    def get_kv_cache_utilization(self) -> float:
        return self.llm.n_past / self.llm.n_ctx

    def get_telemetry(self) -> dict:
        up = time.time() - self._started
        ttft = sorted(self._ttft)
        return {
            "uptime_seconds": up, "total_tokens_generated": self._tokens, "total_requests_completed": self._completed,
            "serving_throughput_tok_per_sec": self._tokens / up if up > 0 else 0.0,
            "running_requests": len(self.running_batch), "waiting_queue_depth": len(self.waiting_queue),
            "preemptions": 0, "kv_memory_pool": self.kv_pool.get_stats(),
            "latency": {"avg_ttft_ms": sum(ttft) / len(ttft) if ttft else 0.0,
                        "p99_ttft_ms": ttft[min(len(ttft) - 1, int(0.99 * len(ttft)))] if ttft else 0.0,
                        "avg_itl_ms": sum(self._itl) / len(self._itl) if self._itl else 0.0},
            "speculation": self._speculation(),
        }

    def _speculation(self) -> dict:
        s = self.llm.spec_stats()
        return {"drafted_tokens": s["drafted"], "accepted_tokens": s["accepted"],
                "acceptance_rate": s["accepted"] / s["drafted"] if s["drafted"] else 0.0}

    def get_llmd_metrics(self) -> dict:
        return {"num_requests_waiting": len(self.waiting_queue), "num_requests_running": len(self.running_batch),
                "kv_cache_usage_perc": self.get_kv_cache_utilization(), "block_size": PAGE,
                "num_gpu_blocks": self.kv_pool.table_width, "total_tokens_generated": self._tokens}
