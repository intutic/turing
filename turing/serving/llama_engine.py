"""The serving engine for `turing serve --backend libllama`: one libllama context (turing.turing_llama, a GGUF file,
CPU kernels from the pinned third_party/llama.cpp) behind the same request interface as ContinuousBatchEngine, so the
server's OpenAI / Anthropic / Ollama endpoints, chat templates, tools, reasoning and grammar constraints run on it.

With max_batch_size 1 (the default) requests are served one at a time in arrival order (batch 1: decision CF2 in
bmad-output/cpu-first/decision-log.md); the context's KV cache is kept between them, so a request that shares a prompt
prefix with the last one only evaluates the rest. With max_batch_size > 1 (Wave 7) up to that many requests decode
together, each in its own libllama sequence of one shared KV pool of n_ctx tokens: every step feeds each running request
its next input in one batched call, so a weight read serves all of them. A request without a logits processor runs libllama's own sampling loop; one with a processor (grammar,
tools, a thinking budget) gets its logits in Python each step and the engine's own sampler, as ContinuousBatchEngine
samples."""
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
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
        self.media = None  # (prompt text with media markers, encoded images) when the request has images
        self.prompt_positions: Optional[int] = None  # positions the prompt took, images included
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


class LlamaTokenizer:
    """libllama's own tokenizer (the one llama-server uses) under the Hugging Face method names the server calls, and the
    GGUF's chat template rendered as Hugging Face renders one. GGUFTokenizer is a greedy word lookup: it drops the spaces
    of byte-level BPE vocabularies (Qwen, Llama 3, gpt-oss) and ignores the chat template."""

    def __init__(self, llm, metadata: dict):
        self.llm = llm
        tokens = metadata.get("tokenizer.ggml.tokens") or []
        self._vocab = {str(t): i for i, t in enumerate(tokens)}
        self.chat_template = metadata.get("tokenizer.chat_template")
        self.bos_token_id = llm.bos if llm.bos >= 0 else None
        self.eos_token_id = llm.eos if llm.eos >= 0 else None
        self.bos_token = str(tokens[llm.bos]) if 0 <= llm.bos < len(tokens) else None
        self.eos_token = str(tokens[llm.eos]) if 0 <= llm.eos < len(tokens) else None

    def __len__(self) -> int:
        return self.llm.n_vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.llm.tokenize(text, add_special=add_special_tokens, parse_special=True)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return b"".join(self.llm.piece(int(t), not skip_special_tokens) for t in token_ids).decode("utf-8", "replace")

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        return self._vocab.get(token)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    def apply_chat_template(self, conversation, tools=None, add_generation_prompt: bool = False, tokenize: bool = False,
                            **kwargs):
        from transformers.utils.chat_template_utils import render_jinja_template  # transformers >= 4.51
        if not self.chat_template:
            raise ValueError("the GGUF has no chat template")
        rendered, _ = render_jinja_template([conversation], tools=tools, chat_template=self.chat_template,
                                            add_generation_prompt=add_generation_prompt, bos_token=self.bos_token,
                                            eos_token=self.eos_token, **kwargs)
        return self.encode(rendered[0], add_special_tokens=False) if tokenize else rendered[0]


class _NoModule(torch.nn.Module):
    """The weights live in libllama, not in torch: no parameters, no LoRA adapters."""
    adapter_names = ()


class LlamaCppEngine:
    # the engine's own sampling and logits-processor semantics (they read only the request)
    _probs = ContinuousBatchEngine._probs
    _processed = ContinuousBatchEngine._processed

    def __init__(self, path: str, tokenizer, name: str, metadata: Optional[dict] = None, n_ctx: int = 4096,
                 n_threads: int = 0, seed: int = 0, draft_tokens: int = 0, max_batch_size: int = 1,
                 mmproj: Optional[str] = None, **llama_kwargs):
        """tokenizer: None for libllama's own (LlamaTokenizer), as turing serve runs it.
        mmproj: the model's vision projector GGUF: chat requests may then carry images (one request at a time).
        metadata: the GGUF's key/values (GGUFReader(path, willneed=False).metadata), for the model's shape.
        draft_tokens: prompt-lookup drafts per step for greedy requests without a logits processor (0 = off). Greedy in
        exact arithmetic; libllama's batched kernels round differently from its one-token kernels (fp16 KV, 4-bit
        weights), so a near-tie can flip a token (results_cpu_waves123_c4a.json)."""
        from turing import turing_llama  # built only with TURING_BUILD_LLAMA=1
        self.max_batch_size = max(1, max_batch_size)
        self.llm = turing_llama.Llama(path, n_ctx=n_ctx, n_threads=n_threads, n_seq=self.max_batch_size, **llama_kwargs)
        self.media_marker = None  # the text that stands for an image in a prompt, with a projector loaded
        if mmproj:
            self.llm.load_mmproj(mmproj)
            self.media_marker = turing_llama.Llama.media_marker
        # Every libllama call runs on this one thread: libgomp keeps a team of threads per calling thread, and once a
        # process holds more OpenMP threads than CPUs it stops spinning between parallel regions (80 against 60 ms a
        # token, Qwen3-8B on 16 vCPUs: results_cpu_wave7_c4.json).
        self._compute = ThreadPoolExecutor(1, thread_name_prefix="libllama")
        meta = metadata or {}
        self.tokenizer = tokenizer if tokenizer is not None else LlamaTokenizer(self.llm, meta)
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
        self._lengths: dict = {}

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
        await self._call(self.llm.prefill, [max(self.llm.bos, 0)])
        self.llm.truncate(0)  # the batched loop hands sequence 0 out as empty
        return time.perf_counter() - t0

    async def _call(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._compute, fn, *args)

    # ------------------------------------------------------------------ requests
    async def add_request(self, prompt_tokens: List[int], max_new_tokens: int, temperature: float = 0.0, top_k: int = 0,
                          top_p: float = 1.0, draft_tokens: Optional[int] = None, lane=None, prefix_kv=None,
                          logits_processor=None, adapter=None, media=None, **_) -> _Request:
        from turing.serving.generation import bad_request
        if prefix_kv is not None or adapter:
            raise bad_request("the libllama backend serves neither a caller's KV cache nor LoRA adapters")
        if media is not None and self.media_marker is None:
            raise bad_request("this model takes no images: start turing serve with --mmproj")
        if media is not None and self.max_batch_size > 1:  # shortcut: batched decode has no image prefill, add it when asked
            raise bad_request("image input runs one request at a time: start turing serve without --max-batch-size")
        if len(prompt_tokens) + max_new_tokens > self.llm.n_ctx:
            raise bad_request(f"prompt plus max_tokens is {len(prompt_tokens) + max_new_tokens} tokens, but this "
                              f"server holds at most {self.llm.n_ctx} (--ctx)")
        req = _Request(prompt_tokens, max_new_tokens, temperature, top_k, top_p, logits_processor)
        req.media = media
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
        if self.max_batch_size > 1:
            return await self._batched_loop()
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
                await self._call(self._run, req, loop)
            except BaseException as e:  # reported to the request's stream, the engine keeps serving
                req.error = e
            finally:
                self.running_batch.remove(req)
                self._completed += 1
                loop.call_soon(req.output_queue.put_nowait, None)

    async def _batched_loop(self):
        """Wave 7: up to max_batch_size requests decode together, one libllama sequence each. A request is admitted when a
        sequence is free and the shared pool can hold its prompt plus max_tokens beside the running ones."""
        free = list(range(self.max_batch_size))[::-1]
        seq, feed, reserved = {}, {}, {}
        length = self._lengths  # tokens in each running sequence's K/V
        while self.is_running:
            while free and self.waiting_queue:
                req = self.waiting_queue[0]
                need = len(req.prompt_tokens) + req.max_new_tokens
                if req.cancelled:
                    self.waiting_queue.pop(0)
                    req.output_queue.put_nowait(None)
                    continue
                if sum(reserved.values()) + need > self.llm.n_ctx:
                    break  # waits for a running request to finish
                self.waiting_queue.pop(0)
                s = free.pop()
                seq[req], length[s], feed[req], reserved[req] = s, 0, list(req.prompt_tokens), need
                req.generator = torch.Generator().manual_seed(self._seed)
                req.start = time.perf_counter()
                self.running_batch.append(req)
            if not self.running_batch:
                self._wake.clear()
                await self._wake.wait()
                continue
            running = list(self.running_batch)
            seqs = [seq[r] for r in running]
            try:
                rows = await self._call(self.llm.step, seqs, [feed[r] for r in running], [length[s] for s in seqs])
            except BaseException as e:  # the whole step failed: fail its requests, keep serving
                for r in running:
                    r.error = e
                    self._finish(r, seq, free, reserved)
                continue
            now = time.perf_counter()
            for i, r in enumerate(running):
                s = seq[r]
                length[s] += len(feed[r])
                if r.cancelled:
                    self._finish(r, seq, free, reserved)
                    continue
                row = self._processed(r, torch.from_numpy(rows[i]))
                p = self._probs(r, row)
                tok = int(row.argmax()) if p is None else int(torch.multinomial(p, 1, generator=r.generator))
                if tok in self.eos_token_ids or self.llm.is_eog(tok):
                    r.finish_reason = "stop"
                    self._finish(r, seq, free, reserved)
                    continue
                if r.first_token_time is None:
                    r.first_token_time = now
                    self._ttft.append((now - r.start) * 1000)
                else:
                    self._itl.append((now - r.last) * 1000)
                r.last = now
                r.generated_tokens.append(tok)
                self._tokens += 1
                r.output_queue.put_nowait(tok)
                if len(r.generated_tokens) >= r.max_new_tokens:
                    r.finish_reason = "length"
                    self._finish(r, seq, free, reserved)
                else:
                    feed[r] = [tok]

    def _finish(self, r: _Request, seq: dict, free: list, reserved: dict) -> None:
        s = seq.pop(r)
        self._lengths.pop(s, None)
        self.llm.seq_clear(s)
        free.append(s)
        reserved.pop(r, None)
        self.running_batch.remove(r)
        self._completed += 1
        if r.finish_reason is None:
            r.finish_reason = "stop"
        r.output_queue.put_nowait(None)

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

        if req.logits_processor is None and req.media is None:
            self.llm.generate(req.prompt_tokens, req.max_new_tokens, temperature=max(req.temperature, 0.0),
                              top_p=req.top_p, top_k=req.top_k, seed=self._seed, on_token=emit,
                              n_draft=req.draft_tokens if req.temperature <= 1e-5 else 0)
            req.cached_prompt_tokens = self.llm.cached_prompt_tokens
        else:
            if req.media is not None:
                logits = torch.from_numpy(self.llm.prefill_media(*req.media))
                req.prompt_positions = self.llm.n_past
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
        used = sum(self._lengths.values()) if self.max_batch_size > 1 else self.llm.n_past
        return used / self.llm.n_ctx

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
