"""
Continuous Batching Scheduler and Async Stateful Inference Serving Engine.
Supports chunked prefill, speculative interleaving, and Prometheus telemetry.
"""

import asyncio
import uuid
import statistics
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional, AsyncGenerator, Deque, Any, Callable, Set
from collections import deque
import torch
import torch.nn.functional as F

from ..config import ModelConfig, TuringConfig
from ..models.causal_lm import SubspaceCausalLM
from ..core.kv_cache import PAGE, KVCache, KVPoolExhausted, kv_bytes_per_token
from .traffic import (
    AdmissionController, AdmissionDecision,
    AdmissionResult, Lane, LanePolicy
)
from .spec_gate import SpecGateDecision, SpeculationGatePolicy
from .prefix_cache import PrefixCache

class RequestRejected(RuntimeError):
    """The engine cannot take this request now. The HTTP layer returns status_code (429 or 503) with Retry-After."""

    def __init__(self, message: str, status_code: int = 503, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code, self.retry_after = status_code, retry_after


def _eos_ids(tokenizer) -> Set[int]:
    """End-of-sequence ids of a tokenizer: eos_token_id plus the generation config's (chat models end turns with
    <|im_end|>, <|eot_id|>, <end_of_turn> ...). Empty without a tokenizer (synthetic models run to max_tokens)."""
    if tokenizer is None:
        return set()
    ids = set()
    eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos, int):
        ids.add(eos)
    for tok in ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if isinstance(tid, int) and tid != getattr(tokenizer, "unk_token_id", None):
            ids.add(tid)
    return ids


class RequestState(Enum):
    WAITING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3

class AsyncSequenceRequest:
    """One generation request. logits_processor(generated, logits) may change the next-token logits [V] in place or
    return new ones (constrained decoding, reasoning budgets); it sees every token generated so far. Generation ends at
    max_new_tokens ("length"), at a token in stop_token_ids ("stop", the token is not emitted), or by cancel()."""

    def __init__(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        request_id: Optional[str] = None,
        draft_tokens: Optional[int] = None,
        lane: Optional[Lane] = None,
        top_p: float = 1.0,
        stop_token_ids: Optional[Set[int]] = None,
        logits_processor: Optional[Callable[[List[int], torch.Tensor], Optional[torch.Tensor]]] = None,
        adapter: Optional[str] = None,
    ):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.prompt_tokens = list(prompt_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.draft_tokens = draft_tokens
        self.lane = lane
        self.stop_token_ids = set(stop_token_ids or ())
        self.logits_processor = logits_processor
        self.adapter = adapter
        self.finish_reason: Optional[str] = None
        self.cached_prompt_tokens = 0  # prompt tokens served from the prefix cache
        self.uncacheable = False  # continues a caller's KVCache (prefix_kv): its pages are not keyed by its tokens
        self.needs_pages = False  # admitted while the pool was short: reserves its pages when it reaches the front
        self.state = RequestState.WAITING
        self.generated_tokens: List[int] = []
        self.output_queue: asyncio.Queue[Optional[int]] = asyncio.Queue()
        self.created_at: float = time.time()
        self.first_token_at: Optional[float] = None
        self.last_token_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.past_kv = None
        self.prefill_offset: int = 0
        self.current_input: Optional[torch.Tensor] = None

class ContinuousBatchEngine:
    """
    High-Throughput Continuous Batching Engine managing iteration-level scheduling,
    chunked prefilling, and asynchronous streaming token generation with Static Paged KV Pooling.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        turing_config: Optional[TuringConfig] = None,
        prefill_chunk_size: int = 512,
        model: Optional[SubspaceCausalLM] = None,
        tokenizer: Optional[Any] = None,
        admission: Optional[AdmissionController] = None,
        lane_policy: Optional[LanePolicy] = None,
        spec_gate: Optional[SpeculationGatePolicy] = None,
        dist_driver: Optional[Any] = None,
        prefix_caching: bool = True,
        draft_tokens: int = 0,
        eos_token_ids: Optional[Set[int]] = None,
        remote_prefill: Optional[Tuple[str, int]] = None,
        max_waiting: int = 1024,
    ):
        self.model_config = model_config
        self.turing_config = turing_config or TuringConfig()
        self.device = self.turing_config.resolve_device()
        self.prefill_chunk_size = prefill_chunk_size
        self.tokenizer = tokenizer
        self.admission = admission
        self.lane_policy = lane_policy
        self.spec_gate = spec_gate
        self.dist_driver = dist_driver

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = SubspaceCausalLM(model_config).to(self.device).eval()

        # The model's paged KV pool, sized once from free memory after the weights (ModelConfig.kv_pool_*)
        self.kv_pool = self.model.kv_pool()
        self._rope_switch = self.model.rope_switch_length()  # LongRoPE's original context (Phi-3), else None
        self.prefix_cache = PrefixCache(self.kv_pool) if prefix_caching and dist_driver is None else None
        self.eos_token_ids = set(eos_token_ids) if eos_token_ids is not None else _eos_ids(tokenizer)
        self.default_draft_tokens = draft_tokens  # n-gram drafts per step when the gate allows (0 = off)
        self.spec_drafted = self.spec_accepted = 0
        self.remote_prefill = remote_prefill  # (host, port) of a PrefillWorker: prompts are prefilled there
        self.max_waiting = max_waiting  # requests that may wait for KV pages before new ones get 503

        self.max_batch_size = self.turing_config.max_batch_size
        self.waiting_queue: List[AsyncSequenceRequest] = []
        self.running_batch: List[AsyncSequenceRequest] = []
        self.is_running = False
        self._loop_task: Optional[asyncio.Task] = None

        # Telemetry & Performance Tracking
        self.start_time: float = time.time()
        self.total_tokens_generated: int = 0
        self.total_requests_completed: int = 0
        self.recent_ttft: Deque[float] = deque(maxlen=1000)
        self.recent_itl: Deque[float] = deque(maxlen=10000)

    def encode_prompt(self, text: str) -> List[int]:
        """Encodes text using real tokenizer or ASCII fallback."""
        if self.tokenizer is not None:
            try:
                return self.tokenizer.encode(text)
            except Exception:
                pass
        return [ord(c) % self.model_config.vocab_size for c in text]

    def decode_tokens(self, tokens: List[int]) -> str:
        """Decodes token IDs into coherent text using real tokenizer or ASCII fallback."""
        if self.tokenizer is not None:
            try:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception:
                pass
        return "".join([chr(t % 128) if (32 <= (t % 128) <= 126) else f"<{t}>" for t in tokens])

    async def start(self):
        if not self.is_running:
            self.is_running = True
            self.start_time = time.time()
            self._loop_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        self.is_running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def add_request(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        draft_tokens: Optional[int] = None,
        lane: Optional[Lane] = None,
        prefix_kv: Optional[Any] = None,
        top_p: float = 1.0,
        stop_token_ids: Optional[Set[int]] = None,
        logits_processor: Optional[Callable] = None,
        adapter: Optional[str] = None,
    ) -> AsyncSequenceRequest:
        """Admit a request: reserve every KV page it can need (reusing cached prefix pages) and queue it. Raises
        RequestRejected (400 never fits, 429 admission says wait, 503 overloaded). stop_token_ids defaults to the
        tokenizer's end-of-sequence ids; pass an empty set to generate exactly max_new_tokens."""
        if lane is None and self.lane_policy is not None:
            lane = self.lane_policy.classify_request(max_tokens=max_new_tokens, stream=True)
        if adapter is not None and adapter not in getattr(self.model, "adapter_names", ()):
            raise RequestRejected(f"unknown LoRA adapter '{adapter}'", status_code=400)
        if self.remote_prefill is not None and prefix_kv is None and adapter is None and logits_processor is None:
            return await self._add_remote_prefilled(prompt_tokens, max_new_tokens, temperature, top_k, top_p,
                                                    stop_token_ids, lane)
        req = AsyncSequenceRequest(
            prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            draft_tokens=draft_tokens, lane=lane, top_p=top_p,
            stop_token_ids=self.eos_token_ids if stop_token_ids is None else stop_token_ids,
            logits_processor=logits_processor, adapter=adapter)
        if not req.prompt_tokens:
            raise RequestRejected("empty prompt", status_code=400)

        def reject(msg, status, retry=None):
            req.state = RequestState.FINISHED
            req.output_queue.put_nowait(None)
            if self.admission is not None:
                self.admission.release(req.request_id)
            raise RequestRejected(f"Request {req.request_id} rejected: {msg}", status_code=status, retry_after=retry)

        if self.admission is not None:
            cfg = self.model_config
            native_bytes = next(self.model.parameters()).element_size()
            est_bytes = (len(prompt_tokens) + max_new_tokens) * kv_bytes_per_token(
                cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, cfg.kv_cache_dtype, native_bytes)
            adm = self.admission.admit(req.request_id, est_bytes)
            if adm.decision in (AdmissionDecision.SHED, AdmissionDecision.QUEUE):
                shed = adm.decision == AdmissionDecision.SHED
                reject(f"admission controller: {adm.reason or 'VRAM limit exceeded'}", 503 if shed else 429,
                       adm.retry_after_seconds)

        # A request larger than the pool (or the page table) can never run: that is the client's error, not overload
        prefix_len = prefix_kv.length if isinstance(prefix_kv, KVCache) else 0
        need = prefix_len + len(prompt_tokens) + max_new_tokens
        cap = min(self.kv_pool.table_width, self.kv_pool.page_limit) * PAGE
        if need > cap:
            reject(f"prompt plus max_tokens is {need} tokens, but this server holds at most {cap} per request", 400)

        # Reserve every page the request can need, so it can never run out of pages mid-decode. A request that fits
        # the pool but not its free pages right now waits in the queue and reserves when pages free up.
        if isinstance(prefix_kv, KVCache):
            try:
                req.past_kv = prefix_kv.clone()
                req.past_kv.rope_len = prefix_len + len(prompt_tokens)
                req.uncacheable = True  # its cache starts with tokens this request does not know: never index it
                self._reserve(req.past_kv, req.past_kv.length + len(prompt_tokens) + max_new_tokens)
            except KVPoolExhausted as e:
                if req.past_kv is not None:
                    req.past_kv.free()
                req.past_kv = None
                reject(str(e), 503, 1.0)
        elif not self._try_reserve(req):
            waiting = sum(1 for r in self.waiting_queue if r.needs_pages)
            if waiting >= self.max_waiting:
                reject(f"KV pool full and {waiting} requests already wait for pages", 503, 1.0)
            req.needs_pages = True

        self.waiting_queue.append(req)
        return req

    async def _add_remote_prefilled(self, prompt_tokens, max_new_tokens, temperature, top_k, top_p, stop_token_ids,
                                    lane) -> AsyncSequenceRequest:
        """Prefill/decode disaggregation: the prefill worker at self.remote_prefill computes the prompt and its first
        token; its KV pages are imported here and decoding continues. The returned request streams the first token
        too, as a local request would."""
        from .disagg_sidecar import remote_prefill
        from .kv_transfer import import_kv
        token, header, blob = await asyncio.to_thread(remote_prefill, self.remote_prefill, list(prompt_tokens),
                                                      temperature, top_k, top_p)
        stop = self.eos_token_ids if stop_token_ids is None else set(stop_token_ids)
        if token in stop or max_new_tokens <= 1:
            req = AsyncSequenceRequest(list(prompt_tokens), max_new_tokens, temperature, top_k, lane=lane)
            if token not in stop:
                req.generated_tokens.append(token)
                req.output_queue.put_nowait(token)
            req.state, req.finish_reason = RequestState.FINISHED, "stop" if token in stop else "length"
            req.output_queue.put_nowait(None)
            return req
        cache = import_kv(self.kv_pool, header, blob)
        try:
            req = await self.add_request([token], max_new_tokens=max_new_tokens - 1, temperature=temperature,
                                         top_k=top_k, top_p=top_p, stop_token_ids=stop, lane=lane, prefix_kv=cache)
        finally:
            cache.free()  # the request holds its own references to the imported pages
        req.output_queue.put_nowait(token)  # nothing has run since admission: the first token streams first
        req.max_new_tokens += 1
        req.generated_tokens.insert(0, token)
        req.remote_prompt_tokens = len(prompt_tokens)
        return req

    def _try_reserve(self, req: AsyncSequenceRequest) -> bool:
        """Give req a cache holding every page it can need (reusing cached prefix pages); False if the pool is short."""
        if self.prefix_cache is not None:  # at least one prompt token is prefilled
            req.past_kv = self.prefix_cache.attach(req.prompt_tokens, self._prefix_key(req.adapter, len(req.prompt_tokens)),
                                                   limit=len(req.prompt_tokens) - 1)
            req.prefill_offset = req.cached_prompt_tokens = req.past_kv.length
        else:
            req.past_kv = KVCache(self.kv_pool)
        req.past_kv.rope_len = len(req.prompt_tokens)  # chunked prefill uses what one forward over the prompt would
        try:
            self._reserve(req.past_kv, len(req.prompt_tokens) + req.max_new_tokens)
            return True
        except KVPoolExhausted:
            req.past_kv.free()
            req.past_kv, req.prefill_offset, req.cached_prompt_tokens = None, 0, 0
            return False

    def _reserve(self, cache: KVCache, tokens: int) -> None:
        """cache.reserve(tokens), evicting unused prefix-cache pages when the pool is short."""
        try:
            cache.reserve(tokens)
        except KVPoolExhausted:
            if self.prefix_cache is None:
                raise
            self.prefix_cache.evict(-(-tokens // PAGE) - len(cache.pages))
            cache.reserve(tokens)

    def cancel(self, req: AsyncSequenceRequest) -> None:
        """End a request early (client gone, stop string seen): its pages return to the pool at the next step."""
        if req.state != RequestState.FINISHED:
            req.cancelled = True

    async def submit_with_shared_prefix(
        self,
        prefix_kv: Any,
        suffix_tokens_list: List[List[int]],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        lane: Optional[Lane] = None
    ) -> List[AsyncSequenceRequest]:
        """N requests continuing one prefilled prefix (DSL fork() branches): each shares its full pages copy-on-write."""
        return [await self.add_request(prompt_tokens=suffix, max_new_tokens=max_new_tokens, temperature=temperature,
                                       top_k=top_k, lane=lane, prefix_kv=prefix_kv) for suffix in suffix_tokens_list]

    async def stream_generate(self, prompt_tokens: List[int], max_new_tokens: int = 32, temperature: float = 0.7,
                              top_k: int = 50, **kw) -> AsyncGenerator[int, None]:
        req = await self.add_request(prompt_tokens=prompt_tokens, max_new_tokens=max_new_tokens,
                                     temperature=temperature, top_k=top_k, **kw)
        async for token in self.stream_request(req):
            yield token

    async def stream_request(self, req: AsyncSequenceRequest) -> AsyncGenerator[int, None]:
        """Tokens of a request already admitted by add_request (admit first so a rejection precedes any response).
        Closing the generator early (client disconnect) cancels the request."""
        try:
            while True:
                token = await req.output_queue.get()
                if token is None:
                    break
                yield token
        finally:
            self.cancel(req)

    async def _scheduler_loop(self):
        while self.is_running:
            if self.spec_gate is not None:
                self.spec_gate.gate_decision(len(self.running_batch))
            if self.lane_policy is not None and self.waiting_queue:
                self.waiting_queue.sort(key=lambda r: self.lane_policy.priority(r.lane) if r.lane else 0)
            while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
                req = self.waiting_queue[0]
                if getattr(req, "cancelled", False):
                    self.waiting_queue.pop(0)
                    self._finish_request(req, time.time(), "cancelled")
                    continue
                if req.needs_pages:
                    if not self._try_reserve(req):
                        break  # first come, first served: later requests wait behind it
                    req.needs_pages = False
                self.waiting_queue.pop(0)
                req.state = RequestState.PREFILLING
                self.running_batch.append(req)
            if not self.running_batch:
                await asyncio.sleep(0.001)
                continue
            self._step_batch()
            await asyncio.sleep(0)

    # ------------------------------------------------------------------ one scheduler step

    def _forward(self, ids: torch.Tensor, caches, adapters=None, logits_to_keep: int = 0):
        if self.dist_driver is not None:
            return self.dist_driver.forward_distributed(ids, past_key_values=caches)
        kw = {"adapters": adapters} if adapters is not None and any(adapters) else {}
        return self.model(ids, past_key_values=caches, logits_to_keep=logits_to_keep, **kw)

    def _step_batch(self):
        if not self.running_batch:
            return
        for req in self.running_batch:
            if getattr(req, "cancelled", False) and req.state != RequestState.FINISHED:
                self._finish_request(req, time.time(), "cancelled")
        self.running_batch = [r for r in self.running_batch if r.state != RequestState.FINISHED]

        with torch.inference_mode():
            # Prefill: chunks of the prefilling requests, in priority order, until this step's token budget is spent
            # (compute-bound); a request's chunk is its remaining prompt, capped by what is left of the budget
            prefill = [r for r in self.running_batch if r.state == RequestState.PREFILLING]
            budget = None
            for req in prefill:
                if budget is None:
                    budget = self.lane_policy.prefill_chunk_budget(req.lane) if (self.lane_policy and req.lane) \
                        else self.prefill_chunk_size
                if budget <= 0:
                    break
                end = min(req.prefill_offset + budget, len(req.prompt_tokens))
                budget -= end - req.prefill_offset
                chunk = torch.tensor([req.prompt_tokens[req.prefill_offset:end]], dtype=torch.long, device=self.device)
                logits, req.past_kv = self._forward(chunk, req.past_kv, [req.adapter], logits_to_keep=1)
                req.prefill_offset = end
                if end >= len(req.prompt_tokens):
                    req.state = RequestState.DECODING
                    if self.prefix_cache is not None and self.dist_driver is None and not req.uncacheable:
                        self.prefix_cache.insert(req.prompt_tokens, req.past_kv,
                                                 self._prefix_key(req.adapter, req.past_kv.rope_len))
                    token = self._sample([req], logits[:, -1])[0]
                    now = time.time()  # after the forward and the sample (which syncs), so TTFT includes prefill
                    req.first_token_at = req.last_token_at = now
                    self.recent_ttft.append(now - req.created_at)
                    self._emit(req, token, now)

            # Decode: speculative rows one by one, the rest in one batched forward (bandwidth-bound)
            decode = [r for r in self.running_batch if r.state == RequestState.DECODING and r.first_token_at is not None
                      and r.generated_tokens]
            crossing = [r for r in decode if self._crosses_rope_switch(r, 1)]
            for r in crossing:
                self._recompute_long_rope(r)
            decode = [r for r in decode if r.state == RequestState.DECODING and r not in crossing]
            spec, plain = [], []
            for r in decode:
                drafts = self._drafts(r, len(decode))
                if drafts and self._crosses_rope_switch(r, 1 + len(drafts)):
                    drafts = None  # verify only up to the switch; the step that crosses it recomputes the cache
                (spec if drafts else plain).append((r, drafts))
            for r, drafts in spec:
                self._speculative_step(r, drafts)
            plain = [r for r, _ in plain]
            if len(plain) > 1 and self.dist_driver is None:
                ids = torch.tensor([[r.generated_tokens[-1]] for r in plain], dtype=torch.long, device=self.device)
                logits, _ = self._forward(ids, [r.past_kv for r in plain], [r.adapter for r in plain])
                tokens = self._sample(plain, logits[:, -1])
                now = time.time()
                for r, t in zip(plain, tokens):
                    self._emit(r, t, now)
            else:
                for r in plain:
                    ids = torch.tensor([[r.generated_tokens[-1]]], dtype=torch.long, device=self.device)
                    logits, r.past_kv = self._forward(ids, r.past_kv, [r.adapter])
                    self._emit(r, self._sample([r], logits[:, -1])[0], time.time())

        self.running_batch = [r for r in self.running_batch if r.state != RequestState.FINISHED]

    def _prefix_key(self, adapter: Optional[str], rope_len: int) -> Optional[str]:
        """The prefix-cache namespace: keys computed with LongRoPE's long frequencies never match short ones."""
        if self._rope_switch is None or rope_len <= self._rope_switch:
            return adapter
        return f"{adapter or ''}\x00longrope"

    def _crosses_rope_switch(self, req: AsyncSequenceRequest, new_tokens: int) -> bool:
        """True when req's cache holds short-frequency keys and new_tokens more would pass LongRoPE's switch."""
        c = req.past_kv
        return (self._rope_switch is not None and isinstance(c, KVCache) and not req.uncacheable
                and c.rope_len <= self._rope_switch < c.length + new_tokens)

    def _recompute_long_rope(self, req: AsyncSequenceRequest) -> None:
        """The sequence is about to pass LongRoPE's original context: as transformers does for Phi-3
        (Phi3ForCausalLM.prepare_inputs_for_generation), recompute the whole cache with the long frequencies, in one
        forward into a fresh cache, and emit the next token from it."""
        tokens = req.prompt_tokens + req.generated_tokens  # the last generated token has no KV yet
        req.past_kv.free()
        cache = KVCache(self.kv_pool)
        cache.rope_len = len(tokens)
        req.past_kv = cache
        self._reserve(cache, len(req.prompt_tokens) + req.max_new_tokens)
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        logits, req.past_kv = self._forward(ids, cache, [req.adapter], logits_to_keep=1)
        self._emit(req, self._sample([req], logits[:, -1])[0], time.time())

    def _emit(self, req: AsyncSequenceRequest, token: int, now: float) -> bool:
        """Deliver one sampled token; finish the request on a stop token or the length limit. True while it runs.
        A stop token ends the request without being emitted (as OpenAI-compatible servers do)."""
        if token in req.stop_token_ids:
            self._finish_request(req, now, "stop")
            return False
        if req.generated_tokens and req.last_token_at is not None and req.first_token_at != now:
            self.recent_itl.append(now - req.last_token_at)
        req.last_token_at = now
        req.generated_tokens.append(token)
        self.total_tokens_generated += 1
        req.output_queue.put_nowait(token)
        if len(req.generated_tokens) >= req.max_new_tokens:
            self._finish_request(req, now, "length")
            return False
        return True

    # ------------------------------------------------------------------ sampling

    def _probs(self, req: AsyncSequenceRequest, logits: torch.Tensor) -> Optional[torch.Tensor]:
        """Next-token distribution [V] after temperature, top-k and top-p; None for greedy requests."""
        if req.temperature <= 1e-5:
            return None
        x = logits.float() / req.temperature
        if 0 < req.top_k < x.shape[-1]:
            kth = torch.topk(x, req.top_k).values[-1]
            x = x.masked_fill(x < kth, float("-inf"))
        if req.top_p < 1.0:
            sorted_x, order = torch.sort(x, descending=True)
            cum = torch.softmax(sorted_x, -1).cumsum(-1)
            drop = cum - torch.softmax(sorted_x, -1) > req.top_p  # keep the smallest set reaching top_p
            x = x.scatter(-1, order, sorted_x.masked_fill(drop, float("-inf")))
        return torch.softmax(x, -1)

    def _processed(self, req: AsyncSequenceRequest, logits: torch.Tensor) -> torch.Tensor:
        if req.logits_processor is None:
            return logits
        out = req.logits_processor(req.generated_tokens, logits)
        return logits if out is None else out

    def _sample(self, reqs: List[AsyncSequenceRequest], logits: torch.Tensor) -> List[int]:
        """One token per row of logits [B, V], each with its request's processor and sampling settings (one host sync).
        Greedy rows without a processor share one argmax launch over the batch."""
        plain = [r.logits_processor is None and r.temperature <= 1e-5 for r in reqs]
        argmax = logits.argmax(-1) if any(plain) else None
        if all(plain):
            return argmax.tolist()
        rows = []
        for i, (r, row) in enumerate(zip(reqs, logits)):
            if plain[i]:
                rows.append(argmax[i])
                continue
            row = self._processed(r, row)
            p = self._probs(r, row)
            rows.append(row.argmax(-1) if p is None else torch.multinomial(p, 1)[0])
        return torch.stack(rows).tolist()

    # ------------------------------------------------------------------ speculative decoding (prompt lookup)

    def _drafts(self, req: AsyncSequenceRequest, n_decoding: int) -> List[int]:
        """Draft tokens for req from n-gram lookup in its own prompt and output (no draft model): the tokens that
        followed the most recent earlier occurrence of its last 3 (else 2, else 1) tokens. Off for requests with a
        logits processor, on the distributed path, and while the speculation gate says the batch is large enough."""
        k = req.draft_tokens if req.draft_tokens is not None else self.default_draft_tokens
        if not k or req.logits_processor is not None or self.dist_driver is not None:
            return []
        if self.spec_gate is not None and not self.spec_gate.should_speculate(n_decoding):
            return []
        k = min(k, req.max_new_tokens - len(req.generated_tokens) - 1)
        if k <= 0:
            return []
        seq = req.prompt_tokens + req.generated_tokens
        for n in (3, 2, 1):
            if len(seq) <= n:
                continue
            tail = seq[-n:]
            for start in range(len(seq) - n - 1, -1, -1):
                if seq[start:start + n] == tail:
                    return seq[start + n:start + n + k]
        return []

    def _speculative_step(self, req: AsyncSequenceRequest, drafts: List[int]) -> None:
        """Verify drafts in one forward: [last token] + drafts. Greedy requests accept a draft when it is the argmax;
        sampled requests accept draft d with probability p(d) and otherwise sample from p with d removed, which keeps
        the output distribution exactly that of plain sampling (the draft distribution is a point mass)."""
        cache = req.past_kv
        start = cache.length
        ids = torch.tensor([[req.generated_tokens[-1]] + drafts], dtype=torch.long, device=self.device)
        logits, req.past_kv = self._forward(ids, cache, [req.adapter])
        logits = logits[0]
        emitted = []
        for i, d in enumerate(drafts):
            p = self._probs(req, logits[i])
            if p is None:
                best = int(logits[i].argmax())
                emitted.append(best)
                if best != d:
                    break
            else:
                if float(torch.rand(())) < float(p[d]):
                    emitted.append(d)
                    continue
                p = p.clone()
                p[d] = 0
                emitted.append(int(torch.multinomial(p / p.sum(), 1)))
                break
        else:
            p = self._probs(req, logits[len(drafts)])
            emitted.append(int(logits[len(drafts)].argmax()) if p is None else int(torch.multinomial(p, 1)))
        # the cache keeps the positions of the inputs whose successor was accepted; the last emitted token is the next
        # step's input
        req.past_kv.length = start + len(emitted)
        self.spec_drafted += len(drafts)
        self.spec_accepted += len(emitted) - 1
        now = time.time()
        for t in emitted:
            if not self._emit(req, t, now):
                break

    def _finish_request(self, req: AsyncSequenceRequest, finished_time: float, reason: str = "length"):
        req.state = RequestState.FINISHED
        req.finished_at = finished_time
        req.finish_reason = req.finish_reason or reason
        if isinstance(req.past_kv, KVCache):
            if self.prefix_cache is not None and self.dist_driver is None and reason != "cancelled" \
                    and not req.uncacheable:
                self.prefix_cache.insert(req.prompt_tokens + req.generated_tokens, req.past_kv,
                                         self._prefix_key(req.adapter, req.past_kv.rope_len))
            req.past_kv.free()  # return its pages to the pool (cached pages keep the index's reference)
        req.past_kv = None
        if self.admission is not None:
            self.admission.release(req.request_id)
        req.output_queue.put_nowait(None)
        self.total_requests_completed += 1

    def get_telemetry(self) -> Dict[str, Any]:
        uptime = max(0.001, time.time() - self.start_time)
        throughput = self.total_tokens_generated / uptime

        def ms(vals, q=None):
            """Mean (q=None) or q-quantile in ms; None when there are too few samples to say (never another statistic)."""
            if len(vals) < (1 if q is None else 2):
                return None
            v = sum(vals) / len(vals) if q is None else statistics.quantiles(vals, n=100, method="inclusive")[q - 1]
            return round(v * 1000.0, 2)

        telemetry: Dict[str, Any] = {
            "uptime_seconds": round(uptime, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "total_requests_completed": self.total_requests_completed,
            "serving_throughput_tok_per_sec": round(throughput, 2),
            "running_requests": len(self.running_batch),
            "waiting_queue_depth": len(self.waiting_queue),
            "kv_memory_pool": self.kv_pool.get_stats(),
            "latency": {
                "avg_ttft_ms": ms(self.recent_ttft),
                "p50_ttft_ms": ms(self.recent_ttft, 50),
                "p95_ttft_ms": ms(self.recent_ttft, 95),
                "p99_ttft_ms": ms(self.recent_ttft, 99),
                "avg_itl_ms": ms(self.recent_itl),
                "p50_itl_ms": ms(self.recent_itl, 50),
                "p95_itl_ms": ms(self.recent_itl, 95),
                "p99_itl_ms": ms(self.recent_itl, 99),
            }
        }
        if self.admission is not None:
            telemetry["admission"] = self.admission.stats
        if self.spec_gate is not None:
            telemetry["spec_gate"] = self.spec_gate.stats
        telemetry["speculation"] = {"drafted_tokens": self.spec_drafted, "accepted_tokens": self.spec_accepted,
                                    "acceptance_rate": round(self.spec_accepted / self.spec_drafted, 4)
                                    if self.spec_drafted else None}
        if self.prefix_cache is not None:
            telemetry["prefix_cache"] = self.prefix_cache.stats()
        if self.lane_policy is not None:
            telemetry["lane_policy"] = {"slo_target_p99_ms": self.lane_policy.slo_target_p99_ms}

        return telemetry

    def get_kv_cache_utilization(self) -> float:
        """Returns KV cache memory pool utilization as a fraction (0.0 to 1.0)."""
        stats = self.kv_pool.get_stats()
        if stats["total_pages"] == 0:
            return 0.0
        return stats["used_pages"] / stats["total_pages"]

    def get_llmd_metrics(self) -> Dict[str, Any]:
        """
        Returns metric key-values structured for llm-d EPP router metric scrapers.
        """
        return {
            "num_requests_waiting": len(self.waiting_queue),
            "num_requests_running": len(self.running_batch),
            "kv_cache_usage_perc": self.get_kv_cache_utilization(),
            "block_size": self.kv_pool.page_size,
            "num_gpu_blocks": self.kv_pool.page_limit,
            "total_tokens_generated": self.total_tokens_generated,
        }


