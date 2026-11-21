"""
Continuous Batching Scheduler and Async Stateful Inference Serving Engine.
Supports chunked prefill, speculative interleaving, and Prometheus telemetry.
"""

import asyncio
import logging
import uuid
import statistics
import math
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional, AsyncGenerator, Deque, Any, Callable, Set
from collections import deque
import torch
import torch.nn.functional as F

from ..config import ModelConfig, TuringConfig
from ..models.causal_lm import DecodeGraphs, SubspaceCausalLM
from ..core.kv_cache import PAGE, KVCache, KVPoolExhausted, kv_bytes_per_token
from .traffic import (
    AdmissionController, AdmissionDecision,
    AdmissionResult, Lane, LanePolicy
)
from .spec_gate import SpecGateDecision, SpeculationGatePolicy
from .prefix_cache import PrefixCache

logger = logging.getLogger(__name__)

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
        self.error: Optional[BaseException] = None  # a scheduler step failed: stream_request raises it
        self.cached_prompt_tokens = 0  # prompt tokens served from the prefix cache
        self.uncacheable = False  # continues a caller's KVCache (prefix_kv): its pages are not keyed by its tokens
        self.needs_pages = False  # queued without pages (pool short, or preempted): takes them at the queue's head
        # what its cache is computed from: the prompt, or after a preemption the prompt and the output so far
        self.prefill_tokens = self.prompt_tokens
        self.tried = None  # (_frees, free pages) when it last failed to get pages: no retry until one of them moves
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
        # EAGLE-3 draft head state: its attention cache over verified positions, the position it expects next, and
        # the drafts it made after the last target forward
        self.head_cache = None
        self.head_next: Optional[int] = None
        self.head_drafts: List[int] = []
        # drafts the head makes next (None: the request's limit) and, decaying, how many of its drafts were accepted
        # of those the target got to check (_verified)
        self.head_len: Optional[int] = None
        self.head_hits = self.head_tries = 0.0
        # speculation that pays for itself: tokens per verify pass so far (a moving average), verify passes and steps
        # without a draft since drafting (re)started, and plain tokens left before drafting is tried again
        self.spec_tpp: Optional[float] = None
        self.spec_passes = self.spec_misses = self.spec_rest = 0
        self.ngrams: dict = {}  # prompt lookup: the last 1, 2 or 3 tokens -> position of what followed them
        self.ngram_seq: List[int] = []  # prompt + output indexed so far

def _bucket(rows: int) -> int:
    """The decode-graph batch bucket a step of `rows` rows runs in: steps of one bucket cost the same."""
    return next((b for b in DecodeGraphs.BUCKETS if b >= rows), rows)


class ContinuousBatchEngine:
    """
    High-Throughput Continuous Batching Engine managing iteration-level scheduling,
    chunked prefilling, and asynchronous streaming token generation with Static Paged KV Pooling.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        turing_config: Optional[TuringConfig] = None,
        prefill_chunk_size: int = 2048,  # prompt tokens prefilled per scheduler step, over all waiting requests
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
        draft_head: Optional[Any] = None,
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

        # an EAGLE-3 head (turing.models.eagle3) drafts instead of prompt lookup; the target then exposes the hidden
        # states the head reads. On the device before the KV pool is sized: the pool takes a share of what is free.
        self.draft_head = draft_head
        if draft_head is not None:
            if model_config.embed_proj_dim or model_config.embed_scale is not None or dist_driver is not None:
                raise ValueError("an EAGLE-3 draft head needs a target with plain token embeddings, served locally")
            aux = tuple(draft_head.aux_layers)
            if list(aux) != sorted(set(aux)) or not 1 <= aux[0] <= aux[-1] <= model_config.num_layers:
                raise ValueError(f"the draft head reads layers {aux}: they must be distinct, ascending and within "
                                 f"1..{model_config.num_layers}")
            self.draft_head = draft_head.to(self.device, self.model.embed_tokens.weight.dtype).eval()
            self.model.aux_layers = aux
        # The model's paged KV pool, sized once from free memory after the weights (ModelConfig.kv_pool_*)
        self.kv_pool = self.model.kv_pool()
        if draft_head is not None and self.device.type == "cuda":
            # the head's keys and values in pages of its own, for as many positions as the target's pool holds
            self.draft_head.use_pool(self.kv_pool.num_pages * self.kv_pool.page_size)
        self._rope_switch = self.model.rope_switch_length()  # LongRoPE's original context (Phi-3), else None
        self.prefix_cache = PrefixCache(self.kv_pool) if prefix_caching and dist_driver is None else None
        self.eos_token_ids = set(eos_token_ids) if eos_token_ids is not None else _eos_ids(tokenizer)
        self.default_draft_tokens = draft_tokens  # drafts per step when the gate allows (0 = off)
        self.spec_drafted = self.spec_accepted = 0
        self.remote_prefill = remote_prefill  # (host, port) of a PrefillWorker: prompts are prefilled there
        self.max_waiting = max_waiting  # requests that may wait for KV pages before new ones get 503

        # Overlapped decode (CUDA): a decode step is launched from the previous step's tokens while they are still on
        # the device, and the host reads and emits them while that step runs, so the GPU does not wait for the host
        # between steps. _ahead is the launched step whose tokens the host has not read: [requests, tokens on the
        # device, their host copy, the CUDA event that marks the copy].
        self.overlap_decode = self.device.type == "cuda"
        self._ahead: Optional[list] = None
        self.ride_prefill = True  # decoding requests take their token in a step's packed prefill forward (_riders)
        # Seconds per step (moving averages, _timed): a request keeps drafting only while its tokens per step exceed
        # what the step's extra rows cost against a plain step of the same requests (_speculation_pays). While
        # _spec_hold > 0 no request drafts: those plain steps time a batch size that has not been timed yet.
        self._step_s: Dict[Any, Optional[float]] = {}
        self._spec_hold = 0
        self._last_step: Optional[tuple] = None  # (its _step_s key, when it began) of the previous decode step
        self._decode_steps = 0
        self._held: Dict[int, int] = {}  # batch bucket -> _decode_steps when drafting was last held to time it
        self.preemptions = 0  # running requests whose pages were taken for senior ones (_preempt)
        self._frees = 0  # caches freed so far: a request waiting for pages tries again when this moves

        self.max_batch_size = self.turing_config.max_batch_size
        self.waiting_queue: List[AsyncSequenceRequest] = []
        self.running_batch: List[AsyncSequenceRequest] = []
        self.is_running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._wake = asyncio.Event()  # set by add_request: an idle scheduler starts on a request at once

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
            self._wake = asyncio.Event()  # a new one for this event loop (an Event stays with the loop that waited on it)
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
        """Admit a request: reserve the KV pages of its prompt (reusing cached prefix pages) and queue it; it takes
        the pages of its output as it decodes (_grow). Raises RequestRejected (400 never fits, 429 admission says
        wait, 503 overloaded). stop_token_ids defaults to the
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
            est_bytes = len(prompt_tokens) * kv_bytes_per_token(  # what it holds at once is decided as it decodes
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

        # A request continuing a caller's cache cannot be recomputed from its tokens, so it reserves every page it
        # can need and is never preempted. Any other request reserves its prompt's pages; if the pool is short now
        # it waits in the queue and reserves when pages free up.
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
        elif any(r.needs_pages for r in self.waiting_queue) or not self._try_reserve(req):
            # behind the requests already waiting for pages (a preempted one among them): first come, first served
            waiting = sum(1 for r in self.waiting_queue if r.needs_pages)
            if waiting >= self.max_waiting:
                reject(f"KV pool full and {waiting} requests already wait for pages", 503, 1.0)
            req.needs_pages = True

        self.waiting_queue.append(req)
        self._wake.set()
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

    GROW = 32  # tokens of KV pages a decoding request takes at a time

    def _try_reserve(self, req: AsyncSequenceRequest) -> bool:
        """Give req a cache with the pages of its prefill tokens (reusing cached prefix pages). False if the pool is
        short, or if that would leave the requests already holding pages, and this one, fewer than GROW tokens each
        to decode into."""
        tokens = req.prefill_tokens
        holders = len(self.running_batch) + sum(1 for r in self.waiting_queue if r.past_kv is not None)
        spare = -(-self.GROW // PAGE) * (holders + 1) if holders else 0
        if self.prefix_cache is not None:  # at least one token is prefilled
            req.past_kv = self.prefix_cache.attach(tokens, self._prefix_key(req.adapter, len(tokens)),
                                                   limit=len(tokens) - 1)
            req.prefill_offset = req.past_kv.length
            if tokens is req.prompt_tokens:
                req.cached_prompt_tokens = req.prefill_offset
        else:
            req.past_kv = KVCache(self.kv_pool)
        req.past_kv.rope_len = len(tokens)  # chunked prefill uses what one forward over the tokens would
        try:
            self._reserve(req.past_kv, len(tokens))
            free = self.kv_pool.free_pages
            if free < spare and self.prefix_cache is not None:
                free += self.prefix_cache.evict(spare - free)
            if free >= spare:
                return True
        except KVPoolExhausted:
            pass
        req.past_kv.free()
        req.past_kv, req.prefill_offset = None, 0
        if tokens is req.prompt_tokens:
            req.cached_prompt_tokens = 0
        req.tried = (self._frees, self.kv_pool.free_pages)
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

    def _seniority(self, req: AsyncSequenceRequest):
        """Sorts requests by who keeps its pages when the pool is short: the more urgent lane, then the older."""
        lane = self.lane_policy.priority(req.lane) if self.lane_policy is not None and req.lane else 0
        return lane, req.created_at

    def _need(self, r: AsyncSequenceRequest) -> int:
        """Tokens r's cache must hold after its next decode step: its next position and room for its drafts."""
        return r.past_kv.length + 1 + min(self._draft_count(r), max(r.max_new_tokens - len(r.generated_tokens) - 1, 0))

    def _top_up(self, decode_kv: List[AsyncSequenceRequest]) -> bool:
        """_grow without preemption: False when the pool ran out before every request had its pages."""
        if all(len(r.past_kv.pages) * PAGE >= self._need(r) for r in decode_kv):
            return True
        try:
            for r in decode_kv:  # never past what the request can still use
                limit = r.past_kv.length + r.max_new_tokens - len(r.generated_tokens)
                self._reserve(r.past_kv, max(self._need(r), min(self._need(r) + self.GROW, limit)))
            return True
        except KVPoolExhausted:
            return False

    def _grow(self, decode: List[AsyncSequenceRequest]) -> List[AsyncSequenceRequest]:
        """Pages for this decode step: every request needs its next position and room for its drafts. When one has
        run out, all of them are topped up by GROW tokens (the page-table writes of many requests then fall in one
        step). When the pool is short, the least senior requests are preempted until the others have their pages.
        Returns the requests that still decode."""
        need = self._need
        decode_kv = [r for r in decode if isinstance(r.past_kv, KVCache)]
        if self._top_up(decode_kv):
            return decode
        self._settle()  # a preempted request needs every token it has generated: read the step in flight first
        for r in sorted(decode_kv, key=self._seniority):
            while r.state == RequestState.DECODING:
                try:
                    self._reserve(r.past_kv, need(r))
                    break
                except KVPoolExhausted:
                    victims = [v for v in self.running_batch + self.waiting_queue  # queued with their prompt's pages too
                               if isinstance(v.past_kv, KVCache) and not v.uncacheable]
                    if not victims:
                        raise
                    self._preempt(max(victims, key=self._seniority))
        return [r for r in decode if r.state == RequestState.DECODING]

    def _preempt(self, req: AsyncSequenceRequest, count: bool = True) -> None:
        """Take req's pages (for requests senior to it). A queued request keeps its place and reserves again later. A
        running one goes back to the head of the queue, and once pages are free its cache is recomputed from its
        prompt and the tokens it has generated, through the prefill path (recompute preemption, as vLLM's)."""
        req.past_kv.free()
        req.past_kv, req.prefill_offset, req.needs_pages = None, 0, True
        self._head_reset(req)
        self._frees += 1
        if req.state == RequestState.WAITING:
            return
        if req.generated_tokens:
            req.prefill_tokens = req.prompt_tokens + req.generated_tokens
        req.state = RequestState.WAITING
        self.running_batch.remove(req)
        self.waiting_queue.insert(0, req)
        self.preemptions += count

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
                    if req.error is not None:
                        raise RuntimeError(f"request {req.request_id} failed in the engine: {req.error!r}") from req.error
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
            i = 0  # the requests before i wait for pages; behind them only requests that hold theirs may start
            while len(self.running_batch) < self.max_batch_size and i < len(self.waiting_queue):
                req = self.waiting_queue[i]
                if getattr(req, "cancelled", False):
                    self.waiting_queue.pop(i)
                    self._finish_request(req, time.time(), "cancelled")
                    continue
                if req.needs_pages:  # first come, first served among the requests that need pages
                    if i or req.tried == (self._frees, self.kv_pool.free_pages) or not self._try_reserve(req):
                        i += 1
                        continue
                    req.needs_pages = False
                self.waiting_queue.pop(i)
                req.state = RequestState.PREFILLING
                self.running_batch.append(req)
            if not self.running_batch:
                if self.waiting_queue:  # they wait for pages (or for a cancel to be seen): look again shortly
                    await asyncio.sleep(0.001)
                else:
                    self._wake.clear()
                    await self._wake.wait()
                continue
            try:
                self._step_batch()
            except Exception as e:  # fail this step's requests; a dead loop would leave every caller waiting forever
                logger.exception("scheduler step failed; finishing %d requests with an error", len(self.running_batch))
                now = time.time()
                for req in self.running_batch:
                    if req.state != RequestState.FINISHED:
                        req.error, req.uncacheable = e, True  # its cache may be half-written: never index it
                        self._finish_request(req, now, "error")
                self.running_batch = []
                self._ahead = None
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
            budget, work = None, []
            for req in prefill:
                if budget is None:
                    budget = self.lane_policy.prefill_chunk_budget(req.lane) if (self.lane_policy and req.lane) \
                        else self.prefill_chunk_size
                if budget <= 0:
                    break
                end = min(req.prefill_offset + budget, len(req.prefill_tokens))
                budget -= end - req.prefill_offset
                work.append((req, end))
            caches = [r.past_kv for r, _ in work]
            packs = (bool(work) and self.dist_driver is None and not any(r.adapter for r, _ in work)
                     and all(isinstance(c, KVCache) for c in caches)
                     and getattr(self.model, "can_pack_prefill", lambda _: False)(caches))
            ride_plain, ride_spec, n_riding = self._riders(work) if packs else ([], [], 0)
            riders, ride_starts = self._rows(ride_plain, ride_spec)
            if packs and len(work) + len(riders) > 1:
                # every chunk of this step in one forward (one set of GEMMs), attention per request
                last = self.model.prefill_packed([r.prefill_tokens[r.prefill_offset:e] for r, e in work]
                                                 + [[t] for _, t in riders],
                                                 caches + [r.past_kv for r, _ in riders], riders=len(riders))
                aux = self.model.last_aux[0].split([e - r.prefill_offset for r, e in work]) \
                    if self.draft_head is not None else [None] * len(work)
                done = [(r, e, last[i:i + 1], aux[i]) for i, (r, e) in enumerate(work)]
            else:
                done = []
                for req, end in work:
                    chunk = torch.tensor([req.prefill_tokens[req.prefill_offset:end]], dtype=torch.long,
                                         device=self.device)
                    logits, req.past_kv = self._forward(chunk, req.past_kv, [req.adapter], logits_to_keep=1)
                    done.append((req, end, logits[:, -1], self.model.last_aux[0] if self.draft_head is not None else None))
            head_work = []  # requests whose prompt ended in this step: the draft head takes them in one pass
            for req, end, last_logits, aux in done:
                start, req.prefill_offset = req.prefill_offset, end
                # the chunk's first position in the cache (not its offset: a continued cache starts later)
                pos = req.past_kv.length - (end - start) if aux is not None else None
                if aux is not None and end < len(req.prefill_tokens):  # not the last chunk: every next token is known
                    self._head_extend(req, aux, req.prefill_tokens[start + 1:end + 1], pos)
                if end >= len(req.prefill_tokens):
                    req.state = RequestState.DECODING
                    if self.prefix_cache is not None and self.dist_driver is None and not req.uncacheable:
                        self.prefix_cache.insert(req.prefill_tokens, req.past_kv,
                                                 self._prefix_key(req.adapter, req.past_kv.rope_len))
                    token = self._sample([req], last_logits)[0]
                    now = time.time()  # after the forward and the sample (which syncs), so TTFT includes prefill
                    if req.first_token_at is None:  # else it resumes after a preemption: this is its next token
                        req.first_token_at = req.last_token_at = now
                        self.recent_ttft.append(now - req.created_at)
                    if self._emit(req, token, now) and aux is not None:
                        head_work.append((req, aux, req.prefill_tokens[start + 1:end] + [token], pos))
            if head_work:
                self._head_draft(head_work)
            if riders:
                self._decoded(ride_plain, ride_spec, riders, ride_starts, last[len(work):], n_riding)

            # Decode: speculative rows one by one, the rest in one batched forward (bandwidth-bound)
            # a request whose first token came from this step's prefill decodes from the next step: its consumer gets
            # that token now instead of after a decode forward (one step time of TTFT, P4.1)
            first = {id(r) for r, e, _, _ in done if e >= len(r.prefill_tokens)} | {id(r) for r, _ in riders}
            decode = [r for r in self.running_batch if r.state == RequestState.DECODING and r.first_token_at is not None
                      and r.generated_tokens and id(r) not in first]
            decode = self._grow(decode)
            now = time.perf_counter()
            if self._last_step is not None and decode and not work:  # a whole step: forward, host work, event loop
                self._timed(self._last_step[0], now - self._last_step[1])
            self._last_step = None
            for r in decode:
                if self._crosses_rope_switch(r, 1):
                    # about to pass LongRoPE's original context: as transformers does for Phi-3
                    # (Phi3ForCausalLM.prepare_inputs_for_generation), the whole cache is recomputed with the long
                    # frequencies, here through the prefill path from the head of the queue
                    self._preempt(r, count=False)
            decode = [r for r in decode if r.state == RequestState.DECODING]
            n_decoding = rows = len(decode)
            if self._ahead is not None or (decode and all(self._overlaps(r, len(decode)) for r in decode)):
                pending = [r for r in self._ahead[0] if r.state != RequestState.FINISHED] if self._ahead else decode
                if pending == decode and all(self._overlaps(r, len(decode)) for r in decode):
                    self._decode_ahead(decode)
                    decode = []
                else:  # the set of requests changed, or one of them now needs the host between steps
                    self._settle()
                    decode = [r for r in decode if r.state == RequestState.DECODING]
            in_step, serial, plain = [], [], []
            room = max(DecodeGraphs.BUCKETS[-1] - len(decode), 0)  # draft rows the largest decode graph still takes
            for r in decode:
                drafts = self._drafts(r, len(decode))
                if drafts and self._crosses_rope_switch(r, 1 + len(drafts)):
                    drafts = None  # verify only up to the switch; the step that crosses it recomputes the cache
                if drafts is not None and self._verify_in_step(r):
                    drafts = drafts[:room]
                    room -= len(drafts)
                    if drafts or self.draft_head is not None:  # a head restarts from a row without drafts
                        in_step.append((r, drafts))
                        rows += len(drafts)
                        continue
                    drafts = None
                if drafts is None:
                    plain.append(r)
                else:
                    serial.append((r, drafts))
                    rows += len(drafts)
            for r, drafts in serial:
                self._speculative_step(r, drafts, rows, n_decoding)
            if in_step or (len(plain) > 1 and self.dist_driver is None):
                self._decode_rows(plain, in_step, n_decoding)
            else:
                for r in plain:
                    ids = torch.tensor([[r.generated_tokens[-1]]], dtype=torch.long, device=self.device)
                    logits, r.past_kv = self._forward(ids, r.past_kv, [r.adapter])
                    self._emit(r, self._sample([r], logits[:, -1])[0], time.time())
            if n_decoding or riders:
                self._decode_steps += 1
                self._spec_hold = max(self._spec_hold - 1, 0)
                if not work:  # a prefill in the same step would be timed with it
                    self._last_step = (("drafts", _bucket(rows)) if in_step or serial else _bucket(rows), now)

        self.running_batch = [r for r in self.running_batch if r.state != RequestState.FINISHED]

    def _riders(self, work: list) -> tuple:
        """The decoding requests that take their next tokens from this step's packed prefill forward, as rows of one
        token after work's prompt chunks, instead of from a decode forward of their own: one pass over the weights
        serves both (vLLM schedules a step the same way), which a burst of arrivals otherwise pays for in every step.
        Returns (plain requests, (request, drafts) pairs, requests decoding), as _decode_rows takes them: drafts are
        verified as rows there too. Not the requests the pool has no pages for yet (they decode after the prefill,
        where _grow may preempt), and none with Quest page selection (SubspaceCausalLM.can_ride_prefill), LongRoPE's
        switch to watch, or a draft head, which reads every position's states."""
        can = getattr(self.model, "can_ride_prefill", None)
        if not self.ride_prefill or can is None or self.draft_head is not None or self._rope_switch is not None:
            return [], [], 0
        decode = [r for r in self.running_batch if r.state == RequestState.DECODING and r.first_token_at is not None
                  and r.generated_tokens and r.adapter is None and isinstance(r.past_kv, KVCache)]
        if not decode or not can([r.past_kv for r, _ in work] + [r.past_kv for r in decode]):
            return [], [], 0
        self._settle()  # the step launched ahead: its tokens are these rows' inputs
        decode = [r for r in decode if r.state == RequestState.DECODING]
        self._top_up(decode)
        plain, spec = [], []
        for r in decode:
            drafts = self._drafts(r, len(decode))
            if len(r.past_kv.pages) * PAGE < r.past_kv.length + 1 + len(drafts or ()):
                continue
            if drafts:
                spec.append((r, drafts))
            else:
                plain.append(r)
        return plain, spec, len(decode)

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

    def _emit(self, req: AsyncSequenceRequest, token: int, now: float) -> bool:
        """Deliver one sampled token; finish the request on a stop token or the length limit. True while it runs.
        A stop token ends the request without being emitted (as OpenAI-compatible servers do)."""
        if token in req.stop_token_ids:
            self._finish_request(req, now, "stop")
            return False
        if req.generated_tokens and req.last_token_at is not None and req.first_token_at != now:
            self.recent_itl.append(now - req.last_token_at)
        req.last_token_at = now
        if req.spec_rest:
            req.spec_rest -= 1
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

    def _sample_tokens(self, reqs: List[AsyncSequenceRequest], logits: torch.Tensor) -> torch.Tensor:
        """One token per row of logits [B, V], each with its request's processor and sampling settings, on the
        device. Greedy rows without a processor share one argmax launch over the batch."""
        plain = [r.logits_processor is None and r.temperature <= 1e-5 for r in reqs]
        argmax = logits.argmax(-1) if any(plain) else None
        if all(plain):
            return argmax
        rows = []
        for i, (r, row) in enumerate(zip(reqs, logits)):
            if plain[i]:
                rows.append(argmax[i])
                continue
            row = self._processed(r, row)
            p = self._probs(r, row)
            rows.append(row.argmax(-1) if p is None else torch.multinomial(p, 1)[0])
        return torch.stack(rows)

    def _sample(self, reqs: List[AsyncSequenceRequest], logits: torch.Tensor) -> List[int]:
        """_sample_tokens on the host (one sync)."""
        return self._sample_tokens(reqs, logits).tolist()

    # ------------------------------------------------------------------ overlapped decode

    def _overlaps(self, req: AsyncSequenceRequest, n_decoding: int) -> bool:
        """req's next decode step can be launched before the host has read its previous token: nothing on the host
        looks at that token first (no logits processor, no drafting this step, no LongRoPE switch to watch)."""
        if not self.overlap_decode or self.dist_driver is not None or self._rope_switch is not None \
                or req.logits_processor is not None:
            return False
        return not self._may_draft(req, n_decoding)

    def _launch(self, reqs: List[AsyncSequenceRequest], ids: torch.Tensor) -> list:
        """One decode forward for reqs from ids [B, 1] on the device; its tokens are sampled there and copied to
        pinned host memory without waiting. Returns the _ahead entry."""
        logits, _ = self._forward(ids, [r.past_kv for r in reqs], [r.adapter for r in reqs])
        tokens = self._sample_tokens(reqs, logits[:, -1])
        if tokens.is_cuda:
            host = torch.empty(tokens.shape, dtype=tokens.dtype, pin_memory=True)
            host.copy_(tokens, non_blocking=True)
            event = torch.cuda.Event()
            event.record()  # reached when this step and the copy are done, whatever is queued after them
        else:
            host, event = tokens, None
        return [list(reqs), tokens, host, event]

    def _settle(self) -> None:
        """Read the tokens of the step launched ahead (waiting for that step only) and emit them."""
        ahead, self._ahead = self._ahead, None
        if ahead is None:
            return
        reqs, _, host, event = ahead
        if event is not None:
            event.synchronize()
        now = time.time()
        for r, t in zip(reqs, host.tolist()):
            if r.state != RequestState.FINISHED:  # cancelled since the launch
                self._emit(r, t, now)

    def _decode_ahead(self, decode: List[AsyncSequenceRequest]) -> None:
        """One overlapped step for decode (the unfinished requests of _ahead, or the first step of a run): launch
        the next forward from the tokens still on the device, then read and emit those tokens while it runs. A
        request that turns out to have stopped on such a token has one forward computed for nothing; its row of
        the step ahead is dropped."""
        if self._ahead is None:
            ids = torch.tensor([[r.generated_tokens[-1]] for r in decode], dtype=torch.long, device=self.device)
            self._ahead = self._launch(decode, ids)
            return
        reqs, tokens = self._ahead[0], self._ahead[1]
        # rows that take another token after the one in flight
        more = [i for i, r in enumerate(reqs)
                if r.state != RequestState.FINISHED and len(r.generated_tokens) + 1 < r.max_new_tokens]
        # every row goes on in most steps: no index then (one from the host is copied with a wait for the step in
        # flight, which is the overlap lost)
        ids = tokens if len(more) == len(reqs) else tokens[more]
        nxt = self._launch([reqs[i] for i in more], ids[:, None]) if more else None
        self._settle()
        self._ahead = nxt if nxt is not None and any(r.state != RequestState.FINISHED for r in nxt[0]) else None

    # ------------------------------------------------------------------ speculative decoding (prompt lookup)

    def _draft_count(self, req: AsyncSequenceRequest) -> int:
        return req.draft_tokens if req.draft_tokens is not None else self.default_draft_tokens

    def _head_reset(self, req: AsyncSequenceRequest) -> None:
        """Drop what the draft head holds for req (the pages of its cache go back to the head's pool)."""
        if isinstance(req.head_cache, KVCache):
            req.head_cache.free()
        req.head_cache, req.head_next, req.head_drafts = None, None, []

    def _head_lost(self, cache) -> bool:
        """The head's pool had no room for this request's positions: the head starts again at its next step."""
        return cache is None and self.draft_head.pool is not None

    def _head_seen(self, req: AsyncSequenceRequest, start: int):
        """The head's cache for req if it ends where position `start` begins; else the head starts again."""
        if req.head_next != start:
            self._head_reset(req)
        return req.head_cache

    def _head_extend(self, req: AsyncSequenceRequest, aux: torch.Tensor, next_tokens: List[int], start: int) -> None:
        """Give the draft head the verified positions start .. start + len(aux) - 1 (aux: the target's tapped hidden
        states there; next_tokens: the tokens that follow them). A gap since the positions it last saw resets it."""
        item = (aux, torch.tensor(next_tokens, device=self.device), start, 0, self._head_seen(req, start))
        req.head_cache = self.draft_head.draft_many(self.model.embed_tokens, [item])[0][1]
        req.head_next = None if self._head_lost(req.head_cache) else start + len(next_tokens)

    def _head_draft(self, work: list) -> None:
        """work: (request, aux, next_tokens, start) for every request whose target forward just verified positions
        (as _head_extend; the last of next_tokens is the token just sampled). The head takes them in and drafts each
        request's next tokens, all requests in one pass over its weights per draft position."""
        items, reqs = [], []
        for req, aux, next_tokens, start in work:
            k = min(self._draft_count(req), req.head_len or self._draft_count(req))
            if k and req.adapter is None and req.logits_processor is None:
                items.append((aux, next_tokens, start, k, self._head_seen(req, start)))
                reqs.append(req)
        if not items:
            return
        tokens = torch.tensor([t for item in items for t in item[1]], device=self.device)
        tokens = tokens.split([len(item[1]) for item in items])
        items = [(aux, toks, start, k, cache) for (aux, _, start, k, cache), toks in zip(items, tokens)]
        drafted = self.draft_head.draft_many(self.model.embed_tokens, items)
        for req, (_, toks, start, _, _), (drafts, cache) in zip(reqs, items, drafted):
            req.head_drafts, req.head_cache = drafts, cache
            req.head_next = None if self._head_lost(cache) else start + len(toks)

    # verify passes before judging (and the span of the moving average); plain tokens of rest; steps with no draft
    SPEC_PROBE, SPEC_REST, SPEC_MISSES = 16, 64, 8
    # a draft head's chain stops where the chance of the target reaching that draft falls below this: a draft costs a
    # forward of the head (about 4 ms against a 62 ms target step on Qwen3-8B) whether or not it is reached
    HEAD_REACH = 0.15

    def _may_draft(self, req: AsyncSequenceRequest, n_decoding: int) -> bool:
        """req may be decoded speculatively this step: drafting is on for it, nothing needs every token on the host
        first, the gate allows it at this batch size, and it is not resting after drafts that did not pay."""
        if not self._draft_count(req) or req.logits_processor is not None or self.dist_driver is not None \
                or req.spec_rest > 0 or self._spec_hold > 0 \
                or (self.draft_head is not None and req.adapter is not None):
            return False
        # the gate is for verify forwards that run one request at a time; drafts verified in the step need none
        return self.spec_gate is None or self._verify_in_step(req) or self.spec_gate.should_speculate(n_decoding)

    def _lookup(self, req: AsyncSequenceRequest, k: int) -> Optional[List[int]]:
        """Up to k tokens that followed the most recent earlier occurrence of req's last 3 (else 2, else 1) tokens in
        its own prompt and output, from an index extended by the tokens added since the last call."""
        seq, index = req.ngram_seq, req.ngrams
        done = max(len(seq) - 1, 0)  # n-grams ending before position `done` are indexed
        if not seq:
            seq.extend(req.prompt_tokens)
        seq.extend(req.generated_tokens[len(seq) - len(req.prompt_tokens):])
        for j in range(done, len(seq) - 1):  # an n-gram ending at j is indexed once the token after it exists
            for n in (1, 2, 3):
                if j + 1 >= n:
                    index[tuple(seq[j - n + 1:j + 1])] = j + 1
        for n in (3, 2, 1):
            if len(seq) > n:
                pos = index.get(tuple(seq[-n:]))
                if pos is not None:
                    return seq[pos:pos + k]
        return None

    def _drafts(self, req: AsyncSequenceRequest, n_decoding: int) -> Optional[List[int]]:
        """Draft tokens for req, or None to decode it without speculation. With an EAGLE-3 head they are the head's
        drafts from the last target forward (an empty list when it has none: the verify forward then restarts it).
        Otherwise prompt lookup (_lookup). None while _may_draft says no; a request for which no draft is found
        SPEC_MISSES steps in a row rests like one whose drafts did not pay (_speculation_pays)."""
        if not self._may_draft(req, n_decoding):
            req.head_drafts = []  # the head misses the positions decoded without it
            return None
        k = min(self._draft_count(req), req.max_new_tokens - len(req.generated_tokens) - 1)
        if k <= 0:
            return None
        if self.draft_head is not None:  # its drafts continue position head_next: stale once plain steps have passed
            drafts = req.head_drafts[:k] if req.head_next == req.past_kv.length else []
            req.head_drafts = []
            return drafts
        drafts = self._lookup(req, k)
        if drafts is None:
            req.spec_misses += 1
            if req.spec_misses >= self.SPEC_MISSES:
                req.spec_rest, req.spec_misses = self.SPEC_REST, 0
        else:
            req.spec_misses = 0
        return drafts

    SPEC_RETIME = 256  # decode steps before drafting is held again to time a batch size that is still untimed

    def _timed(self, key, seconds: float) -> None:
        """Moving average of a decode step's seconds in _step_s, from one step's start to the next (the forward, the
        host's work and the event loop's turn; steps next to a prefill are left out). key: the step's row bucket, or
        ("drafts", row bucket) for a step that verified drafts, which also covers a draft head's time. A key's first
        sample is dropped (it may include a graph capture or a kernel compilation)."""
        if key not in self._step_s:
            self._step_s[key] = None
        else:
            prev = self._step_s[key]
            self._step_s[key] = seconds if prev is None else 0.8 * prev + 0.2 * seconds

    def _speculation_pays(self, req: AsyncSequenceRequest, emitted: int, rows: int, n: int) -> None:
        """After a step of `rows` rows that verified req's drafts and emitted `emitted` of its tokens, with n requests
        decoding. Once SPEC_PROBE such steps have been seen, a request whose tokens per step are no more than the
        seconds of such a step over those of a plain step of n rows rests for SPEC_REST tokens, then drafts again.
        While no plain step of that size has been timed, every request decodes a few tokens plainly (at most once in
        SPEC_RETIME steps)."""
        req.spec_passes += 1
        # the mean of the first SPEC_PROBE passes, then a moving average over about as many: accepted drafts come in
        # runs, and a few passes without one say little (resting on them cost chat a third of its drafting passes)
        req.spec_tpp = float(emitted) if req.spec_tpp is None else \
            req.spec_tpp + (emitted - req.spec_tpp) / min(req.spec_passes, self.SPEC_PROBE)
        cost, plain = self._step_s.get(("drafts", _bucket(rows))), self._step_s.get(_bucket(n))
        if req.spec_passes < self.SPEC_PROBE or cost is None:
            return
        if plain is None:
            if self._decode_steps - self._held.get(_bucket(n), -self.SPEC_RETIME) >= self.SPEC_RETIME:
                self._held[_bucket(n)], self._spec_hold = self._decode_steps, 4
        elif req.spec_tpp <= cost / plain:
            req.spec_rest, req.spec_passes, req.spec_tpp = self.SPEC_REST, 0, None

    def _speculative_step(self, req: AsyncSequenceRequest, drafts: List[int], rows: int, n: int) -> None:
        """Verify drafts in one forward: [last token] + drafts. Greedy requests accept a draft when it is the argmax;
        sampled requests accept draft d with probability p(d) and otherwise sample from p with d removed, which keeps
        the output distribution exactly that of plain sampling (the draft distribution is a point mass)."""
        cache = req.past_kv
        start = cache.length
        ids = torch.tensor([[req.generated_tokens[-1]] + drafts], dtype=torch.long, device=self.device)
        logits, req.past_kv = self._forward(ids, cache, [req.adapter])
        emitted = self._accept(req, drafts, logits[0])  # last_aux is [1, T, .] here, or [1, 1, .] without drafts
        aux = self.model.last_aux.flatten(0, 1)[:len(emitted)] if self.draft_head is not None else None
        work = self._verified(req, drafts, emitted, start, aux, rows, n)
        if work is not None:
            self._head_draft([work])

    def _accept(self, req: AsyncSequenceRequest, drafts: List[int], logits: torch.Tensor,
                tops: Optional[List[int]] = None) -> List[int]:
        """The tokens to emit after verifying drafts: logits [len(drafts) + 1, V] are the target's next-token logits
        after the request's last token and after each draft (tops: their argmax, if the caller has read it)."""
        greedy = None if req.temperature > 1e-5 else tops if tops is not None else logits.argmax(-1).tolist()
        emitted = []
        for i, d in enumerate(drafts):
            p = None if greedy is not None else self._probs(req, logits[i])
            if p is None:
                best = greedy[i]
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
            p = None if greedy is not None else self._probs(req, logits[len(drafts)])
            emitted.append(greedy[len(drafts)] if p is None else int(torch.multinomial(p, 1)))
        return emitted

    def _verified(self, req: AsyncSequenceRequest, drafts: List[int], emitted: List[int], start: int,
                  aux: Optional[torch.Tensor], rows: int, n: int) -> Optional[tuple]:
        """Bookkeeping after a verify: the cache keeps the positions of the inputs whose successor was accepted (the
        last emitted token is the next step's input), the tokens are emitted, and the drafting rule is updated.
        Returns the draft head's work for this request (_head_draft), or None."""
        req.past_kv.length = start + len(emitted)
        self.spec_drafted += len(drafts)
        self.spec_accepted += len(emitted) - 1
        now = time.time()
        for t in emitted:
            if not self._emit(req, t, now):
                return None
        if drafts:  # a row without drafts (a head starting again) is not a drafting step to judge
            self._speculation_pays(req, len(emitted), rows, n)
        if self.draft_head is None:
            return None
        if drafts:  # the next chain's length, from the share p of checked drafts that were accepted lately: the
            # target reaches the j-th draft with probability about p^(j-1) * p, so draft while p^j >= HEAD_REACH
            req.head_hits = 0.9 * req.head_hits + len(emitted) - 1
            req.head_tries = 0.9 * req.head_tries + min(len(emitted), len(drafts))  # the accepted and the first refused
            p = req.head_hits / req.head_tries
            req.head_len = self._draft_count(req) if p >= 1 else \
                max(1, math.ceil(math.log(self.HEAD_REACH) / math.log(p))) if p > 0 else 1
        # the inputs whose successors were emitted are verified positions: aux holds the target's states there
        return (req, aux, emitted, start)

    def _verify_in_step(self, req: AsyncSequenceRequest) -> bool:
        """req's drafts can be verified as rows of the batched decode step (SubspaceCausalLM row_starts): its last
        token and each draft take consecutive positions of its cache in one step, which costs about what a plain
        step costs. Not on the distributed path, with LongRoPE's switch, or with Quest page tracking."""
        return (self.dist_driver is None and self._rope_switch is None
                and isinstance(req.past_kv, KVCache) and not req.past_kv.pool.track_pages)

    @staticmethod
    def _rows(plain: List[AsyncSequenceRequest], spec: list) -> tuple:
        """The rows of one batched decode step, [(request, input token)]: one for each plain request, and for each
        (request, drafts) of spec one for its last token and one for every draft; and each spec request's cache
        length before the step."""
        rows = [(r, r.generated_tokens[-1]) for r in plain]
        starts = []
        for r, drafts in spec:
            starts.append(r.past_kv.length)
            rows += [(r, t) for t in [r.generated_tokens[-1]] + drafts]
        return rows, starts

    def _decode_rows(self, plain: List[AsyncSequenceRequest], spec: list, n: int) -> None:
        """One batched decode step for _rows(plain, spec). n: the requests decoding in this scheduler step."""
        rows, starts = self._rows(plain, spec)
        ids = torch.tensor([[t] for _, t in rows], dtype=torch.long, device=self.device)
        logits, _ = self._forward(ids, [r.past_kv for r, _ in rows], [r.adapter for r, _ in rows])
        self._decoded(plain, spec, rows, starts, logits[:, -1], n)

    def _decoded(self, plain: List[AsyncSequenceRequest], spec: list, rows: list, starts: List[int],
                 logits: torch.Tensor, n: int) -> None:
        """Sample, verify and emit after the forward that gave `rows` their logits [len(rows), V] (a decode step, or
        a packed prefill they rode in)."""
        aux = self.model.last_aux[:, 0] if self.draft_head is not None else None  # each row's tapped states
        tops = None
        if all(r.temperature <= 1e-5 and r.logits_processor is None for r, _ in rows):
            tops = logits.argmax(-1).tolist()  # greedy rows only: one host read for the whole step
        tokens = tops[:len(plain)] if tops is not None else self._sample(plain, logits[:len(plain)]) if plain else []
        at, verified = len(plain), []
        for (r, drafts), start in zip(spec, starts):
            emitted = self._accept(r, drafts, logits[at:at + len(drafts) + 1],
                                   tops[at:at + len(drafts) + 1] if tops is not None else None)
            verified.append((r, drafts, emitted, start, aux[at:at + len(emitted)] if aux is not None else None))
            at += len(drafts) + 1
        now = time.time()
        for r, t in zip(plain, tokens):
            self._emit(r, t, now)
        work = [self._verified(r, drafts, emitted, start, a, len(rows), n) for r, drafts, emitted, start, a in verified]
        work = [w for w in work if w is not None]
        if work:
            self._head_draft(work)  # every request's next drafts in one pass of the head

    def _finish_request(self, req: AsyncSequenceRequest, finished_time: float, reason: str = "length"):
        req.state = RequestState.FINISHED
        self._head_reset(req)
        req.finished_at = finished_time
        req.finish_reason = req.finish_reason or reason
        if isinstance(req.past_kv, KVCache):
            if self.prefix_cache is not None and self.dist_driver is None and reason != "cancelled" \
                    and not req.uncacheable:
                self.prefix_cache.insert(req.prompt_tokens + req.generated_tokens, req.past_kv,
                                         self._prefix_key(req.adapter, req.past_kv.rope_len))
            req.past_kv.free()  # return its pages to the pool (cached pages keep the index's reference)
            self._frees += 1
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
            "preemptions": self.preemptions,
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
        offload = getattr(self.model, "expert_offload_stats", lambda: None)()
        if offload is not None:
            telemetry["moe_offload"] = offload
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


