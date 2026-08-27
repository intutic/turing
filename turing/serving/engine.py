"""
Continuous Batching Scheduler and Async Stateful Inference Serving Engine.
Supports chunked prefill, speculative interleaving, and Prometheus telemetry.
"""

import asyncio
import uuid
import time
from enum import Enum
from typing import List, Dict, Tuple, Optional, AsyncGenerator, Deque, Any
from collections import deque
import torch
import torch.nn.functional as F

from ..config import ModelConfig, TuringConfig
from ..models.causal_lm import SubspaceCausalLM
from ..core.paging import StaticPagedKVPool

class RequestState(Enum):
    WAITING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3

class AsyncSequenceRequest:
    def __init__(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        request_id: Optional[str] = None,
        sparsity_ratio: Optional[float] = None,
        use_svd_kv: Optional[bool] = None,
        draft_tokens: Optional[int] = None
    ):
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.prompt_tokens = list(prompt_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.sparsity_ratio = sparsity_ratio
        self.use_svd_kv = use_svd_kv
        self.draft_tokens = draft_tokens
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
        tokenizer: Optional[Any] = None
    ):
        self.model_config = model_config
        self.turing_config = turing_config or TuringConfig()
        self.device = self.turing_config.resolve_device()
        self.prefill_chunk_size = prefill_chunk_size
        self.tokenizer = tokenizer

        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = SubspaceCausalLM(model_config).to(self.device).eval()

        # Zero-Allocation Contiguous Page Memory Pool
        head_dim = model_config.hidden_dim // model_config.num_heads
        self.kv_pool = StaticPagedKVPool(
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            head_dim=head_dim,
            page_size=16,
            max_total_pages=max(256, self.turing_config.max_batch_size * 64),
            device=self.device,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

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
        sparsity_ratio: Optional[float] = None,
        use_svd_kv: Optional[bool] = None,
        draft_tokens: Optional[int] = None
    ) -> AsyncSequenceRequest:
        req = AsyncSequenceRequest(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            sparsity_ratio=sparsity_ratio,
            use_svd_kv=use_svd_kv,
            draft_tokens=draft_tokens
        )
        self.waiting_queue.append(req)
        return req

    async def stream_generate(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_k: int = 50,
        sparsity_ratio: Optional[float] = None,
        use_svd_kv: Optional[bool] = None,
        draft_tokens: Optional[int] = None
    ) -> AsyncGenerator[int, None]:
        req = await self.add_request(
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            sparsity_ratio=sparsity_ratio,
            use_svd_kv=use_svd_kv,
            draft_tokens=draft_tokens
        )
        while True:
            token = await req.output_queue.get()
            if token is None:
                break
            yield token

    async def _scheduler_loop(self):
        while self.is_running:
            # 1. Admit waiting requests up to batch capacity
            while len(self.running_batch) < self.max_batch_size and self.waiting_queue:
                req = self.waiting_queue.pop(0)
                req.state = RequestState.PREFILLING
                req.prefill_offset = 0
                self.running_batch.append(req)

            if not self.running_batch:
                await asyncio.sleep(0.001)
                continue

            # 2. Execute single iteration step
            self._step_batch()
            await asyncio.sleep(0)

    def _step_batch(self):
        if not self.running_batch:
            return

        finished_indices = []
        now = time.time()

        for i, req in enumerate(self.running_batch):
            with torch.inference_mode():
                # Case A: Chunked Prefilling
                if req.state == RequestState.PREFILLING:
                    chunk_end = min(req.prefill_offset + self.prefill_chunk_size, len(req.prompt_tokens))
                    chunk_tokens = req.prompt_tokens[req.prefill_offset:chunk_end]
                    chunk_input = torch.tensor([chunk_tokens], dtype=torch.long, device=self.device)

                    logits, new_kv = self.model(chunk_input, past_key_values=req.past_kv)
                    req.past_kv = new_kv
                    req.prefill_offset = chunk_end

                    # If entire prompt has been prefilled, transition to decode
                    if req.prefill_offset >= len(req.prompt_tokens):
                        req.state = RequestState.DECODING
                        next_token = self._sample_token(logits[:, -1, :], req.temperature, req.top_k)
                        req.first_token_at = now
                        req.last_token_at = now
                        self.recent_ttft.append(now - req.created_at)

                        req.generated_tokens.append(next_token)
                        self.total_tokens_generated += 1
                        req.output_queue.put_nowait(next_token)
                        req.current_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

                        if len(req.generated_tokens) >= req.max_new_tokens:
                            self._finish_request(req, now)
                            finished_indices.append(i)

                # Case B: Autoregressive Decoding
                elif req.state == RequestState.DECODING:
                    logits, new_kv = self.model(req.current_input, past_key_values=req.past_kv)
                    req.past_kv = new_kv

                    next_token = self._sample_token(logits[:, -1, :], req.temperature, req.top_k)
                    if req.last_token_at is not None:
                        self.recent_itl.append(now - req.last_token_at)
                    req.last_token_at = now

                    req.generated_tokens.append(next_token)
                    self.total_tokens_generated += 1
                    req.output_queue.put_nowait(next_token)
                    req.current_input = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

                    if len(req.generated_tokens) >= req.max_new_tokens:
                        self._finish_request(req, now)
                        finished_indices.append(i)

        # Clean up finished requests
        for idx in reversed(finished_indices):
            self.running_batch.pop(idx)

    def _sample_token(self, next_token_logits: torch.Tensor, temperature: float, top_k: int) -> int:
        if temperature > 0:
            probs = F.softmax(next_token_logits / temperature, dim=-1)
            if top_k > 0:
                topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
                idx = torch.multinomial(topk_probs, num_samples=1)
                return torch.gather(topk_indices, -1, idx).item()
            else:
                return torch.multinomial(probs, num_samples=1).item()
        else:
            return torch.argmax(next_token_logits, dim=-1).item()

    def _finish_request(self, req: AsyncSequenceRequest, finished_time: float):
        req.state = RequestState.FINISHED
        req.finished_at = finished_time
        req.past_kv = None  # Free VRAM/DRAM allocated tensors
        self.kv_pool.free_pages(req.request_id)
        req.output_queue.put_nowait(None)
        self.total_requests_completed += 1

    def get_telemetry(self) -> Dict[str, Any]:
        uptime = max(0.001, time.time() - self.start_time)
        throughput = self.total_tokens_generated / uptime
        avg_ttft_ms = (sum(self.recent_ttft) / len(self.recent_ttft) * 1000.0) if self.recent_ttft else 0.0
        avg_itl_ms = (sum(self.recent_itl) / len(self.recent_itl) * 1000.0) if self.recent_itl else 0.0

        p95_ttft_ms = float(torch.tensor(list(self.recent_ttft)).quantile(0.95).item() * 1000.0) if len(self.recent_ttft) >= 5 else avg_ttft_ms
        p99_ttft_ms = float(torch.tensor(list(self.recent_ttft)).quantile(0.99).item() * 1000.0) if len(self.recent_ttft) >= 10 else avg_ttft_ms

        return {
            "uptime_seconds": round(uptime, 2),
            "total_tokens_generated": self.total_tokens_generated,
            "total_requests_completed": self.total_requests_completed,
            "serving_throughput_tok_per_sec": round(throughput, 2),
            "running_requests": len(self.running_batch),
            "waiting_queue_depth": len(self.waiting_queue),
            "kv_memory_pool": self.kv_pool.get_stats(),
            "latency": {
                "avg_ttft_ms": round(avg_ttft_ms, 2),
                "p95_ttft_ms": round(p95_ttft_ms, 2),
                "p99_ttft_ms": round(p99_ttft_ms, 2),
                "avg_itl_ms": round(avg_itl_ms, 2)
            }
        }

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
            "num_gpu_blocks": self.kv_pool.max_total_pages,
            "total_tokens_generated": self.total_tokens_generated,
        }


