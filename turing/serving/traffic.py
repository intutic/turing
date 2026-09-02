"""
AI Traffic Management module for the Turing Engine serving layer.

This module implements token-budget-aware routing, VRAM admission control,
and 3-lane QoS scheduling. It is adapted from memra's lane system and
borrows concepts from vLLM/SGLang traffic patterns for efficient LLM serving.
"""

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

__all__ = [
    "KVMemoryEstimator",
    "PrefixHashRouter",
    "AdmissionDecision",
    "AdmissionResult",
    "AdmissionController",
    "Lane",
    "LanePolicy",
    "ComputeCostEstimator",
    "AdaptiveComputeBatcher",
]

class KVMemoryEstimator:
    """Estimates GPU memory footprint of KV caches for admission decisions."""
    
    @staticmethod
    def estimate_kv_bytes(
        num_prompt_tokens: int,
        max_new_tokens: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,
        svd_compression_ratio: float = 0.0,
    ) -> int:
        """Estimate the KV cache size in bytes."""
        try:
            from turing.turing_csrc import NativeTrafficManager
            return int(NativeTrafficManager.estimate_kv_bytes(
                num_prompt_tokens, max_new_tokens, num_layers, num_kv_heads, head_dim, dtype_bytes, svd_compression_ratio
            ))
        except Exception:
            pass

        total_tokens = num_prompt_tokens + max_new_tokens
        bytes_per_token = num_kv_heads * head_dim * num_layers * 2 * dtype_bytes
        total_bytes = total_tokens * bytes_per_token * (1.0 - svd_compression_ratio)
        return int(total_bytes)

    @staticmethod
    def estimate_vram_utilization(active_requests: List[Dict[str, Any]], total_vram_bytes: int) -> float:
        """Estimate the overall VRAM utilization for a list of active requests."""
        total_bytes = 0
        for req in active_requests:
            total_bytes += KVMemoryEstimator.estimate_kv_bytes(
                num_prompt_tokens=req.get("num_tokens", 0),
                max_new_tokens=req.get("max_new_tokens", 0),
                num_layers=req.get("num_layers", 0),
                num_kv_heads=req.get("num_kv_heads", 0),
                head_dim=req.get("head_dim", 0),
            )
        return float(total_bytes) / max(total_vram_bytes, 1)

class PrefixHashRouter:
    """Routes requests with matching system prompts to the same worker for prefix cache hits."""
    
    def __init__(self, window: int = 128) -> None:
        self.window = window

    def compute_prefix_hash(self, token_ids: List[int]) -> int:
        """Compute FNV-1a 64-bit hash over the prefix of tokens."""
        try:
            import turing.turing_csrc as turing_csrc
            import numpy as np
            tok_arr = np.array(token_ids[:self.window], dtype=np.int32)
            return int(turing_csrc.compute_prefix_hash_fast(tok_arr, self.window))
        except Exception:
            pass

        offset_basis = 0xcbf29ce484222325
        prime = 0x100000001b3
        
        hash_val = offset_basis
        for token in token_ids[:self.window]:
            hash_val ^= (token & 0xFF)
            hash_val *= prime
            hash_val &= 0xFFFFFFFFFFFFFFFF
        return hash_val

    def route_to_worker(self, prefix_hash: int, num_workers: int) -> int:
        """Route to a worker using a simple consistent hash."""
        return prefix_hash % num_workers

class AdmissionDecision(enum.Enum):
    """Possible decisions for admission control."""
    ADMIT = "admit"
    QUEUE = "queue"
    SHED = "shed"

@dataclass
class AdmissionResult:
    """Result of an admission request."""
    decision: AdmissionDecision
    retry_after_seconds: Optional[float] = None
    reason: Optional[str] = None

class AdmissionController:
    """Prevents OOM by tracking estimated VRAM budget."""
    
    def __init__(
        self,
        vram_budget_bytes: int,
        high_watermark: float = 0.90,
        shed_watermark: float = 0.95
    ) -> None:
        self.vram_budget_bytes = vram_budget_bytes
        self.high_watermark = high_watermark
        self.shed_watermark = shed_watermark
        self._allocated: Dict[str, int] = {}
        self._allocated_total: int = 0
        self._shed_count: int = 0
        self._queue_count: int = 0

    def admit(self, request_id: str, estimated_bytes: int) -> AdmissionResult:
        """Admit, queue, or shed a request based on available VRAM budget."""
        current_usage = self._allocated_total
        utilization = (current_usage + estimated_bytes) / float(max(self.vram_budget_bytes, 1))
        
        if utilization >= self.shed_watermark:
            self._shed_count += 1
            return AdmissionResult(
                decision=AdmissionDecision.SHED,
                reason="VRAM usage exceeds shed watermark."
            )
        
        if utilization >= self.high_watermark:
            self._queue_count += 1
            return AdmissionResult(
                decision=AdmissionDecision.QUEUE,
                retry_after_seconds=2.0,
                reason="VRAM usage exceeds high watermark."
            )
            
        prev = self._allocated.get(request_id, 0)
        self._allocated[request_id] = estimated_bytes
        self._allocated_total += (estimated_bytes - prev)
        return AdmissionResult(decision=AdmissionDecision.ADMIT)

    def release(self, request_id: str) -> None:
        """Release allocated budget for a request."""
        freed = self._allocated.pop(request_id, 0)
        self._allocated_total = max(0, self._allocated_total - freed)

    @property
    def utilization(self) -> float:
        """Current VRAM utilization ratio."""
        return float(self._allocated_total) / max(self.vram_budget_bytes, 1)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the admission controller."""
        return {
            "utilization": self.utilization,
            "allocated_bytes": self._allocated_total,
            "shed_count": self._shed_count,
            "queue_count": self._queue_count,
            "active_requests": len(self._allocated),
        }

class Lane(enum.Enum):
    """QoS lanes for request scheduling."""
    INTERACTIVE = "interactive"
    BATCH = "batch"
    BACKGROUND = "background"

class LanePolicy:
    """3-lane QoS scheduling policy adapted from memra."""
    
    def __init__(
        self,
        slo_target_p99_ms: float = 50.0,
        interactive_prefill_chunk: int = 512,
        batch_prefill_chunk: int = 256,
        background_prefill_chunk: int = 128
    ) -> None:
        self.slo_target_p99_ms = slo_target_p99_ms
        self.interactive_prefill_chunk = interactive_prefill_chunk
        self.batch_prefill_chunk = batch_prefill_chunk
        self.background_prefill_chunk = background_prefill_chunk

    def classify_request(
        self,
        max_tokens: int = 32,
        stream: bool = True,
        lane_header: Optional[str] = None
    ) -> Lane:
        """Classify a request into a QoS lane."""
        if lane_header is not None:
            try:
                return Lane(lane_header.lower())
            except ValueError:
                pass
                
        if stream and max_tokens <= 256:
            return Lane.INTERACTIVE
            
        if max_tokens > 1024:
            return Lane.BATCH
            
        return Lane.INTERACTIVE

    def prefill_chunk_budget(self, lane: Lane) -> int:
        """Get the prefill chunk size for the given lane."""
        if lane == Lane.INTERACTIVE:
            return self.interactive_prefill_chunk
        elif lane == Lane.BATCH:
            return self.batch_prefill_chunk
        elif lane == Lane.BACKGROUND:
            return self.background_prefill_chunk
        return self.interactive_prefill_chunk

    def should_shed(self, lane: Lane, interactive_p99_ms: float) -> bool:
        """Decide whether to shed requests from a lane based on interactive p99 latency."""
        if lane == Lane.INTERACTIVE:
            return False
        elif lane == Lane.BATCH:
            return interactive_p99_ms > self.slo_target_p99_ms
        elif lane == Lane.BACKGROUND:
            return interactive_p99_ms > (self.slo_target_p99_ms * 0.9)
        return False

    def priority(self, lane: Lane) -> int:
        """Get priority of the lane (lower is higher priority)."""
        if lane == Lane.INTERACTIVE:
            return 0
        elif lane == Lane.BATCH:
            return 1
        elif lane == Lane.BACKGROUND:
            return 2
        return 0


class ComputeCostEstimator:
    """Estimates the computational complexity (FLOPs / memory operations) of heterogeneous AI tasks."""

    @staticmethod
    def estimate_cost(
        task_type: str,
        seq_len: int,
        hidden_dim: int = 4096,
        max_new_tokens: int = 0,
        num_candidates: int = 1,
    ) -> int:
        """
        Calculates theoretical unit compute cost.
        - Embedding (linear in length): seq_len * hidden_dim
        - Rerank / Cross-Encoder (quadratic in combined length): (seq_len ** 2) * num_candidates
        - Entity Extraction (linear in document length): seq_len * hidden_dim
        - Generation (prefill + autoregressive steps): seq_len + max_new_tokens
        """
        task = task_type.lower().strip()
        if task in ("embedding", "embed", "encode"):
            return max(1, seq_len * hidden_dim)
        elif task in ("rerank", "score", "cross_encoder"):
            return max(1, (seq_len ** 2) * max(1, num_candidates))
        elif task in ("extract", "ner", "ocr"):
            return max(1, seq_len * hidden_dim)
        elif task in ("generate", "chat", "completion"):
            return max(1, seq_len + max_new_tokens)
        else:
            return max(1, seq_len * hidden_dim)


class AdaptiveComputeBatcher:
    """
    Compute-cost adaptive variable-length batch scheduler.
    Groups pending requests into cost-matched buckets to prevent padding waste,
    and packs variable-length token arrays into flat tensors with cu_seqlens.
    """

    def __init__(
        self,
        max_batch_cost: int = 1048576,
        max_batch_size: int = 64,
        cost_tolerance_ratio: float = 0.50,
    ) -> None:
        self.max_batch_cost = max_batch_cost
        self.max_batch_size = max_batch_size
        self.cost_tolerance_ratio = cost_tolerance_ratio

    def bucket_requests(self, requests: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Sorts and partitions requests into cost-homogeneous batches where the total
        batch compute cost does not exceed max_batch_cost and within-batch cost divergence
        is bounded by cost_tolerance_ratio.
        """
        if not requests:
            return []

        # Tag each request with its computed cost
        decorated = []
        for req in requests:
            cost = req.get("_compute_cost")
            if cost is None:
                cost = ComputeCostEstimator.estimate_cost(
                    task_type=req.get("task", "embedding"),
                    seq_len=req.get("seq_len", len(req.get("tokens", [])) or 1),
                    hidden_dim=req.get("hidden_dim", 4096),
                    max_new_tokens=req.get("max_new_tokens", 0),
                    num_candidates=req.get("num_candidates", 1),
                )
            decorated.append((cost, req))

        # Sort by cost ascending
        decorated.sort(key=lambda x: x[0])

        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
        current_batch_cost: int = 0
        pivot_cost: Optional[int] = None

        for cost, req in decorated:
            if not current_batch:
                current_batch.append(req)
                current_batch_cost = cost
                pivot_cost = cost
                continue

            # Check capacity and cost divergence
            cost_divergence = abs(cost - pivot_cost) / float(max(pivot_cost, 1))
            will_exceed_cost = (current_batch_cost + cost) > self.max_batch_cost
            will_exceed_size = len(current_batch) >= self.max_batch_size
            divergence_too_high = cost_divergence > self.cost_tolerance_ratio

            if will_exceed_cost or will_exceed_size or divergence_too_high:
                batches.append(current_batch)
                current_batch = [req]
                current_batch_cost = cost
                pivot_cost = cost
            else:
                current_batch.append(req)
                current_batch_cost += cost

        if current_batch:
            batches.append(current_batch)

        return batches

    @staticmethod
    def pack_varlen_tokens(token_lists: List[List[int]]) -> tuple[List[int], List[int]]:
        """
        Packs a list of variable-length token lists into a contiguous flat 1D token array
        and a cumulative sequence lengths array (cu_seqlens) starting at 0.
        
        Returns:
            (flat_tokens, cu_seqlens)
        """
        try:
            import turing.turing_csrc as turing_csrc
            return turing_csrc.pack_varlen_tokens_fast(token_lists)
        except Exception:
            pass

        flat_tokens: List[int] = []
        cu_seqlens: List[int] = [0]
        cumulative = 0

        for tokens in token_lists:
            flat_tokens.extend(tokens)
            cumulative += len(tokens)
            cu_seqlens.append(cumulative)

        return flat_tokens, cu_seqlens

