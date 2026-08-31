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
        self._shed_count: int = 0
        self._queue_count: int = 0

    def admit(self, request_id: str, estimated_bytes: int) -> AdmissionResult:
        """Admit, queue, or shed a request based on available VRAM budget."""
        current_usage = sum(self._allocated.values())
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
            
        self._allocated[request_id] = estimated_bytes
        return AdmissionResult(decision=AdmissionDecision.ADMIT)

    def release(self, request_id: str) -> None:
        """Release allocated budget for a request."""
        self._allocated.pop(request_id, None)

    @property
    def utilization(self) -> float:
        """Current VRAM utilization ratio."""
        return float(sum(self._allocated.values())) / max(self.vram_budget_bytes, 1)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the admission controller."""
        return {
            "utilization": self.utilization,
            "allocated_bytes": sum(self._allocated.values()),
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
