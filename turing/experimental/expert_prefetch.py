"""
One-layer-ahead expert prefetcher: while layer L computes, copies the experts predicted for layer L+1 into the
expert cache on a separate CUDA stream (synchronously elsewhere).
"""

import time
from typing import Dict, List, Tuple, Optional, Set, Any, Union
import torch
import torch.nn as nn

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.heterogeneous_moe import HostExpertBank


class OneLayerAheadExpertPrefetcher:
    """
    Asynchronously streams layer L+1 experts into GPU VRAM while layer L computes on GPU/CPU.
    Tracks prediction accuracy and hides PCIe DMA latency.
    """
    def __init__(
        self,
        expert_cache: GPULRUExpertCache,
        host_bank: Optional[HostExpertBank] = None,
        prefetch_k: int = 4,
        device: torch.device = torch.device("cpu")
    ):
        self.expert_cache = expert_cache
        self.host_bank = host_bank
        self.prefetch_k = prefetch_k
        self.device = device
        self.is_cuda = (device.type == "cuda")

        # Dedicated asynchronous prefetch CUDA stream and events
        if self.is_cuda:
            self.prefetch_stream = torch.cuda.Stream(device=device)
            self.prefetch_event = torch.cuda.Event()
        else:
            self.prefetch_stream = None
            self.prefetch_event = None

        # Telemetry
        self._prefetches = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0

    def prefetch_layer_experts_async(
        self,
        next_layer_idx: int,
        predicted_experts: Union[List[int], torch.Tensor, Set[int]],
        weight_provider: Optional[Any] = None
    ) -> List[int]:
        """
        Schedules asynchronous prefetch of missing predicted experts for layer L+1.
        Returns list of expert indices queued for prefetch.
        """
        if isinstance(predicted_experts, torch.Tensor):
            predicted_list = predicted_experts.detach().cpu().view(-1).tolist()
        elif isinstance(predicted_experts, set):
            predicted_list = list(predicted_experts)
        else:
            predicted_list = list(predicted_experts)

        # Truncate to prefetch_k unique candidates
        unique_candidates = []
        for e in predicted_list:
            if e not in unique_candidates:
                unique_candidates.append(int(e))
            if len(unique_candidates) >= self.prefetch_k:
                break

        provider = weight_provider or (self.host_bank.slot_weights if self.host_bank is not None else None)
        scheduled = []
        for exp_id in unique_candidates:
            self._prefetches += 1
            # If already resident in cache, count as prefetch hit and skip I/O
            if self.expert_cache.contains(next_layer_idx, exp_id):
                self._prefetch_hits += 1
                continue

            # Missing: allocate or evict slot in GPU LRU/LFRU cache
            slot_idx, _ = self.expert_cache.allocate_or_evict_slot(next_layer_idx, exp_id)
            if slot_idx is None:  # LFRU gate kept the resident expert: nothing to prefetch into
                continue
            scheduled.append(exp_id)
            if provider is None:  # bookkeeping only: the caller loads the slot
                continue
            if self.is_cuda and self.prefetch_stream is not None:
                with torch.cuda.stream(self.prefetch_stream):
                    self.expert_cache.load_expert_weights(slot_idx, *provider(next_layer_idx, exp_id))
                    self.prefetch_event.record(self.prefetch_stream)
            else:
                self.expert_cache.load_expert_weights(slot_idx, *provider(next_layer_idx, exp_id))

        return scheduled

    def sync_compute_stream(self, compute_stream: Optional[torch.cuda.Stream] = None):
        """
        Blocks the compute stream until scheduled prefetch DMA transfers finish.
        """
        if self.is_cuda and self.prefetch_event is not None and compute_stream is not None:
            compute_stream.wait_event(self.prefetch_event)

    def record_access_result(self, was_hit: bool):
        """
        Records whether subsequent actual execution hit or missed the prefetched cache state.
        """
        if was_hit:
            self._prefetch_hits += 1
        else:
            self._prefetch_misses += 1

    @property
    def hit_rate(self) -> float:
        total = self._prefetch_hits + self._prefetch_misses
        if total == 0:
            return 0.0
        return (self._prefetch_hits / total) * 100.0

    def stats(self) -> Dict[str, Any]:
        return {
            "prefetches": self._prefetches,
            "prefetch_hits": self._prefetch_hits,
            "prefetch_misses": self._prefetch_misses,
            "prefetch_hit_rate_pct": f"{self.hit_rate:.2f}%",
            "prefetch_k": self.prefetch_k
        }
