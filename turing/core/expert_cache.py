"""
Global Multi-Layer LRU Expert Cache in GPU VRAM (Turing Engine Integration).
Maintains an in-VRAM slot cache of recently activated MoE experts across all layers
with asynchronous PCIe prefetching, eliminating transfer latency on temporal hits.
"""

import time
from typing import Dict, Tuple, Optional, List, Set, Any
from collections import OrderedDict
import torch
import torch.nn as nn

class GPULRUExpertCache:
    """
    Global LRU cache residing in GPU VRAM with fixed slots across all layers.
    """
    def __init__(
        self,
        num_slots: int = 32,
        hidden_dim: int = 4096,
        active_subspace_dim: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32
    ):
        self.num_slots = num_slots
        self.active_slots = num_slots
        self.hidden_dim = hidden_dim
        self.active_dim = active_subspace_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocated GPU VRAM tensor buffers for slots
        # Gate, Up, Down projections for each slot
        self.slot_gate_weights = torch.empty(num_slots, hidden_dim, active_subspace_dim, device=device, dtype=dtype)
        self.slot_up_weights = torch.empty(num_slots, hidden_dim, active_subspace_dim, device=device, dtype=dtype)
        self.slot_down_weights = torch.empty(num_slots, active_subspace_dim, hidden_dim, device=device, dtype=dtype)

        # Fast C++20 LRU cache backend if available
        try:
            import turing.turing_csrc as turing_csrc
            self.csrc_cache = turing_csrc.LRUExpertCacheFast(num_slots)
            self.has_csrc = True
        except ImportError:
            try:
                import turing_csrc
                self.csrc_cache = turing_csrc.LRUExpertCacheFast(num_slots)
                self.has_csrc = True
            except ImportError:
                self.csrc_cache = None
                self.has_csrc = False

        # LRU bookkeeping fallback: (layer_idx, expert_idx) -> slot_idx
        self.cache_map: OrderedDict[Tuple[int, int], int] = OrderedDict()
        self.free_slots: List[int] = list(range(num_slots))

        # Metrics
        self._hits = 0
        self._misses = 0

    def resize_active_slots(self, new_num_slots: int) -> int:
        """
        Dynamically adjusts the active capacity of the expert cache without reallocating tensors.
        If contracting, evicts LRU experts down to new_num_slots.
        Returns the new active capacity.
        """
        new_slots = max(1, min(new_num_slots, self.num_slots))
        self.active_slots = new_slots

        # If shrinking below current used slots, evict LRU entries
        while len(self.cache_map) > new_slots:
            self.cache_map.popitem(last=False)

        # Recompute free slots based on active capacity
        occupied = set(self.cache_map.values())
        self.free_slots = [s for s in range(new_slots) if s not in occupied]
        return new_slots


    @property
    def used_slots(self) -> int:
        if self.has_csrc and self.csrc_cache is not None:
            return min(self.active_slots, self.csrc_cache.used_slots)
        return len(self.cache_map)

    @property
    def hits(self) -> int:
        if self.has_csrc and self.csrc_cache is not None:
            return self.csrc_cache.hits
        return self._hits


    @property
    def misses(self) -> int:
        if self.has_csrc and self.csrc_cache is not None:
            return self.csrc_cache.misses
        return self._misses

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        if self.has_csrc and self.csrc_cache is not None:
            return self.csrc_cache.contains(layer_idx, expert_idx)
        return (layer_idx, expert_idx) in self.cache_map

    def get_slot(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """
        Retrieves slot index on cache hit and updates LRU order.
        """
        if self.has_csrc and self.csrc_cache is not None:
            slot = self.csrc_cache.get_slot(layer_idx, expert_idx)
            return slot if slot >= 0 else None

        key = (layer_idx, expert_idx)
        if key in self.cache_map:
            self._hits += 1
            # Move to most recently used end
            self.cache_map.move_to_end(key)
            return self.cache_map[key]
        self._misses += 1
        return None

    def allocate_or_evict_slot(self, layer_idx: int, expert_idx: int) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Allocates a slot for a missing expert. If full, evicts LRU expert.
        Returns: (slot_idx, evicted_key)
        """
        if self.has_csrc and self.csrc_cache is not None:
            slot, ev_layer, ev_expert = self.csrc_cache.allocate_or_evict_slot(layer_idx, expert_idx)
            evicted_key = (ev_layer, ev_expert) if ev_layer >= 0 else None
            return slot, evicted_key

        key = (layer_idx, expert_idx)
        if key in self.cache_map:
            self.cache_map.move_to_end(key)
            return self.cache_map[key], None

        evicted_key = None
        if len(self.cache_map) >= self.active_slots or not self.free_slots:
            # Evict least recently used (first item)
            evicted_key, slot_idx = self.cache_map.popitem(last=False)
        else:
            slot_idx = self.free_slots.pop()

        self.cache_map[key] = slot_idx
        return slot_idx, evicted_key


    def load_expert_weights(
        self,
        slot_idx: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor
    ):
        """
        Loads expert weights into designated GPU slot buffer.
        """
        self.slot_gate_weights[slot_idx].copy_(w_gate)
        self.slot_up_weights[slot_idx].copy_(w_up)
        self.slot_down_weights[slot_idx].copy_(w_down)

    @property
    def hit_rate(self) -> float:
        if self.has_csrc and self.csrc_cache is not None:
            return self.csrc_cache.hit_rate
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    def stats(self) -> Dict[str, Any]:
        if self.has_csrc and self.csrc_cache is not None:
            used = self.csrc_cache.used_slots
            hits = self.csrc_cache.hits
            misses = self.csrc_cache.misses
            return {
                "total_slots": self.num_slots,
                "used_slots": used,
                "free_slots": self.num_slots - used,
                "hits": hits,
                "misses": misses,
                "hit_rate_pct": f"{self.hit_rate:.2f}%"
            }
        return {
            "total_slots": self.num_slots,
            "used_slots": len(self.cache_map),
            "free_slots": len(self.free_slots),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": f"{self.hit_rate:.2f}%"
        }
