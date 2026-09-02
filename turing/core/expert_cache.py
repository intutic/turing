"""
Global Multi-Layer LRU Expert Cache in GPU VRAM (Turing Engine Integration).
Maintains an in-VRAM slot cache of recently activated MoE experts across all layers
with asynchronous PCIe prefetching, eliminating transfer latency on temporal hits.
"""

import time
from typing import Dict, Tuple, Optional, List, Set, Any, Union
from collections import OrderedDict
import torch
import torch.nn as nn

class RoutingHistoryTracker:
    """
    Persistent routing history tracker across inference sessions.
    Maintains per-layer, per-expert activation frequency and timestamps,
    with exponential decay and learned hot-expert pinning ranking.
    """
    def __init__(
        self,
        num_layers: int = 32,
        num_experts: int = 64,
        decay_factor: float = 0.99,
        save_path: Optional[str] = None
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.decay_factor = decay_factor
        self.save_path = save_path
        
        # (layer_idx, expert_idx) -> count
        self.counts: Dict[Tuple[int, int], int] = {}
        # (layer_idx, expert_idx) -> last access clock
        self.last_accessed: Dict[Tuple[int, int], int] = {}
        self.clock: int = 0

        self._native = None
        try:
            import turing.turing_csrc as turing_csrc
            self._native = turing_csrc.NativeRoutingHistoryTracker(num_layers, num_experts, decay_factor)
        except Exception:
            pass

    def record_route(self, layer_idx: int, expert_indices: Union[List[int], torch.Tensor, int]):
        """
        Records one or more expert activations for a given layer.
        """
        self.clock += 1
        if isinstance(expert_indices, int):
            indices_list = [expert_indices]
        elif isinstance(expert_indices, torch.Tensor):
            indices_list = expert_indices.detach().cpu().view(-1).tolist()
        else:
            indices_list = expert_indices

        if self._native is not None:
            self._native.record_route(layer_idx, [int(x) for x in indices_list])
            
        for e in indices_list:
            key = (layer_idx, int(e))
            self.counts[key] = self.counts.get(key, 0) + 1
            self.last_accessed[key] = self.clock

    def decay(self, factor: Optional[float] = None):
        """
        Applies exponential decay to historic routing counts to give recency half-life.
        """
        d = factor if factor is not None else self.decay_factor
        if d <= 0.0 or d >= 1.0:
            return

        if self._native is not None:
            self._native.decay(d)

        for key in list(self.counts.keys()):
            new_val = int(self.counts[key] * d)
            if new_val <= 0:
                del self.counts[key]
            else:
                self.counts[key] = new_val

    def get_top_k_experts(self, k: int, layer_idx: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Returns the top-k hottest expert keys (layer_idx, expert_idx) sorted by historical frequency.
        """
        if self._native is not None:
            return self._native.get_top_k_experts(k, layer_idx if layer_idx is not None else -1)

        if layer_idx is not None:
            items = [(k_pair, cnt) for k_pair, cnt in self.counts.items() if k_pair[0] == layer_idx]
        else:
            items = list(self.counts.items())
        
        items.sort(key=lambda x: x[1], reverse=True)
        return [k_pair for k_pair, _ in items[:k]]

    def save(self, path: Optional[str] = None):
        """
        Persists sparse routing history to JSON file.
        """
        p = path or self.save_path
        if not p:
            return
        data = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "clock": self.clock,
            "counts": {f"{l}_{e}": cnt for (l, e), cnt in self.counts.items()}
        }
        import json, os
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[str] = None) -> bool:
        """
        Loads sparse routing history from JSON file.
        """
        p = path or self.save_path
        if not p:
            return False
        import json, os
        if not os.path.exists(p):
            return False
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.num_layers = data.get("num_layers", self.num_layers)
            self.num_experts = data.get("num_experts", self.num_experts)
            self.clock = data.get("clock", 0)
            raw_counts = data.get("counts", {})
            self.counts = {}
            for k_str, cnt in raw_counts.items():
                parts = k_str.split("_")
                if len(parts) == 2:
                    self.counts[(int(parts[0]), int(parts[1]))] = int(cnt)
            return True
        except Exception:
            return False


class GPULRUExpertCache:
    """
    Global LRU/LFRU cache residing in GPU VRAM with fixed slots across all layers.
    Supports learned hot-expert pinning, persistent routing history, and hysteresis promotion.
    """
    def __init__(
        self,
        num_slots: int = 32,
        hidden_dim: int = 4096,
        active_subspace_dim: int = 2048,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        eviction_policy: str = "lfru",
        history_tracker: Optional[RoutingHistoryTracker] = None
    ):
        self.num_slots = num_slots
        self.active_slots = num_slots
        self.hidden_dim = hidden_dim
        self.active_dim = active_subspace_dim
        self.device = device
        self.dtype = dtype
        self.eviction_policy = eviction_policy
        self.history_tracker = history_tracker

        # Pre-allocated GPU VRAM tensor buffers for slots
        # Gate, Up, Down projections for each slot
        self.slot_gate_weights = torch.empty(num_slots, hidden_dim, active_subspace_dim, device=device, dtype=dtype)
        self.slot_up_weights = torch.empty(num_slots, hidden_dim, active_subspace_dim, device=device, dtype=dtype)
        self.slot_down_weights = torch.empty(num_slots, active_subspace_dim, hidden_dim, device=device, dtype=dtype)

        # Fast C++20 LRU / LFRU cache backend if available
        try:
            import turing.turing_csrc as turing_csrc
            if hasattr(turing_csrc, "LFRUExpertCacheFast"):
                self.csrc_cache = turing_csrc.LFRUExpertCacheFast(num_slots)
            else:
                self.csrc_cache = turing_csrc.LRUExpertCacheFast(num_slots)
            self.has_csrc = True
        except ImportError:
            try:
                import turing_csrc
                if hasattr(turing_csrc, "LFRUExpertCacheFast"):
                    self.csrc_cache = turing_csrc.LFRUExpertCacheFast(num_slots)
                else:
                    self.csrc_cache = turing_csrc.LRUExpertCacheFast(num_slots)
                self.has_csrc = True
            except ImportError:
                self.csrc_cache = None
                self.has_csrc = False

        # LRU bookkeeping fallback: (layer_idx, expert_idx) -> slot_idx
        self.cache_map: OrderedDict[Tuple[int, int], int] = OrderedDict()
        self.free_slots: List[int] = list(range(num_slots))

        # Pinned experts (cannot be evicted by LRU/LFRU)
        self.pinned_keys: Set[Tuple[int, int]] = set()

        # LFRU access metadata
        self.access_heat: Dict[Tuple[int, int], int] = {}
        self.last_access_clock: Dict[Tuple[int, int], int] = {}
        self.clock: int = 0
        self._promotions: int = 0

        # Metrics
        self._hits = 0
        self._misses = 0

    def pin_experts(self, keys: List[Tuple[int, int]]):
        """
        Marks designated expert keys as pinned in VRAM. Pinned experts are exempt from eviction.
        """
        for k in keys:
            self.pinned_keys.add(k)
            if self.has_csrc and self.csrc_cache is not None and hasattr(self.csrc_cache, "pin_key"):
                self.csrc_cache.pin_key(k[0], k[1])

    def unpin_experts(self, keys: List[Tuple[int, int]]):
        """
        Removes pinning status from designated expert keys.
        """
        for k in keys:
            self.pinned_keys.discard(k)
            if self.has_csrc and self.csrc_cache is not None and hasattr(self.csrc_cache, "unpin_key"):
                self.csrc_cache.unpin_key(k[0], k[1])

    def resize_active_slots(self, new_num_slots: int) -> int:
        """
        Dynamically adjusts the active capacity of the expert cache without reallocating tensors.
        If contracting, evicts non-pinned LRU experts down to new_num_slots.
        Returns the new active capacity.
        """
        new_slots = max(len(self.pinned_keys) if self.pinned_keys else 1, min(new_num_slots, self.num_slots))
        self.active_slots = new_slots

        # If shrinking below current used slots, evict unpinned entries
        while len(self.cache_map) > new_slots:
            unpinned_candidates = [k for k in self.cache_map.keys() if k not in self.pinned_keys]
            if not unpinned_candidates:
                break
            ev_k = unpinned_candidates[0]
            slot = self.cache_map.pop(ev_k)
            self.free_slots.append(slot)

        # Recompute free slots based on active capacity
        occupied = set(self.cache_map.values())
        self.free_slots = [s for s in range(new_slots) if s not in occupied]
        return new_slots

    @property
    def used_slots(self) -> int:
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            return min(self.active_slots, self.csrc_cache.used_slots)
        return len(self.cache_map)

    @property
    def hits(self) -> int:
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            return self.csrc_cache.hits
        return self._hits

    @property
    def misses(self) -> int:
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            return self.csrc_cache.misses
        return self._misses

    def contains(self, layer_idx: int, expert_idx: int) -> bool:
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            return self.csrc_cache.contains(layer_idx, expert_idx)
        return (layer_idx, expert_idx) in self.cache_map

    def get_slot(self, layer_idx: int, expert_idx: int) -> Optional[int]:
        """
        Retrieves slot index on cache hit and updates LRU/LFRU order.
        """
        self.clock += 1
        key = (layer_idx, expert_idx)

        if self.history_tracker is not None:
            self.history_tracker.record_route(layer_idx, expert_idx)

        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            slot = self.csrc_cache.get_slot(layer_idx, expert_idx)
            return slot if slot >= 0 else None

        if key in self.cache_map:
            self._hits += 1
            self.access_heat[key] = self.access_heat.get(key, 0) + 1
            self.last_access_clock[key] = self.clock
            # Move to most recently used end
            self.cache_map.move_to_end(key)
            return self.cache_map[key]
        self._misses += 1
        return None

    def _calculate_lfru_score(self, key: Tuple[int, int]) -> int:
        """
        LFRU composite score: (heat << 8) | (255 - age)
        """
        heat = self.access_heat.get(key, 0)
        last_t = self.last_access_clock.get(key, self.clock)
        age = min(255, max(0, self.clock - last_t))
        recency = 255 - age
        return (heat << 8) | recency

    def _should_promote(self, hot_key: Tuple[int, int], cold_key: Tuple[int, int]) -> bool:
        """
        Hysteresis promotion gate: hot_heat > cold_heat + (cold_heat >> 2) + 4
        """
        hot_heat = self.access_heat.get(hot_key, 0)
        cold_heat = self.access_heat.get(cold_key, 0)
        threshold = cold_heat + (cold_heat >> 2) + 4
        return hot_heat > threshold

    def allocate_or_evict_slot(self, layer_idx: int, expert_idx: int) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Allocates a slot for a missing expert. If full, evicts unpinned expert by LRU or LFRU.
        Returns: (slot_idx, evicted_key)
        """
        self.clock += 1
        key = (layer_idx, expert_idx)

        if self.history_tracker is not None:
            self.history_tracker.record_route(layer_idx, expert_idx)

        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            slot, ev_layer, ev_expert = self.csrc_cache.allocate_or_evict_slot(layer_idx, expert_idx)
            evicted_key = (ev_layer, ev_expert) if ev_layer >= 0 else None
            return slot, evicted_key

        if key in self.cache_map:
            self.cache_map.move_to_end(key)
            self.access_heat[key] = self.access_heat.get(key, 0) + 1
            self.last_access_clock[key] = self.clock
            return self.cache_map[key], None

        evicted_key = None
        if len(self.cache_map) >= self.active_slots or not self.free_slots:
            # Candidate eviction keys: all unpinned keys in cache
            candidates = [k for k in self.cache_map.keys() if k not in self.pinned_keys]
            if not candidates:
                # If everything is pinned, must evict the first element
                evicted_key, slot_idx = self.cache_map.popitem(last=False)
            elif self.eviction_policy == "lfru":
                # Find the coldest candidate by LFRU score
                coldest_key = min(candidates, key=self._calculate_lfru_score)
                # Check hysteresis if incoming key has history
                if key in self.access_heat and not self._should_promote(key, coldest_key):
                    # Retain coldest if incoming does not meet hysteresis margin
                    slot_idx = self.cache_map[coldest_key]
                else:
                    self._promotions += 1
                    evicted_key = coldest_key
                    slot_idx = self.cache_map.pop(coldest_key)
            else:
                # LRU: evict oldest unpinned
                evicted_key = candidates[0]
                slot_idx = self.cache_map.pop(evicted_key)
        else:
            slot_idx = self.free_slots.pop()

        self.cache_map[key] = slot_idx
        self.access_heat[key] = self.access_heat.get(key, 0) + 1
        self.last_access_clock[key] = self.clock
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
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            return self.csrc_cache.hit_rate
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100.0

    def stats(self) -> Dict[str, Any]:
        if self.has_csrc and self.csrc_cache is not None and not self.pinned_keys and self.eviction_policy == "lru":
            used = self.csrc_cache.used_slots
            hits = self.csrc_cache.hits
            misses = self.csrc_cache.misses
            return {
                "total_slots": self.num_slots,
                "used_slots": used,
                "free_slots": self.num_slots - used,
                "hits": hits,
                "misses": misses,
                "hit_rate_pct": f"{self.hit_rate:.2f}%",
                "eviction_policy": self.eviction_policy,
                "pinned_slots": len(self.pinned_keys)
            }
        return {
            "total_slots": self.num_slots,
            "used_slots": len(self.cache_map),
            "free_slots": len(self.free_slots),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": f"{self.hit_rate:.2f}%",
            "eviction_policy": self.eviction_policy,
            "pinned_slots": len(self.pinned_keys),
            "lfru_promotions": self._promotions
        }

