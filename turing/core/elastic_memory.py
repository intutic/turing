"""
Elastic Memory Budget Manager (FreeToken / Turing Engine Co-Design).
Dynamically rebalances memory between the MoE Expert Slot Cache (GPULRUExpertCache)
and the KV Cache Page Pool (StaticPagedKVPool / HierarchicalVirtualPageManager)
at runtime based on active batch context lengths without engine restarts.
"""

from typing import Dict, Any, Optional
import math
from .expert_cache import GPULRUExpertCache
from .paging import StaticPagedKVPool


class ElasticMemoryBudgetManager:
    """
    Orchestrates dynamic memory reallocation between MoE expert slots and KV pages.
    """
    def __init__(
        self,
        expert_cache: GPULRUExpertCache,
        kv_pool: StaticPagedKVPool,
        bytes_per_expert_slot: Optional[int] = None,
        bytes_per_kv_page: Optional[int] = None,
        min_expert_slots: int = 4,
        max_expert_slots: Optional[int] = None,
        min_kv_pages: int = 32,
        max_kv_pages: Optional[int] = None,
        target_kv_headroom_ratio: float = 0.25
    ):
        self.expert_cache = expert_cache
        self.kv_pool = kv_pool

        # Determine bytes per slot and page
        if bytes_per_expert_slot is None:
            # 3 projections (gate, up, down): 3 * (hidden * active) * element_size
            g_bytes = expert_cache.slot_gate_weights[0].numel() * expert_cache.slot_gate_weights.element_size()
            u_bytes = expert_cache.slot_up_weights[0].numel() * expert_cache.slot_up_weights.element_size()
            d_bytes = expert_cache.slot_down_weights[0].numel() * expert_cache.slot_down_weights.element_size()
            self.bytes_per_slot = max(1024, g_bytes + u_bytes + d_bytes)
        else:
            self.bytes_per_slot = bytes_per_expert_slot

        if bytes_per_kv_page is None:
            # layers * heads * page_size * head_dim * element_size * 2 (k and v)
            k_page_bytes = (
                kv_pool.k_pool.shape[0] * kv_pool.k_pool.shape[2] * kv_pool.page_size * kv_pool.head_dim * kv_pool.k_pool.element_size()
            )
            self.bytes_per_page = max(1024, 2 * k_page_bytes)
        else:
            self.bytes_per_page = bytes_per_kv_page

        self.min_expert_slots = min_expert_slots
        self.max_expert_slots = max_expert_slots or expert_cache.num_slots
        self.min_kv_pages = min_kv_pages
        self.max_kv_pages = max_kv_pages or kv_pool.max_total_pages
        self.target_headroom = target_kv_headroom_ratio
        self.rebalance_count = 0

    def evaluate_and_rebalance(
        self,
        current_active_tokens: int,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates KV page demand based on active tokens + headroom.
        If KV pool is constrained (<20% free pages), shrinks MoE slots to expand KV pages.
        If KV pool has excess headroom (>50% free pages), expands MoE slots for higher hit rate.
        """
        tokens_per_page = self.kv_pool.page_size
        needed_pages = math.ceil(current_active_tokens / max(1, tokens_per_page))
        target_kv_pages = max(
            self.min_kv_pages,
            min(
                self.max_kv_pages,
                math.ceil(needed_pages * (1.0 + self.target_headroom))
            )
        )

        curr_kv_stats = self.kv_pool.get_stats()
        curr_kv_pages = curr_kv_stats["total_pages"]
        curr_slots = self.expert_cache.active_slots

        rebalanced = False
        action = "steady"

        # Check if KV pool needs expansion
        if target_kv_pages > curr_kv_pages:
            # We need to expand KV pages and may need to contract expert cache slots
            page_diff = target_kv_pages - curr_kv_pages
            slots_to_yield = max(1, math.ceil((page_diff * self.bytes_per_page) / self.bytes_per_slot))
            new_slots = max(self.min_expert_slots, curr_slots - slots_to_yield)
            new_pages = min(self.max_kv_pages, curr_kv_pages + page_diff)

            self.expert_cache.resize_active_slots(new_slots)
            self.kv_pool.adjust_active_capacity(new_pages)
            rebalanced = True
            action = f"expand_kv_{page_diff}_pages"
            self.rebalance_count += 1

        elif target_kv_pages < curr_kv_pages and (curr_kv_pages - target_kv_pages) >= 16:
            # Excess KV pages can be released to expand MoE slots
            page_surplus = curr_kv_pages - target_kv_pages
            slots_to_gain = (page_surplus * self.bytes_per_page) // self.bytes_per_slot
            if slots_to_gain > 0:
                new_slots = min(self.max_expert_slots, curr_slots + slots_to_gain)
                new_pages = max(self.min_kv_pages, target_kv_pages)

                self.expert_cache.resize_active_slots(new_slots)
                self.kv_pool.adjust_active_capacity(new_pages)
                rebalanced = True
                action = f"expand_moe_{slots_to_gain}_slots"
                self.rebalance_count += 1

        return {
            "rebalanced": rebalanced,
            "action": action,
            "rebalance_count": self.rebalance_count,
            "active_expert_slots": self.expert_cache.active_slots,
            "active_kv_pages": self.kv_pool.active_max_pages,
            "kv_utilization_pct": self.kv_pool.get_stats()["utilization_pct"],
            "expert_cache_hit_rate_pct": self.expert_cache.hit_rate
        }
