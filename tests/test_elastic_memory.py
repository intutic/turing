import pytest
import torch

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.paging import StaticPagedKVPool
from turing.core.elastic_memory import ElasticMemoryBudgetManager


def test_elastic_memory_budget_manager_rebalance():
    # Setup expert cache (32 slots) and KV pool (256 pages)
    expert_cache = GPULRUExpertCache(
        num_slots=32,
        hidden_dim=2048,
        active_subspace_dim=1024,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    kv_pool = StaticPagedKVPool(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        page_size=16,
        max_total_pages=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    manager = ElasticMemoryBudgetManager(
        expert_cache=expert_cache,
        kv_pool=kv_pool,
        min_expert_slots=4,
        max_expert_slots=32,
        min_kv_pages=32,
        max_kv_pages=256,
        target_kv_headroom_ratio=0.2,
    )

    # 1. Low context length (e.g. 256 tokens -> ~16 pages needed, min is 32)
    # Manager contracts KV pages to 32 and expands expert slots
    res_low = manager.evaluate_and_rebalance(current_active_tokens=256)
    assert res_low["active_expert_slots"] >= 4
    assert res_low["active_kv_pages"] >= 32

    # 2. Burst high context length (e.g. 2000 tokens -> ~125 pages needed)
    # Manager expands KV pages and contracts expert slots
    res_high = manager.evaluate_and_rebalance(current_active_tokens=2000)
    assert res_high["rebalanced"] is True
    assert "expand_kv" in res_high["action"]
    assert res_high["active_kv_pages"] >= 125
    assert res_high["active_expert_slots"] <= 32


def test_expert_cache_resize_active_slots_eviction():
    cache = GPULRUExpertCache(
        num_slots=8,
        hidden_dim=512,
        active_subspace_dim=256,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Fill 6 slots
    for i in range(6):
        cache.allocate_or_evict_slot(layer_idx=0, expert_idx=i)

    assert cache.used_slots == 6

    # Shrink active capacity to 4 slots -> should evict LRU 2 slots
    new_cap = cache.resize_active_slots(4)
    assert new_cap == 4
    assert cache.used_slots == 4
    assert cache.active_slots == 4



def test_static_paged_kv_pool_adjust_capacity():
    kv_pool = StaticPagedKVPool(
        num_layers=2,
        num_heads=4,
        head_dim=32,
        page_size=16,
        max_total_pages=64,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Allocate 10 pages for req_1
    pages = kv_pool.allocate_pages("req_1", 10)
    assert len(pages) == 10

    # Adjust capacity to 30 pages
    new_cap = kv_pool.adjust_active_capacity(30)
    assert new_cap == 30
    assert kv_pool.active_max_pages == 30
    stats = kv_pool.get_stats()
    assert stats["total_pages"] == 30
    assert stats["used_pages"] == 10
    assert stats["free_pages"] == 20
