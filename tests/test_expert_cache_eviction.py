"""Regression tests for GPULRUExpertCache slot accounting (two keys must never share a slot)."""
import pytest
import torch

from turing.core.expert_cache import GPULRUExpertCache


def _cache(num_slots, policy):
    return GPULRUExpertCache(num_slots=num_slots, hidden_dim=8, active_subspace_dim=4,
                             device=torch.device("cpu"), eviction_policy=policy)


def _assert_slots_consistent(cache):
    slots = list(cache.cache_map.values())
    assert len(slots) == len(set(slots)), f"two keys share a slot: {dict(cache.cache_map)}"
    assert len(cache.cache_map) <= cache.num_slots


def test_lfru_hysteresis_rejection_does_not_share_a_slot():
    cache = _cache(2, "lfru")
    cache.allocate_or_evict_slot(0, 1)
    cache.allocate_or_evict_slot(0, 2)
    for _ in range(10):
        cache.get_slot(0, 1)
        cache.get_slot(0, 2)
    cache.allocate_or_evict_slot(0, 3)  # new key: admitted, evicts one hot key
    hot = next(k for k in [(0, 1), (0, 2)] if not cache.contains(*k))
    cache.allocate_or_evict_slot(*hot)  # hot key returns: beats cold (0, 3), evicts it
    assert not cache.contains(0, 3)
    _assert_slots_consistent(cache)

    # (0, 3) has history but low heat, so the gate rejects it: it must not be cached anywhere.
    slot, evicted = cache.allocate_or_evict_slot(0, 3)
    assert slot is None and evicted is None
    assert not cache.contains(0, 3)
    _assert_slots_consistent(cache)


def test_lfru_rejected_key_is_admitted_after_repeated_requests():
    cache = _cache(1, "lfru")
    cache.allocate_or_evict_slot(0, 1)
    cache.allocate_or_evict_slot(0, 2)  # new key, evicts (0, 1); (0, 1) now has history
    admitted = [cache.allocate_or_evict_slot(0, 1)[0] is not None for _ in range(10)]
    assert any(admitted)
    assert cache.contains(0, 1) and not cache.contains(0, 2)
    _assert_slots_consistent(cache)


def test_lru_policy_keeps_evicting_once_full():
    cache = _cache(2, "lru")
    for e in range(2):
        cache.allocate_or_evict_slot(0, e)
    for e in range(2, 10):
        slot, evicted = cache.allocate_or_evict_slot(0, e)
        assert slot is not None and 0 <= slot < 2, f"expert {e} got slot {slot}"
        assert evicted == (0, e - 2)  # plain LRU: the oldest key goes
        assert cache.contains(0, e)
    assert cache.used_slots == 2
