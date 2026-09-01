import pytest
import turing.turing_csrc as turing_csrc


def test_native_lfru_cache_initialization():
    """Verify initialization and slot availability."""
    cache = turing_csrc.LFRUExpertCacheFast(num_slots=16, hysteresis_threshold=0.25, hysteresis_min_access=4)
    assert cache.num_slots == 16
    assert cache.used_slots == 0
    assert cache.hits == 0
    assert cache.misses == 0


def test_native_lfru_cache_allocation_and_hits():
    """Verify slot allocation, hit tracking, and clock updates."""
    cache = turing_csrc.LFRUExpertCacheFast(num_slots=4)
    
    # Miss on cold key
    assert not cache.contains(0, 5)
    slot, ev_l, ev_e = cache.allocate_or_evict_slot(0, 5, candidate_heat=1)
    assert slot >= 0
    assert ev_l == -1 and ev_e == -1
    assert cache.contains(0, 5)

    # Hit on warm key
    hit_slot = cache.get_slot(0, 5)
    assert hit_slot == slot
    assert cache.hits == 1
    assert cache.get_slot_heat(slot) == 2


def test_native_lfru_cache_pinning():
    """Verify pinned keys are not evicted."""
    cache = turing_csrc.LFRUExpertCacheFast(num_slots=2)
    
    # Fill cache with 2 experts
    s0, _, _ = cache.allocate_or_evict_slot(0, 10, candidate_heat=10)
    s1, _, _ = cache.allocate_or_evict_slot(0, 20, candidate_heat=1)

    # Pin expert (0, 20) even though heat is lower
    cache.pin_key(0, 20)

    # Allocate a 3rd expert with force_promote
    s_new, ev_l, ev_e = cache.allocate_or_evict_slot(0, 30, candidate_heat=50, force_promote=True)
    assert s_new == s0  # Must evict (0, 10), preserving pinned (0, 20)
    assert ev_l == 0 and ev_e == 10
    assert cache.contains(0, 20)  # Pinned key remains intact


def test_native_lfru_cache_hysteresis_rejection():
    """Verify cold candidates are rejected when victim heat exceeds hysteresis threshold."""
    cache = turing_csrc.LFRUExpertCacheFast(num_slots=2, hysteresis_threshold=0.5, hysteresis_min_access=4)
    
    cache.allocate_or_evict_slot(0, 1, candidate_heat=20)
    cache.allocate_or_evict_slot(0, 2, candidate_heat=20)

    # Cold candidate with heat=2 tries to evict without force_promote
    s_rejected, _, _ = cache.allocate_or_evict_slot(0, 3, candidate_heat=2, force_promote=False)
    assert s_rejected == -1  # Rejected by hysteresis gate
