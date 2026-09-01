#!/usr/bin/env python3
"""
Benchmark: LFRU (Least-Frequently-Recently-Used) Eviction & Hysteresis Margin.
Measures:
1. Hit rate under non-uniform Zipfian expert activations.
2. Eviction thrashing and ping-pong reduction from the 25% + 4 hysteresis gate.
"""

import sys
import time
import json
import torch
import numpy as np

from turing.core.expert_cache import GPULRUExpertCache, RoutingHistoryTracker

def generate_bursty_zipf_trace(num_experts=64, total_steps=2000, burst_interval=200):
    """
    Generates an expert access stream with a dominant Zipfian distribution
    interspersed with periodic bursts of cold experts to test thrashing.
    """
    np.random.seed(42)
    ranks = np.arange(1, num_experts + 1)
    probs = 1.0 / (ranks ** 1.3)
    probs = probs / np.sum(probs)

    trace = []
    for step in range(total_steps):
        if (step // burst_interval) % 2 == 1:
            # Cold burst: select from the bottom 50% colder experts
            cold_pool = np.arange(num_experts // 2, num_experts)
            trace.append(int(np.random.choice(cold_pool)))
        else:
            # Normal Zipfian hot access
            trace.append(int(np.random.choice(num_experts, p=probs)))
    return trace

def benchmark_lfru_vs_lru(device_str="cpu"):
    device = torch.device(device_str)
    num_slots = 8
    num_experts = 64
    total_steps = 2000

    print("\n" + "=" * 80)
    print("   [3/5] BENCHMARK: LFRU EVICTION & HYSTERESIS PROMOTION")
    print(f"   Hardware Target: {device_str.upper()} | Slots: {num_slots} | Experts: {num_experts}")
    print("=" * 80)

    trace = generate_bursty_zipf_trace(num_experts=num_experts, total_steps=total_steps)

    # 1. Plain LRU Cache
    cache_lru = GPULRUExpertCache(num_slots=num_slots, hidden_dim=128, active_subspace_dim=64, device=device, eviction_policy="lru")
    lru_evictions = 0
    t0 = time.perf_counter()
    for exp in trace:
        slot = cache_lru.get_slot(0, exp)
        if slot is None:
            _, evicted = cache_lru.allocate_or_evict_slot(0, exp)
            if evicted is not None:
                lru_evictions += 1
    t1 = time.perf_counter()
    lru_time_ms = (t1 - t0) * 1000.0
    lru_hit_rate = cache_lru.hit_rate

    # 2. LFRU Cache with Hysteresis Promotion Gate
    tracker = RoutingHistoryTracker(num_layers=1, num_experts=num_experts)
    cache_lfru = GPULRUExpertCache(
        num_slots=num_slots,
        hidden_dim=128,
        active_subspace_dim=64,
        device=device,
        eviction_policy="lfru",
        history_tracker=tracker
    )
    lfru_evictions = 0
    t0 = time.perf_counter()
    for exp in trace:
        slot = cache_lfru.get_slot(0, exp)
        if slot is None:
            _, evicted = cache_lfru.allocate_or_evict_slot(0, exp)
            if evicted is not None:
                lfru_evictions += 1
    t1 = time.perf_counter()
    lfru_time_ms = (t1 - t0) * 1000.0
    lfru_hit_rate = cache_lfru.hit_rate

    hit_rate_gain = lfru_hit_rate - lru_hit_rate
    eviction_reduction_pct = ((lru_evictions - lfru_evictions) / max(lru_evictions, 1)) * 100.0

    print(f"    • Plain LRU Hit Rate         : {lru_hit_rate:.2f}% (Evictions: {lru_evictions})")
    print(f"    • LFRU + Hysteresis Hit Rate : {lfru_hit_rate:.2f}% (Evictions: {lfru_evictions})")
    print(f"    • Hit Rate Gain              : +{hit_rate_gain:.2f}%")
    print(f"    • Cache Thrashing Reduction  : {eviction_reduction_pct:.2f}% fewer evictions")

    return {
        "suite": "lfru_eviction_hysteresis",
        "lru_hit_rate_pct": round(lru_hit_rate, 2),
        "lfru_hit_rate_pct": round(lfru_hit_rate, 2),
        "hit_rate_gain_pct": round(hit_rate_gain, 2),
        "lru_evictions": lru_evictions,
        "lfru_evictions": lfru_evictions,
        "thrashing_reduction_pct": round(eviction_reduction_pct, 2)
    }

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    res = benchmark_lfru_vs_lru(dev)
    print(json.dumps(res, indent=2))
