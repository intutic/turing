#!/usr/bin/env python3
"""
Master Colibrì × Turing Engine Empirical Benchmark Orchestrator.
Runs all 5 newly integrated systems benchmark suites:
1. MoE Routing History & Learned Hot-Expert Pinning
2. PILOT One-Layer-Ahead Speculative Prefetching
3. LFRU Eviction Policy & Hysteresis Promotion Gate
4. Vectorized Batch-Union Accumulation (Prefill & Speculative Verify)
5. Bandwidth-Adaptive Decider EMA Rate Smoothing
"""

import os
import sys
import time
import json
import argparse

# Ensure scripts dir and repo root are in path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(scripts_dir, ".."))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch

from benchmark_routing_history_and_pinning import benchmark_routing_and_pinning
from benchmark_pilot_expert_prefetch import benchmark_pilot_prefetch
from benchmark_lfru_eviction_hysteresis import benchmark_lfru_vs_lru
from benchmark_vectorized_batch_union import benchmark_batch_union
from benchmark_bandwidth_adaptive_ema import benchmark_bandwidth_ema

def get_platform_telemetry(device_str):
    info = {
        "platform": sys.platform,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "device": device_str,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    }
    if device_str == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = torch.cuda.get_device_capability(0)
        info["total_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    elif device_str == "mps":
        info["device_type"] = "Apple Silicon Metal Performance Shaders"
    else:
        info["device_type"] = "Host CPU SIMD"
    return info

def main():
    parser = argparse.ArgumentParser(description="Turing Engine Colibri Master Benchmark Runner")
    parser.add_argument("--device", type=str, default=None, help="Target compute device: cuda, mps, or cpu")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON benchmark output")
    args = parser.parse_args()

    # Determine default device
    if args.device is not None:
        device_str = args.device
    elif torch.cuda.is_available():
        device_str = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"

    print("=" * 80)
    print("   ⚡ TURING ENGINE: COLIBRÌ MASTER EMPIRICAL BENCHMARK SUITE")
    print(f"   Target Device: {device_str.upper()}")
    print("=" * 80)

    telemetry = get_platform_telemetry(device_str)
    for k, v in telemetry.items():
        print(f"   • {k:20s}: {v}")
    print("=" * 80)

    t_start = time.perf_counter()

    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "telemetry": telemetry,
        "suites": {}
    }

    # Suite 1: Routing History & Pinning
    r1 = benchmark_routing_and_pinning(device_str)
    results["suites"]["routing_history_and_pinning"] = r1

    # Suite 2: PILOT Speculative Prefetch
    r2 = benchmark_pilot_prefetch(device_str)
    results["suites"]["pilot_expert_prefetch"] = r2

    # Suite 3: LFRU Eviction & Hysteresis
    r3 = benchmark_lfru_vs_lru(device_str)
    results["suites"]["lfru_eviction_hysteresis"] = r3

    # Suite 4: Vectorized Batch-Union Accumulation
    r4 = benchmark_batch_union(device_str)
    results["suites"]["vectorized_batch_union"] = r4

    # Suite 5: Bandwidth-Adaptive EMA Calibration
    r5 = benchmark_bandwidth_ema(device_str)
    results["suites"]["bandwidth_adaptive_ema"] = r5

    t_end = time.perf_counter()
    total_duration_s = t_end - t_start
    results["total_benchmark_duration_seconds"] = round(total_duration_s, 2)

    print("\n" + "=" * 80)
    print(f"   🏆 ALL COLIBRÌ BENCHMARKS COMPLETED IN {total_duration_s:.2f}s")
    print("=" * 80)
    print(f"   1. Routing Pinning Hit Gain     : +{r1['hit_rate_improvement_pct']}% (Turn 0: +{r1['warm_initial_hit_rate_pct'] - r1['cold_initial_hit_rate_pct']}%)")
    print(f"   2. PILOT Prefetch DMA Hiding    : {r2['latency_hiding_pct']}% hidden (Speedup: {r2['speedup_x']}x)")
    print(f"   3. LFRU Thrashing Reduction     : {r3['thrashing_reduction_pct']}% fewer evictions")
    print(f"   4. Vectorized Batch-Union Speed : Up to {r4['max_speedup_x']}x vs Python loop")
    print(f"   5. Dispatch Stability Gain      : +{r5['stability_gain_pct']}% thrash reduction")
    print("=" * 80)

    if args.output:
        out_path = args.output
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[*] Benchmark results written to {out_path}")

    return results

if __name__ == "__main__":
    main()
