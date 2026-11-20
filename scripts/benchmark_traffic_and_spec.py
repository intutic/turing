#!/usr/bin/env python3
"""
Benchmark AI traffic management and concurrency-adaptive speculation gating.

Features:
1. Token-budget routing & VRAM admission control overhead measurement.
2. Speculation gate decisions per concurrency (policy only; no throughput is measured).
3. 3-lane QoS scheduling verification.
4. Self-check of the parity verifier on fixed token lists (no spec or plain decode is run).
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)

import argparse
import json
import time
import sys
import os
from typing import Dict, Any

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turing.serving.traffic import (
    KVMemoryEstimator,
    PrefixHashRouter,
    AdmissionController,
    LanePolicy,
    Lane,
)
from turing.serving.spec_gate import (
    SpeculationGatePolicy,
    SpecExactParityVerifier,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark traffic management and spec gating.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto",
                        help="Ignored: these benchmarks are pure Python and run on the host CPU.")
    parser.add_argument("--concurrency_max", type=int, default=16,
                        help="Maximum active concurrency for spec gating.")
    parser.add_argument("--json_out", type=str, default="benchmarks/results/results_traffic_spec.json",
                        help="JSON output path.")
    return parser.parse_args()

def benchmark_part1_admission_overhead() -> Dict[str, Any]:
    print("\n--- Part 1: KV Memory Estimation & Admission Overhead ---")
    results = {}
    
    # KV Memory Estimator tests
    seq_lengths = [128, 512, 2048, 8192, 32768]
    kv_estimates = {}
    for seq_len in seq_lengths:
        bytes_req = KVMemoryEstimator.estimate_kv_bytes(
            num_prompt_tokens=seq_len,
            max_new_tokens=0,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_bytes=2
        )
        kv_estimates[seq_len] = bytes_req
    results["kv_estimates_bytes"] = kv_estimates
    
    # Admission Controller Benchmark (10,000 decisions)
    vram_budget = 40 * 1024**3 # 40GB
    controller = AdmissionController(vram_budget_bytes=vram_budget)
    
    start_time = time.perf_counter()
    iterations = 10000
    for i in range(iterations):
        controller.admit(f"req_{i}", 1024 * 1024)
        if i % 10 == 0:
            controller.release(f"req_{i-10}")
            
    end_time = time.perf_counter()
    duration = end_time - start_time
    us_per_decision = (duration / iterations) * 1e6
    results["admission_overhead_us"] = us_per_decision
    print(f"Admission decision overhead: {us_per_decision:.3f} us (req < 50us)")
    
    # 3-Lane QoS Policy overhead
    lane_policy = LanePolicy()
    start_time = time.perf_counter()
    for i in range(iterations):
        lane_policy.classify_request(max_tokens=(i % 2000))
    end_time = time.perf_counter()
    qos_us = ((end_time - start_time) / iterations) * 1e6
    results["qos_overhead_us"] = qos_us
    print(f"Lane classification overhead: {qos_us:.3f} us")
    
    return results

def benchmark_part2_spec_gating_envelope(max_c: int) -> Dict[str, Any]:
    # Only the gate's decision is computed here; no model runs, so no throughput is reported.
    print("\n--- Part 2: Speculation Gate Decisions by Concurrency (policy only, no decoding) ---")
    results = {"concurrencies": []}
    policy = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    print(f"{'Concurrency':<12} | {'Gate decision':<14}")
    print("-" * 30)
    for c in [1, 2, 3, 4, 8, 12, 16]:
        if c > max_c:
            break
        mode = policy.gate_decision(c)
        print(f"{c:<12} | {mode.value:<14}")
        results["concurrencies"].append({"c": c, "mode": mode.value})
    return results

def benchmark_part3_parity_verifier() -> Dict[str, Any]:
    print("\n--- Part 3: Parity Verifier Self-Check (fixed token lists; no decoding is run) ---")
    
    # Exact Match
    spec_tokens = [1, 5, 23, 99, 102]
    plain_tokens = [1, 5, 23, 99, 102]
    
    report1 = SpecExactParityVerifier.verify_greedy_parity(spec_tokens, plain_tokens)
    print(f"Pass Exact Match: {report1.passed}")
    
    # Divergence
    spec_tokens_div = [1, 5, 23, 98, 102]
    report2 = SpecExactParityVerifier.verify_greedy_parity(spec_tokens_div, plain_tokens)
    print(f"Pass Divergence (expected False): {report2.passed}, Diverged at idx: {report2.divergence_index}")
    
    return {
        "exact_match_pass": report1.passed,
        "divergence_pass": report2.passed,
        "divergence_index": report2.divergence_index
    }

def main():
    args = parse_args()
    print(f"Starting Benchmark. Device: {args.device}, Max Concurrency: {args.concurrency_max}")
    
    out_data = {}
    out_data["part1"] = benchmark_part1_admission_overhead()
    out_data["part2"] = benchmark_part2_spec_gating_envelope(args.concurrency_max)
    out_data["part3"] = benchmark_part3_parity_verifier()
    
    with open(args.json_out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nResults saved to {args.json_out}")

if __name__ == "__main__":
    main()
