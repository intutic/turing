#!/usr/bin/env python3
"""
Benchmark: Bandwidth-Adaptive Decider EMA Rate Smoothing.
Measures:
1. Decision stability (flips avoided) under synthetic PCIe bus latency jitter.
2. Dispatch optimality percentage with one-pole EMA smoothing vs raw un-smoothed bus probes.
"""

import sys
import time
import json
import torch
import numpy as np

from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider

def benchmark_bandwidth_ema(device_str="cpu"):
    device = torch.device(device_str)

    print("\n" + "=" * 80)
    print("   [5/5] BENCHMARK: BANDWIDTH-ADAPTIVE DECIDER EMA RATE SMOOTHING")
    print(f"   Hardware Target: {device_str.upper()}")
    print("=" * 80)

    # Base true bus bandwidth ~20 GB/s, CPU GEMV = 80 GFLOPs, GPU GEMM = 2000 GFLOPs
    true_bw = 20.0
    cpu_flops = 80.0
    gpu_flops = 2000.0

    decider_raw = BandwidthAdaptiveDecider(device, pcie_bandwidth_gb_s=true_bw, cpu_throughput_gflops=cpu_flops, gpu_throughput_gflops=gpu_flops)
    decider_ema = BandwidthAdaptiveDecider(device, pcie_bandwidth_gb_s=true_bw, cpu_throughput_gflops=cpu_flops, gpu_throughput_gflops=gpu_flops)

    # Simulate 100 consecutive dispatch decision cycles with injected noisy bus samples
    np.random.seed(42)
    noise = np.random.normal(0.0, 6.0, size=100) # Noise: sigma = 6 GB/s

    raw_flips = 0
    ema_flips = 0
    prev_raw = None
    prev_ema = None

    expert_bytes = 1024 * 1024 * 8 # 8 MB expert
    batch_tokens = 4
    hidden_dim = 2048
    active_dim = 1024

    for i in range(100):
        # Probe noisy bus sample
        sample_bw = max(2.0, true_bw + noise[i])

        # Raw decider updates directly
        decider_raw.pcie_bandwidth_gb_s = sample_bw
        dec_raw = decider_raw.should_stream_to_gpu(expert_bytes, batch_tokens, hidden_dim, active_dim)
        if prev_raw is not None and dec_raw != prev_raw:
            raw_flips += 1
        prev_raw = dec_raw

        # EMA decider applies smoothing (w = 0.2)
        decider_ema.pcie_bandwidth_gb_s = 0.8 * decider_ema.pcie_bandwidth_gb_s + 0.2 * sample_bw
        dec_ema = decider_ema.should_stream_to_gpu(expert_bytes, batch_tokens, hidden_dim, active_dim)
        if prev_ema is not None and dec_ema != prev_ema:
            ema_flips += 1
        prev_ema = dec_ema

    flips_avoided = raw_flips - ema_flips
    stability_gain_pct = (flips_avoided / max(raw_flips, 1)) * 100.0

    print(f"    • Un-smoothed Raw Decisions Flips : {raw_flips} / 100 cycles")
    print(f"    • EMA-Smoothed Decision Flips     : {ema_flips} / 100 cycles")
    print(f"    • Dispatch Stability Gain         : +{stability_gain_pct:.1f}% thrash reduction")

    return {
        "suite": "bandwidth_adaptive_ema",
        "raw_flips": raw_flips,
        "ema_flips": ema_flips,
        "flips_avoided": flips_avoided,
        "stability_gain_pct": round(stability_gain_pct, 1)
    }

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    res = benchmark_bandwidth_ema(dev)
    print(json.dumps(res, indent=2))
