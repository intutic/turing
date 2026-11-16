"""
Benchmark index for Turing Engine: points to the real-weight measurements and times a few HPC primitives.
1. Quality: scripts/eval_kv_quality.py; GSM8K and needle-in-a-haystack: `turing eval-accuracy`, `turing eval-niah`
2. Throughput: scripts/benchmark_kv_decode.py
3. HPC primitive micro-timings (run here)
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)

import os
import sys
import time
import json
from typing import Dict, Any, List
import numpy as np
import torch

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turing.core.matrix_pow import LogarithmicRecurrenceEngine
from turing.core.swarm_objectives import evaluate_objective
from turing.core.router_annealer import compute_exponential_decay

def run_all_frontier_benchmarks():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 88)
    print("   ⚡ TURING ENGINE FRONTIER COMPREHENSIVE BENCHMARKING ENGINE")
    print("=" * 88)
    print(f"[*] Execution Platform  : {device} ({'NVIDIA GPU' if device.type == 'cuda' else 'Apple Silicon Metal' if device.type == 'mps' else 'CPU SIMD'})")

    print("=" * 88)

    # -------------------------------------------------------------------------
    # 3. Downstream Quality & Accuracy Benchmark Evaluation
    # -------------------------------------------------------------------------
    print("\n[🧠 1/3] QUALITY: not run here. Measure it on real weights with scripts/eval_kv_quality.py")
    print("  (paired NLL against the exact model; results in results_kv_quality_*.json).")
    print("  GSM8K exact match: `turing eval-accuracy --model <id>`; pass-key retrieval: `turing eval-niah --model <id>`.")

    # -------------------------------------------------------------------------
    # 4. Serving throughput
    # -------------------------------------------------------------------------
    print("\n[⚡ 2/3] THROUGHPUT: not run here. Measure it on real weights with scripts/benchmark_kv_decode.py")
    print("  (tokens/s by batch and context against HuggingFace transformers; results in results_kv_decode_*.json).")
    print("  No other engine is run by this repository, so no comparison with vLLM, TensorRT-LLM, SGLang or Ollama is claimed.")

    # -------------------------------------------------------------------------
    # 5. HPC Primitives Verification (High-Performance Systems & Math Integration Benchmarks)
    # -------------------------------------------------------------------------
    print("\n[🔬 3/3] PROFILING High-Performance Systems & Math NATIVE HPC KERNELS...")
    # 1. Logarithmic matrix power jump-ahead
    rec_engine = LogarithmicRecurrenceEngine(101, 202, 3, 7, 11, 1000000007)
    t0 = time.perf_counter()
    k_val = rec_engine.jump_ahead(1000000) # 1 Million step jump
    t_jump_us = (time.perf_counter() - t0) * 1e6
    print(f"  • O(log K) Matrix Recurrence Jump-Ahead (1,000,000 steps) : {t_jump_us:.3f} microseconds")

    # 2. Multi-Modal PSO Objectives
    x_test = np.random.uniform(-5.0, 5.0, size=32)
    t0 = time.perf_counter()
    ackley_val = evaluate_objective("ackley", x_test)
    rastrigin_val = evaluate_objective("rastrigin", x_test)
    t_obj_us = (time.perf_counter() - t0) * 1e6
    print(f"  • C++ Fused Multi-Modal PSO Landscape Evaluation         : {t_obj_us:.3f} microseconds")

    # 3. Router Annealing Schedule
    temp_decay = compute_exponential_decay(2.0, 0.1, 50, 100)
    print(f"  • Exponential Router Annealing Temperature at 50% Step   : {temp_decay:.4f} (init=2.0, min=0.1)")

    print("\n" + "=" * 88)
    print("   ✅ DONE")
    print("=" * 88)

if __name__ == "__main__":
    run_all_frontier_benchmarks()
