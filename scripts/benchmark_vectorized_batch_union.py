#!/usr/bin/env python3
"""
Benchmark: Vectorized Batch-Union Accumulation (Colibrì Paradigm vs Python Loop).
Measures:
1. Multi-token encoder prefill throughput (tokens/sec) across prompt lengths (128 to 4096).
2. Batched speculative draft decode verification throughput across batch sizes (1 to 64).
3. Speedup multiplier of vectorized C++ index_add_ vs sequential Python per-token loop.
"""

import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from turing.config import ModelConfig
from turing.core.heterogeneous_moe import (
    BandwidthAdaptiveDecider,
    HostExpertBank,
    HeterogeneousMoERunner,
)

def benchmark_batch_union(device_str="cpu"):
    device = torch.device(device_str)
    is_cuda = (device.type == "cuda")

    hidden_dim = 128
    ffn_dim = 512
    active_dim = 256
    num_experts = 32
    top_k = 4

    config = ModelConfig(
        name="benchmark_moe",
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        num_heads=8,
        num_kv_heads=4,
        head_dim=16,
        num_layers=4,
        vocab_size=1000
    )

    host_bank = HostExpertBank(
        num_layers=4,
        num_experts=num_experts,
        hidden_dim=hidden_dim,
        ffn_dim=ffn_dim,
        active_subspace_dim=active_dim
    )

    decider = BandwidthAdaptiveDecider(
        device=device,
        pcie_bandwidth_gb_s=25.0,
        cpu_throughput_gflops=100.0,
        gpu_throughput_gflops=1000.0
    )

    runner = HeterogeneousMoERunner(
        config=config,
        host_bank=host_bank,
        decider=decider,
        device=device
    )

    print("\n" + "=" * 80)
    print("   [4/5] BENCHMARK: VECTORIZED BATCH-UNION ACCUMULATION")
    print(f"   Hardware Target: {device_str.upper()} | Hidden: {hidden_dim} | Experts: {num_experts}")
    print("=" * 80)

    # 1. Prefill Scaling Matrix across Prompt Lengths
    prefill_lengths = [128, 512, 1024, 2048]
    prefill_results = []

    for seq_len in prefill_lengths:
        batch = 1
        x = torch.randn(batch, seq_len, hidden_dim, device=device)
        logits = torch.randn(batch, seq_len, num_experts, device=device)

        # Warmup
        _ = runner.route_and_execute(x, logits, layer_idx=0, top_k=top_k)
        if is_cuda: torch.cuda.synchronize()

        # Vectorized Batch-Union Timing
        iters = 10 if seq_len <= 512 else 5
        t0 = time.perf_counter()
        for _ in range(iters):
            out, stats = runner.route_and_execute(x, logits, layer_idx=0, top_k=top_k)
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        vec_ms = ((t1 - t0) / iters) * 1000.0
        vec_tok_s = (seq_len * 1000.0) / max(vec_ms, 1e-6)

        # Baseline: Un-vectorized Python loop simulation
        topk_probs, topk_indices = torch.topk(F.softmax(logits.view(-1, num_experts), dim=-1), k=top_k, dim=-1)
        x_flat = x.view(-1, hidden_dim)
        unique_experts = torch.unique(topk_indices).tolist()
        topk_indices_list = topk_indices.tolist()
        topk_probs_list = topk_probs.tolist()

        t0 = time.perf_counter()
        for _ in range(iters):
            final_output = torch.zeros_like(x_flat)
            for exp_id in unique_experts:
                mask = (topk_indices == exp_id).any(dim=-1)
                token_sub_indices = torch.nonzero(mask).squeeze(-1)
                if len(token_sub_indices) == 0: continue
                sub_x = x_flat[token_sub_indices]
                expert_out = runner.forward_expert_gpu(sub_x, 0, exp_id) if is_cuda else runner.forward_expert_cpu(sub_x, 0, exp_id)
                # Python loop
                tok_idx_list = token_sub_indices.tolist()
                for i, tok_idx in enumerate(tok_idx_list):
                    ranks = [r for r, e in enumerate(topk_indices_list[tok_idx]) if e == exp_id]
                    weight = topk_probs_list[tok_idx][ranks[0]]
                    final_output[tok_idx] += weight * expert_out[i]
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()

        loop_ms = ((t1 - t0) / iters) * 1000.0
        speedup = loop_ms / max(vec_ms, 1e-6)

        print(f"    • Prefill N={seq_len:4d}: Vectorized {vec_ms:.2f} ms ({vec_tok_s:.0f} tok/s) vs Loop {loop_ms:.2f} ms | Speedup: {speedup:.2f}x", flush=True)
        prefill_results.append({
            "seq_len": seq_len,
            "vectorized_ms": round(vec_ms, 2),
            "loop_ms": round(loop_ms, 2),
            "tokens_per_sec": round(vec_tok_s, 0),
            "speedup_x": round(speedup, 2)
        })


    # 2. Batched Speculative Decode Verification Matrix
    batch_sizes = [4, 16, 32, 64]
    batch_results = []
    for b in batch_sizes:
        seq_len = 1
        x = torch.randn(b, seq_len, hidden_dim, device=device)
        logits = torch.randn(b, seq_len, num_experts, device=device)

        iters = 50
        t0 = time.perf_counter()
        for _ in range(iters):
            out, stats = runner.route_and_execute(x, logits, layer_idx=0, top_k=top_k)
        if is_cuda: torch.cuda.synchronize()
        t1 = time.perf_counter()
        vec_ms = ((t1 - t0) / iters) * 1000.0

        batch_results.append({
            "batch_size": b,
            "latency_ms": round(vec_ms, 3),
            "throughput_tok_s": round((b * 1000.0) / max(vec_ms, 1e-6), 0)
        })
        print(f"    • Batched Spec B={b:2d}: Latency {vec_ms:.3f} ms | Throughput: {batch_results[-1]['throughput_tok_s']:.0f} tok/s")

    return {
        "suite": "vectorized_batch_union",
        "prefill_matrix": prefill_results,
        "batched_spec_matrix": batch_results,
        "max_speedup_x": max(r["speedup_x"] for r in prefill_results)
    }

if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if len(sys.argv) > 1:
        dev = sys.argv[1]
    res = benchmark_batch_union(dev)
    print(json.dumps(res, indent=2))
