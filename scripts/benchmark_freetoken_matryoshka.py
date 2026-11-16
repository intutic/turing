"""
Empirical Benchmark: Matryoshka Parameter Slicing & FreeToken Elastic Serving on Turing Engine.
Measures:
1. Matryoshka Sliced Draft Head Latency & Throughput across nested widths (1024, 2048, 4096, 8192).
2. Elastic Memory Dynamic Budget Rebalancing under context expansion.
3. Bandwidth-Adaptive q* Bus Streaming Decisions.
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F

from turing.core.speculation import MatryoshkaDraftHead, QuadtreeMRPSpeculator

from turing.core.expert_cache import GPULRUExpertCache
from turing.core.paging import StaticPagedKVPool
from turing.core.elastic_memory import ElasticMemoryBudgetManager
from turing.core.heterogeneous_moe import BandwidthAdaptiveDecider


def benchmark_matryoshka_draft_head(device: torch.device, hidden_dim: int = 8192, vocab_size: int = 32000):
    print("\n" + "=" * 80)
    print(f"📊 1. MATRYOSHKA PARAMETER-SLICED DRAFT HEAD BENCHMARK ({device})")
    print("=" * 80)

    slice_widths = [1024, 2048, 4096, 8192]
    head = MatryoshkaDraftHead(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        slice_widths=slice_widths,
        bias=False,
    ).to(device)

    batch_size = 4
    x = torch.randn(batch_size, hidden_dim, device=device)

    # Warmup
    for _ in range(10):
        _ = head(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    results = {}
    base_latency = None

    for w in reversed(slice_widths):
        iters = 50
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = head(x, slice_width=w)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        elapsed_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        if base_latency is None:
            base_latency = elapsed_ms
        speedup = base_latency / max(1e-6, elapsed_ms)

        # Verification fidelity against full rank logits
        logits_full = head(x)
        logits_w = head(x, slice_width=w)
        sim = F.cosine_similarity(logits_full.float(), logits_w.float(), dim=-1).mean().item()

        results[w] = {
            "latency_ms": round(elapsed_ms, 3),
            "speedup": f"{speedup:.2f}x",
            "cosine_fidelity": f"{sim * 100:.2f}%",
        }

        print(
            f"  • Hidden Width W={w:4d}: Latency = {elapsed_ms:6.3f} ms | Speedup = {speedup:5.2f}x | Fidelity = {sim*100:6.2f}%"
        )

    return results


def benchmark_elastic_memory(device: torch.device):
    print("\n" + "=" * 80)
    print(f"🔄 3. ELASTIC MEMORY DYNAMIC REBALANCING ({device})")
    print("=" * 80)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    expert_cache = GPULRUExpertCache(
        num_slots=32,
        hidden_dim=2048,
        active_subspace_dim=1024,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    kv_pool = StaticPagedKVPool(
        num_layers=8,
        num_heads=8,
        head_dim=64,
        page_size=16,
        max_total_pages=512,
        device=device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )

    manager = ElasticMemoryBudgetManager(
        expert_cache=expert_cache,
        kv_pool=kv_pool,
        min_expert_slots=4,
        max_expert_slots=32,
        min_kv_pages=32,
        max_kv_pages=512,
        target_kv_headroom_ratio=0.2,
    )

    # 1. Initial State
    print(
        f"  • Initial State: Expert Slots = {expert_cache.active_slots} | KV Pages = {kv_pool.active_max_pages}"
    )

    # 2. Context Burst to 4,096 tokens (~256 pages)
    t0 = time.perf_counter()
    rebalance_res = manager.evaluate_and_rebalance(current_active_tokens=4096)
    elapsed_us = (time.perf_counter() - t0) * 1e6

    print(
        f"  • Context Burst (4K tokens) -> Action: {rebalance_res['action']} (rebalanced in {elapsed_us:.1f} µs)"
    )
    print(
        f"  • Rebalanced State: Expert Slots = {rebalance_res['active_expert_slots']} | KV Pages = {rebalance_res['active_kv_pages']}"
    )

    # 3. Context Drain to 256 tokens (~16 pages)
    rebalance_drain = manager.evaluate_and_rebalance(current_active_tokens=256)
    print(
        f"  • Context Drain (256 tokens) -> Action: {rebalance_drain['action']}"
    )
    print(
        f"  • Restored State: Expert Slots = {rebalance_drain['active_expert_slots']} | KV Pages = {rebalance_drain['active_kv_pages']}"
    )


    return {
        "rebalance_overhead_us": round(elapsed_us, 1),
        "expanded_kv_pages": rebalance_res["active_kv_pages"],
        "contracted_slots": rebalance_res["active_expert_slots"],
    }


def benchmark_bandwidth_adaptive_qstar(device: torch.device):
    print("\n" + "=" * 80)
    print(f"⚡ 4. CLOSED-LOOP q* HARDWARE BUS PROFILER ({device})")
    print("=" * 80)

    decider = BandwidthAdaptiveDecider(device=device)
    telemetry = decider.get_telemetry()

    print(f"  • Target Device: {telemetry['device']}")
    print(f"  • Calibrated Bus Bandwidth: {telemetry['pcie_bandwidth_gb_s']} GB/s")
    print(f"  • CPU Throughput: {telemetry['cpu_throughput_gflops']} GFLOPs")
    print(f"  • GPU Throughput: {telemetry['gpu_throughput_gflops']} GFLOPs")

    # Decision sample for 35B MoE layer
    expert_bytes = 4 * 1024 * 1024  # 4MB packed INT4
    should_stream = decider.should_stream_to_gpu(
        expert_bytes=expert_bytes,
        batch_tokens=8,
        hidden_dim=4096,
        moe_intermediate_dim=2048,
    )
    print(
        f"  • q* Streaming Decision (Batch=8 tokens): {'STREAM TO GPU' if should_stream else 'COMPUTE ON CPU'}"
    )

    return telemetry


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Matryoshka Parameter Slicing & FreeToken Elastic Serving"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"\n🚀 Launching Turing Engine Benchmark on Device: {device}")

    # 1. Matryoshka Draft Head
    m_res = benchmark_matryoshka_draft_head(device)

    # 2. Elastic Memory
    e_res = benchmark_elastic_memory(device)

    # 3. Closed-loop q*
    q_res = benchmark_bandwidth_adaptive_qstar(device)

    print("\n" + "=" * 80)
    print("✅ BENCHMARK COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
