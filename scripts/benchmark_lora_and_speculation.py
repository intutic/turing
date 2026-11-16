"""
Empirical Benchmark Suite: Speculative Drafting (EAGLE-3/DFlash/DSpark), Multi-Tenant LoRA Cache, and Pipelined Cold-Starts.
All inputs are synthetic: random-init adapters and draft head, randn activations, unseeded 80/20 tenant traffic,
constructed target logits, and random/empty weights for the cold-start path (no checkpoint is read).
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turing.config import ModelConfig
from turing.core.speculation import SubspaceEAGLEDraftHead, EntropyConfidenceTreePruner, RidgeAssistedTreeSpeculator
from turing.models.adapters import GPULRUAdapterCache
from turing.models.streaming_loader import PipelinedSubspaceWarmupLoader

def benchmark_multi_tenant_lora_cache(device: str = "cpu", num_tenants: int = 100, cache_capacity: int = 32, num_requests: int = 1000):
    print("\n" + "=" * 80)
    print("  ⚡ BENCHMARK 1: MULTI-TENANT LoRA HOT-SWAP & LRU CACHE (100 ADAPTER POOL)")
    print("=" * 80)

    hidden_dim = 2048
    rank = 8
    dev = torch.device(device)

    cache = GPULRUAdapterCache(
        hidden_dim=hidden_dim,
        rank=rank,
        capacity=cache_capacity,
        device=dev
    )

    # 1. Register 100 tenant adapters in pinned host memory
    for i in range(num_tenants):
        cache.register_host_adapter(f"tenant_{i:03d}")

    # 2. Simulate traffic (80/20 zipfian distribution over 100 tenants)
    popular_tenants = [f"tenant_{i:03d}" for i in range(10)]
    long_tail_tenants = [f"tenant_{i:03d}" for i in range(10, num_tenants)]

    requests = []
    for _ in range(num_requests):
        if np.random.rand() < 0.80:
            requests.append(np.random.choice(popular_tenants))
        else:
            requests.append(np.random.choice(long_tail_tenants))

    # Benchmark hit vs miss latency
    hit_latencies_us = []
    miss_latencies_us = []
    dummy_x = torch.randn(1, 128, hidden_dim, device=dev)

    # Warmup cache with first request
    _ = cache.apply_adapter(dummy_x, requests[0])

    t_start = time.perf_counter()
    for req in requests:
        is_hit = req in cache.gpu_slots
        t0 = time.perf_counter()
        _ = cache.apply_adapter(dummy_x, req)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        lat_us = (t1 - t0) * 1_000_000.0

        if is_hit:
            hit_latencies_us.append(lat_us)
        else:
            miss_latencies_us.append(lat_us)

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    host_mb = sum(a.nbytes + b.nbytes for a, b in cache.host_store.values()) / 1e6
    slot_mb = sum(p.numel() * p.element_size() for ad in cache.gpu_slots.values() for p in ad.parameters()) / 1e6
    print(f"  • Total Tenant Pool Size    : {num_tenants} adapters ({host_mb:.1f} MB in host memory)")
    print(f"  • Resident Slots (LRU)      : {len(cache.gpu_slots)}/{cache_capacity} active slots ({slot_mb:.1f} MB on {dev.type})")
    print(f"  • Evaluated Requests        : {num_requests} requests")
    print(f"  • Measured Cache Hit Rate   : {cache.get_hit_rate() * 100.0:.2f}% ({cache.hits} hits, {cache.misses} misses, {cache.evictions} evictions)")
    print(f"  • Cache-Hit Latency (P50)   : {np.percentile(hit_latencies_us, 50):.2f} µs (includes the LoRA forward on a 128x2048 input)")
    print(f"  • Cold-Load Switch (P50)    : {np.percentile(miss_latencies_us, 50) / 1000.0:.3f} ms (synchronous host-to-device copy)")
    print(f"  • Cold-Load Switch (P99)    : {np.percentile(miss_latencies_us, 99) / 1000.0:.3f} ms")
    print(f"  • Overall Routing Throughput : {num_requests / (total_time_ms / 1000.0):.1f} req/sec")


def benchmark_speculative_drafting(device: str = "cpu", iters: int = 100):
    print("\n" + "=" * 80)
    print("  ⚡ BENCHMARK 2: SUBSPACE-EAGLE3 + DFLASH + DSPARK SPECULATIVE DRAFTING")
    print("=" * 80)

    hidden_dim = 4096
    rank_subspace = 64
    vocab_size = 32000
    future_tokens = 8
    dev = torch.device(device)

    head = SubspaceEAGLEDraftHead(
        hidden_dim=hidden_dim,
        rank_subspace=rank_subspace,
        vocab_size=vocab_size,
        future_tokens=future_tokens,
        use_matryoshka=True,
        slice_widths=[16, 32, 64]
    ).to(dev)

    speculator = RidgeAssistedTreeSpeculator()

    dummy_hidden = torch.randn(1, 16, hidden_dim, device=dev)

    # Warmup
    for _ in range(5):
        _ = head(dummy_hidden, slice_width=64)

    latencies_ms = []
    widths = []

    for _ in range(iters):
        t0 = time.perf_counter()
        nodes, dag_mask, token_ids, entropy, width = head(dummy_hidden, slice_width=64)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        widths.append(width)

    # Verifier self-check only: the target logits are constructed so the first 6 drafts match,
    # so the printed count is not an acceptance rate (that needs a real target model).
    target_logits = torch.randn(future_tokens, vocab_size, device=dev)
    for k in range(min(6, len(token_ids))):
        target_logits[k, token_ids[k]] = 50.0

    accepted, num_acc = speculator.verify_speculative_candidates(token_ids, target_logits)

    print(f"  • Hidden State Proj Dim     : {hidden_dim} -> Rank-{rank_subspace} Subspace")
    print(f"  • Draft Feature Synthesizer : 1D-Depthwise Dilated Conv (DFlash O(1) concurrent)")
    print(f"  • Matryoshka Slice Width    : 64 of rank {rank_subspace} (vocab {vocab_size})")
    print(f"  • Dynamic Tree Width (DSpark): Avg {np.mean(widths):.1f} tokens (Entropy-gated)")
    print(f"  • Candidate Latency (P50)   : {np.percentile(latencies_ms, 50):.3f} ms / draft pass")
    print(f"  • Candidate Latency (P99)   : {np.percentile(latencies_ms, 99):.3f} ms")
    print(f"  • Verifier Self-Check       : {len(accepted)} / {len(token_ids)} accepted against constructed logits (not an acceptance rate)")


def benchmark_pipelined_cold_start(device: str = "cpu"):
    print("\n" + "=" * 80)
    print("  ⚡ BENCHMARK 3: PIPELINED ZERO-OVERHEAD CHECKPOINT LOADING & WARMUP")
    print("=" * 80)

    cfg = ModelConfig(
        name="llama-3.1-70b-subspace",
        hidden_dim=8192,
        ffn_dim=28672,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        num_layers=80,
        tile_size=256,
        active_tiles=48,
        rank_sub=64
    )

    loader = PipelinedSubspaceWarmupLoader(
        model_config=cfg,
        device=device,
        warmup_buckets=[1, 4, 16, 64, 256]
    )

    res = loader.pipelined_load_and_warmup()

    print(f"  • Target Model Profile      : {cfg.name} (80 layers, d_model=8192)")
    print(f"  • Ingestion Architecture    : Stage 1 (layers 0..2 random) -> bucket warmup -> Stage 2 (empty buffers)")
    print(f"  • Warmup Batch Buckets     : {res['captured_buckets']}")
    # Synthetic: the loader fills 3 layers with torch.randn and allocates the rest empty; no file is read.
    print(f"  • Time-To-Ready (synthetic) : {res['time_to_ready_ms']:.2f} ms (random/empty weights, no checkpoint I/O; no baseline measured)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Turing Engine Speculation, LoRA Cache & Cold-Start Benchmarks")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"[*] Initializing Hardware Benchmarks on Device: {device.upper()}")
    benchmark_multi_tenant_lora_cache(device=device)
    benchmark_speculative_drafting(device=device)
    benchmark_pipelined_cold_start(device=device)

if __name__ == "__main__":
    main()
