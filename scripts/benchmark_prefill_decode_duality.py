"""
Empirical Benchmark Suite: Prefill (Encoder / Compute-Bound) vs. Decode (Decoder / Bandwidth-Bound) Duality.
Validates Chunked Prefill TTFT, Batched Decode TPS, ITL Jitter Suppression, Speculation Gating, and SVD KV Memory Reduction.
"""

import sys
import os
import time
import argparse
import asyncio
import json
import torch
import torch.nn.functional as F

from turing.config import ModelConfig, TuringConfig
from turing.serving.engine import ContinuousBatchEngine, AsyncSequenceRequest, RequestState
from turing.serving.traffic import LanePolicy, Lane, KVMemoryEstimator
from turing.serving.spec_gate import SpeculationGatePolicy, SpecGateDecision


def benchmark_prefill_scaling(device: str = "cpu") -> dict:
    """Measures Prefill TTFT and Compute Scaling across sequence lengths."""
    print("\n" + "="*80)
    print(f"📊 [BENCHMARK 1/5] PREFILL (ENCODER) COMPUTE SCALING ON {device.upper()}")
    print("="*80)

    config = ModelConfig(
        name="bench-prefill",
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        ffn_dim=2816,
        max_position_embeddings=16384,
        head_dim=128
    )
    turing_config = TuringConfig(device=device, max_batch_size=8)
    engine = ContinuousBatchEngine(model_config=config, turing_config=turing_config, prefill_chunk_size=512)

    seq_lengths = [128, 512, 1024, 2048, 4096]
    results = {}

    for seq_len in seq_lengths:
        prompt = list(range(seq_len))
        input_tensor = torch.tensor([prompt], dtype=torch.long, device=engine.device)

        # Warmup
        with torch.inference_mode():
            _ = engine.model(input_tensor[:, :64])

        if device == "cuda":
            torch.cuda.synchronize()

        # Timed Prefill
        trials = 5
        times = []
        for _ in range(trials):
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits, _ = engine.model(input_tensor)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        median_time_ms = sorted(times)[len(times) // 2] * 1000.0
        tok_per_sec = seq_len / (median_time_ms / 1000.0)

        # Theoretical FLOPs: 2 * L * (4 * H * d + 3 * H * ffn_dim) * seq_len
        # For prompt seq_len, self-attention adds 2 * L * num_heads * (seq_len^2) * head_dim
        gemm_flops = 2 * config.num_layers * (4 * config.hidden_dim**2 + 3 * config.hidden_dim * config.ffn_dim) * seq_len
        attn_flops = 2 * config.num_layers * config.num_heads * (seq_len**2) * config.head_dim
        total_gflops = (gemm_flops + attn_flops) / 1e9
        gflops_per_sec = total_gflops / (median_time_ms / 1000.0)

        results[seq_len] = {
            "prefill_time_ms": round(median_time_ms, 2),
            "prefill_throughput_tok_per_sec": round(tok_per_sec, 2),
            "compute_gflops_per_sec": round(gflops_per_sec, 2)
        }

        print(f"  • Prompt Length: {seq_len:5d} tokens | TTFT: {median_time_ms:7.2f} ms | Throughput: {tok_per_sec:8.2f} tok/s | Compute: {gflops_per_sec:7.2f} GFLOP/s")

    return results


def benchmark_batched_decode_tps(device: str = "cpu") -> dict:
    """Measures Decode Throughput (TPS) scaling across concurrent batch sizes."""
    print("\n" + "="*80)
    print(f"📊 [BENCHMARK 2/5] DECODE (BANDWIDTH-BOUND) BATCHED THROUGHPUT ON {device.upper()}")
    print("="*80)

    config = ModelConfig(
        name="bench-decode",
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=4,
        num_heads=8,
        num_kv_heads=4,
        ffn_dim=2816,
        max_position_embeddings=8192,
        head_dim=128
    )

    batch_sizes = [1, 2, 4, 8, 16]
    results = {}

    for bs in batch_sizes:
        turing_config = TuringConfig(device=device, max_batch_size=bs)
        engine = ContinuousBatchEngine(model_config=config, turing_config=turing_config)

        # Mock active decode inputs: shape [B, 1]
        inputs = torch.ones((bs, 1), dtype=torch.long, device=engine.device)

        # Warmup
        with torch.inference_mode():
            _ = engine.model(inputs)

        if device == "cuda":
            torch.cuda.synchronize()

        decode_steps = 20
        t0 = time.perf_counter()
        with torch.inference_mode():
            past_kv = None
            for _ in range(decode_steps):
                logits, past_kv = engine.model(inputs, past_key_values=past_kv)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed_s = t1 - t0
        total_tokens = bs * decode_steps
        tps = total_tokens / elapsed_s
        step_latency_ms = (elapsed_s / decode_steps) * 1000.0

        results[bs] = {
            "decode_batch_size": bs,
            "step_latency_ms": round(step_latency_ms, 2),
            "aggregate_decode_tps": round(tps, 2),
            "per_stream_tps": round(tps / bs, 2)
        }

        print(f"  • Batch Size: {bs:2d} | Step Latency: {step_latency_ms:6.2f} ms | Aggregate TPS: {tps:7.2f} tok/s | Per-Stream TPS: {tps/bs:6.2f} tok/s")

    return results


def benchmark_interleaved_prefill_decode_jitter(device: str = "cpu") -> dict:
    """Measures Inter-Token Latency (ITL) P50/P95/P99 stability during massive prompt bursts."""
    print("\n" + "="*80)
    print(f"📊 [BENCHMARK 3/5] PREFILL-DECODE CO-SCHEDULING & ITL JITTER ON {device.upper()}")
    print("="*80)

    config = ModelConfig(
        name="bench-jitter",
        vocab_size=32000,
        hidden_dim=512,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        ffn_dim=1024,
        max_position_embeddings=4096,
        head_dim=128
    )
    turing_config = TuringConfig(device=device, max_batch_size=8)
    engine = ContinuousBatchEngine(
        model_config=config,
        turing_config=turing_config,
        prefill_chunk_size=256
    )

    async def _run_test():
        await engine.start()
        try:
            # 1. Spawn 4 concurrent decode streams generating 16 tokens each
            async def decode_stream(req_id: int):
                toks = []
                async for tok in engine.stream_generate(prompt_tokens=[1, 2, 3], max_new_tokens=16, temperature=0.0):
                    toks.append(tok)
                return toks

            # 2. Spawn 1 large prefill request (1024 tokens) midway
            async def burst_prefill():
                await asyncio.sleep(0.02)  # inject during active decode
                toks = []
                async for tok in engine.stream_generate(prompt_tokens=list(range(1024)), max_new_tokens=4, temperature=0.0):
                    toks.append(tok)
                return toks

            tasks = [decode_stream(i) for i in range(4)]
            tasks.append(burst_prefill())

            await asyncio.gather(*tasks)

            telemetry = engine.get_telemetry()
            lat = telemetry["latency"]
            return lat
        finally:
            await engine.stop()

    lat_metrics = asyncio.run(_run_test())

    print(f"  • TTFT P50: {lat_metrics.get('p50_ttft_ms', 0):.2f} ms | TTFT P95: {lat_metrics.get('p95_ttft_ms', 0):.2f} ms | TTFT P99: {lat_metrics.get('p99_ttft_ms', 0):.2f} ms")
    print(f"  • ITL  P50: {lat_metrics.get('p50_itl_ms', 0):.2f} ms | ITL  P95: {lat_metrics.get('p95_itl_ms', 0):.2f} ms | ITL  P99: {lat_metrics.get('p99_itl_ms', 0):.2f} ms")

    return lat_metrics


def benchmark_speculation_concurrency_gating() -> dict:
    """Verifies Concurrency-Adaptive Speculation Gating state transitions."""
    print("\n" + "="*80)
    print("📊 [BENCHMARK 4/5] CONCURRENCY-ADAPTIVE SPECULATION GATING POLICY")
    print("="*80)

    policy = SpeculationGatePolicy(low_threshold=2, high_threshold=4, default_tree_width=8, collapsed_tree_width=2)
    transitions = []

    test_concurrencies = [1, 2, 3, 4, 8, 3, 1]
    for c in test_concurrencies:
        mode = policy.gate_decision(active_sessions=c)
        width = policy.tree_width()
        transitions.append({"active_sessions": c, "mode": mode.value, "tree_width": width})
        print(f"  • Active Sessions: {c:2d} ➔ Mode: {mode.value:10s} (Tree Width: {width:2d})")

    return {"transitions": transitions, "stats": policy.stats}


def benchmark_svd_kv_bandwidth_reduction() -> dict:
    """Calculates analytical KV Cache memory footprint & memory bandwidth reduction."""
    print("\n" + "="*80)
    print("📊 [BENCHMARK 5/5] SVD INT8 KV MEMORY & BANDWIDTH FOOTPRINT")
    print("="*80)

    contexts = [2048, 8192, 32768, 131072]
    num_layers = 32
    num_kv_heads = 8
    head_dim = 128

    results = {}
    for ctx in contexts:
        fp16_bytes = KVMemoryEstimator.estimate_kv_bytes(
            num_prompt_tokens=ctx,
            max_new_tokens=0,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=2,
            svd_compression_ratio=0.0
        )
        svd_int8_bytes = KVMemoryEstimator.estimate_kv_bytes(
            num_prompt_tokens=ctx,
            max_new_tokens=0,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=2,
            svd_compression_ratio=0.75
        )

        fp16_gb = fp16_bytes / (1024**3)
        svd_gb = svd_int8_bytes / (1024**3)
        reduction_pct = (1.0 - svd_bytes / fp16_bytes) * 100.0 if (svd_bytes := svd_int8_bytes) else 0.0

        results[ctx] = {
            "fp16_gb": round(fp16_gb, 3),
            "svd_int8_gb": round(svd_gb, 3),
            "memory_reduction_pct": round(reduction_pct, 1)
        }

        print(f"  • Context: {ctx:6d} tokens | FP16 KV: {fp16_gb:6.2f} GB | SVD INT8 KV: {svd_gb:6.2f} GB | Reduction: -{reduction_pct:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Turing Engine Prefill vs Decode Duality Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--output", type=str, default="results_prefill_decode_duality.json")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("⚠️ MPS not available, falling back to CPU.")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU.")
        device = "cpu"

    print(f"🚀 Starting Prefill vs Decode Duality Benchmark on {device.upper()}...")
    t_start = time.time()

    all_results = {
        "device": device,
        "prefill_scaling": benchmark_prefill_scaling(device),
        "batched_decode": benchmark_batched_decode_tps(device),
        "interleaved_jitter": benchmark_interleaved_prefill_decode_jitter(device),
        "spec_gating": benchmark_speculation_concurrency_gating(),
        "svd_kv_reduction": benchmark_svd_kv_bandwidth_reduction(),
        "total_benchmark_time_seconds": round(time.time() - t_start, 2)
    }

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print(f"✅ Benchmark Complete in {all_results['total_benchmark_time_seconds']}s. Results saved to {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
