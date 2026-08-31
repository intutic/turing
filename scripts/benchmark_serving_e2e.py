"""
End-to-End Continuous Serving Load Test with Traffic Management & Prometheus Metrics.
Evaluates continuous batching serving runtime under concurrent load across QoS lanes,
measuring TTFT, ITL, generation throughput, VRAM admission control, and dynamic spec gating.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from turing.config import ModelConfig, TuringConfig
from turing.serving.engine import ContinuousBatchEngine, AsyncSequenceRequest
from turing.serving.traffic import AdmissionController, LanePolicy, Lane
from turing.serving.spec_gate import SpeculationGatePolicy


def calculate_percentile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(p * len(sorted_data))
    return sorted_data[min(idx, len(sorted_data) - 1)]


async def run_workload(
    engine: ContinuousBatchEngine,
    num_requests: int = 30,
    concurrency: int = 4
) -> Dict[str, Any]:
    await engine.start()
    semaphore = asyncio.Semaphore(concurrency)
    
    lanes = [Lane.INTERACTIVE, Lane.BATCH, Lane.BACKGROUND]
    requests: List[AsyncSequenceRequest] = []
    
    async def process_request(idx: int):
        async with semaphore:
            prompt_len = random.randint(32, 128)
            prompt_tokens = [random.randint(1, 1000) for _ in range(prompt_len)]
            max_new = random.randint(10, 30)
            lane = random.choice(lanes)
            
            try:
                # Consume output stream
                async for _ in engine.stream_generate(
                    prompt_tokens=prompt_tokens,
                    max_new_tokens=max_new,
                    temperature=0.0,
                    lane=lane
                ):
                    pass
            except Exception:
                pass

    start_time = time.time()
    tasks = [process_request(i) for i in range(num_requests)]
    await asyncio.gather(*tasks)
    duration = max(0.001, time.time() - start_time)
    
    telemetry = engine.get_telemetry()
    await engine.stop()
    
    ttfts = [t * 1000.0 for t in engine.recent_ttft]
    itls = [t * 1000.0 for t in engine.recent_itl]
    
    return {
        "throughput_tok_s": round(telemetry.get("serving_throughput_tok_per_sec", 0.0), 2),
        "total_tokens": telemetry.get("total_tokens_generated", 0),
        "duration_s": round(duration, 2),
        "ttft_ms": {
            "avg": round(telemetry.get("latency", {}).get("avg_ttft_ms", 0.0), 2),
            "p50": round(calculate_percentile(ttfts, 0.50), 2),
            "p95": round(telemetry.get("latency", {}).get("p95_ttft_ms", 0.0), 2),
            "p99": round(telemetry.get("latency", {}).get("p99_ttft_ms", 0.0), 2),
        },
        "itl_ms": {
            "avg": round(telemetry.get("latency", {}).get("avg_itl_ms", 0.0), 2),
            "p50": round(calculate_percentile(itls, 0.50), 2),
            "p95": round(calculate_percentile(itls, 0.95), 2),
        },
        "kv_pool_utilization_pct": round(engine.get_kv_cache_utilization() * 100.0, 2),
        "admission": telemetry.get("admission", {}),
        "spec_gate": telemetry.get("spec_gate", {})
    }


async def main():
    parser = argparse.ArgumentParser(description="End-to-End Continuous Serving Load Test")
    parser.add_argument("--num_requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--json_out", type=str, default=None)
    args = parser.parse_args()

    print(f"Starting Benchmark: {args.num_requests} requests, concurrency {args.concurrency}")

    config = ModelConfig(
        name="Turing-E2E-Serving",
        hidden_dim=256,
        ffn_dim=512,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        num_layers=4,
        vocab_size=1024
    )
    turing_cfg = TuringConfig(max_batch_size=args.concurrency * 2)

    # 1. Baseline: Vanilla Engine (no admission, no lane QoS)
    print("\nRunning Baseline Workload...")
    engine_base = ContinuousBatchEngine(
        model_config=config,
        turing_config=turing_cfg,
        prefill_chunk_size=128
    )
    baseline_stats = await run_workload(engine_base, args.num_requests, args.concurrency)

    # 2. Protected: Engine with AdmissionController, LanePolicy, and SpecGate
    print("Running Protected Workload (Admission + 3-Lane QoS + Spec Gate)...")
    adm_ctrl = AdmissionController(vram_budget_bytes=50_000_000, high_watermark=0.85, shed_watermark=0.95)
    lane_policy = LanePolicy(slo_target_p99_ms=50.0)
    spec_gate = SpeculationGatePolicy(low_threshold=2, high_threshold=4)
    
    engine_prot = ContinuousBatchEngine(
        model_config=config,
        turing_config=turing_cfg,
        prefill_chunk_size=128,
        admission=adm_ctrl,
        lane_policy=lane_policy,
        spec_gate=spec_gate
    )
    protected_stats = await run_workload(engine_prot, args.num_requests, args.concurrency)

    # Summary
    print("\n" + "=" * 55)
    print("END-TO-END SERVING SUMMARY")
    print("=" * 55)
    print(f"{'Metric':<24} | {'Baseline':<12} | {'Protected':<12}")
    print("-" * 55)
    print(f"{'Throughput (tok/s)':<24} | {baseline_stats['throughput_tok_s']:<12.2f} | {protected_stats['throughput_tok_s']:<12.2f}")
    print(f"{'TTFT Avg (ms)':<24} | {baseline_stats['ttft_ms']['avg']:<12.2f} | {protected_stats['ttft_ms']['avg']:<12.2f}")
    print(f"{'TTFT P95 (ms)':<24} | {baseline_stats['ttft_ms']['p95']:<12.2f} | {protected_stats['ttft_ms']['p95']:<12.2f}")
    print(f"{'TTFT P99 (ms)':<24} | {baseline_stats['ttft_ms']['p99']:<12.2f} | {protected_stats['ttft_ms']['p99']:<12.2f}")
    print(f"{'ITL Avg (ms)':<24} | {baseline_stats['itl_ms']['avg']:<12.2f} | {protected_stats['itl_ms']['avg']:<12.2f}")
    print(f"{'KV Util (%)':<24} | {baseline_stats['kv_pool_utilization_pct']:<12.2f} | {protected_stats['kv_pool_utilization_pct']:<12.2f}")

    results = {
        "baseline": baseline_stats,
        "protected": protected_stats
    }

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.json_out}")


if __name__ == "__main__":
    asyncio.run(main())
