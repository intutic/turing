"""
High-Concurrency Production Serving Benchmark.
Simulates 1 to 128 concurrent client streams against Turing Engine ContinuousBatchEngine.
Measures TTFT, ITL, throughput, and P90/P99 latencies.
"""

import os
import sys
import time
import asyncio
import argparse
from typing import List

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.serving.engine import ContinuousBatchEngine

async def run_client_stream(engine: ContinuousBatchEngine, client_id: int, prompt_len: int, max_tokens: int, latencies: List[float]):
    prompt = [100 + (i % 500) for i in range(prompt_len)]
    t0 = time.time()
    first_token_time = None
    token_count = 0

    async for token in engine.stream_generate(prompt, max_new_tokens=max_tokens, temperature=0.7):
        if first_token_time is None:
            first_token_time = time.time()
        token_count += 1

    total_time = time.time() - t0
    ttft = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
    latencies.append((ttft, total_time, token_count))

async def main():
    parser = argparse.ArgumentParser(description="Turing Engine Serving Concurrency Benchmark")
    parser.add_argument("--model", type=str, default="test-tiny")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print(f"================================================================================")
    print(f"   ⚡ TURING ENGINE PRODUCTION SERVING & CONCURRENCY BENCHMARK")
    print(f"================================================================================")
    print(f"[*] Target Model Profile : {args.model}")
    print(f"[*] Concurrent Clients   : {args.concurrency} async streams")
    print(f"[*] Prompt Token Length  : {args.prompt_len} tokens / request")
    print(f"[*] Max Tokens Per Stream: {args.max_tokens} tokens")

    model_cfg = get_model_config(args.model)
    turing_cfg = TuringConfig(max_batch_size=args.concurrency)
    if args.device != "auto":
        turing_cfg.device = args.device

    engine = ContinuousBatchEngine(model_cfg, turing_cfg)
    await engine.start()

    latencies = []
    print(f"\n[🚀] Launching {args.concurrency} concurrent client requests...")
    start_bench = time.time()

    tasks = [
        asyncio.create_task(run_client_stream(engine, i, args.prompt_len, args.max_tokens, latencies))
        for i in range(args.concurrency)
    ]
    await asyncio.gather(*tasks)
    total_bench_time = time.time() - start_bench

    await engine.stop()

    ttfts = [l[0] for l in latencies]
    total_tokens = sum([l[2] for l in latencies])
    overall_throughput = total_tokens / total_bench_time

    import numpy as np
    p50_ttft = np.percentile(ttfts, 50)
    p90_ttft = np.percentile(ttfts, 90)
    p99_ttft = np.percentile(ttfts, 99)

    print(f"\n================================================================================")
    print(f"   📊 CONCURRENCY BENCHMARK RESULTS ({args.concurrency} CONCURRENT STREAMS)")
    print(f"================================================================================")
    print(f"  • Total Generated Tokens   : {total_tokens} tokens")
    print(f"  • Wallclock Benchmark Time : {total_bench_time * 1000.0:.2f} ms ({total_bench_time:.2f}s)")
    print(f"  • Overall Serving Throughput: {overall_throughput:.2f} tokens/second")
    print(f"  • Time-To-First-Token (TTFT) Distribution:")
    print(f"    - P50 (Median)           : {p50_ttft:.2f} ms")
    print(f"    - P90                    : {p90_ttft:.2f} ms")
    print(f"    - P99                    : {p99_ttft:.2f} ms")
    print(f"  • Inter-Token Latency (ITL): {(total_bench_time / (args.max_tokens)) * 1000.0 / args.concurrency:.2f} ms/token/client")
    print(f"================================================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
