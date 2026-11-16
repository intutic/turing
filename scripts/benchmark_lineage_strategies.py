"""
Multi-turn lineage drift and k-slot pooling speedup benchmark.

Multi-turn lineage drift occurs when KV states are passed across multiple 
deliberation turns. Naive strategies accumulate projection errors iteratively,
causing the residual norm to explode. The Clean Base strategy keeps the error 
bounded by re-projecting from a stable base, while the Append-only strategy 
flatlines but consumes linearly increasing memory.

Part 2 times a full-length linear transfer against k-slot pooling followed by the
same transfer, on randn K/V with a random-init pooler; it measures latency only,
not what the pooled slots preserve.
"""

import argparse
import json
import time
import math
import torch
import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from turing.core.lineage import (
    CleanBaseStrategy, NaiveStrategy, AppendOnlyStrategy,
    CleanBaseLineageBuffer, CacheLineage
)
from turing.core.kslot_pooling import KSlotCachePooler

def get_device(device_arg):
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_arg)

def sync_device(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()

@torch.inference_mode()
def benchmark_lineage_strategies(args, device):
    """Drive the library's turn strategies with a synthetic translator.

    Synthetic: the cache is torch.randn and the per-turn translation is a seeded random
    linear map with spectral norm `--gain`, so the residual norms show how each strategy
    compounds (or does not compound) a translation; they are not a model-quality measurement.
    """
    print(f"\n--- Part 1: Multi-Turn Lineage Strategies ({args.num_turns} turns, seq_len={args.seq_len}; "
          f"synthetic cache and translator, gain={args.gain}) ---")
    hidden_size = 4096
    gen = torch.Generator(device="cpu").manual_seed(0)
    base = torch.randn(args.seq_len, hidden_size, generator=gen).to(device)
    m = torch.randn(hidden_size, hidden_size, generator=gen)
    m = (m * (args.gain / torch.linalg.matrix_norm(m, ord=2))).to(device)

    def translate(cache):
        # One deliberation turn: the receiver cache plus the translated residual.
        return cache + cache @ m

    results = {}
    for strategy in (CleanBaseStrategy(), NaiveStrategy(), AppendOnlyStrategy()):
        print(f"\nStrategy: {strategy.name.upper()}")
        print(f"{'Turn':<6} | {'Residual Norm ||dC_R||_2':<25} | {'Latency (ms)':<15} | {'Cache bytes (MB)':<15}")
        print("-" * 65)
        previous, live = None, []
        strategy_results = []
        for turn in range(args.num_turns):
            sync_device(device)
            t0 = time.perf_counter()
            src = strategy.cache_for_turn(turn, [base], previous)[0]
            if strategy.translates_on_turn(turn):
                out = translate(src)
            else:
                # Append-only keeps the cache it has and appends this turn's new tokens.
                out = torch.cat([src, base], dim=0)
            sync_device(device)
            latency_ms = (time.perf_counter() - t0) * 1000
            residual_norm = torch.linalg.norm(out[:args.seq_len] - base).item()
            cache_mb = out.numel() * out.element_size() / (1024 * 1024)
            previous = [out]
            print(f"{turn + 1:<6} | {residual_norm:<25.4f} | {latency_ms:<15.2f} | {cache_mb:<15.2f}")
            strategy_results.append({'turn': turn + 1, 'residual_norm': residual_norm,
                                     'latency_ms': latency_ms, 'cache_mb': cache_mb})
        results[strategy.name] = strategy_results
    return results

@torch.inference_mode()
def benchmark_kslot_pooling(device):
    print("\n--- Part 2: k-Slot Pooling vs Full Transfer (k=4; random-init pooler, randn K/V; median of 20) ---")
    seq_lens = [512, 1024, 4096, 8192, 16384]
    k_slots = 4
    num_layers = 4
    num_kv_heads = 8
    head_dim = 128
    hidden_dim = num_kv_heads * head_dim

    pooler = KSlotCachePooler(num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, num_slots=k_slots).to(device)
    linear_map = torch.nn.Linear(hidden_dim, hidden_dim, bias=False, device=device)

    print(f"{'Context (N)':<12} | {'Standard (ms)':<15} | {'k-Slot (ms)':<15} | {'Ratio':<10} | {'Speedup':<10}")
    print("-" * 70)

    results = []

    # Warmup
    warm_k = torch.randn(1, num_layers, num_kv_heads, 512, head_dim, device=device)
    warm_v = torch.randn(1, num_layers, num_kv_heads, 512, head_dim, device=device)
    _ = pooler(warm_k, warm_v)
    _ = linear_map(warm_k.view(num_layers, 512, hidden_dim))
    sync_device(device)

    def median_ms(fn, reps=20):
        times = []
        for _ in range(reps):
            sync_device(device)
            t0 = time.perf_counter()
            fn()
            sync_device(device)
            times.append((time.perf_counter() - t0) * 1000.0)
        return sorted(times)[reps // 2]

    for n in seq_lens:
        k = torch.randn(1, num_layers, num_kv_heads, n, head_dim, device=device)
        v = torch.randn(1, num_layers, num_kv_heads, n, head_dim, device=device)

        # Standard transfer: map all N tokens across layers
        standard_ms = median_ms(lambda: linear_map(k.view(num_layers, n, hidden_dim)))
        # k-Slot pooled transfer: pool to k slots, then map only k slots
        pooled_ms = median_ms(lambda: linear_map(pooler(k, v)[0].view(num_layers, k_slots, hidden_dim)))

        speedup = standard_ms / max(pooled_ms, 1e-6)
        compression = float(n) / k_slots

        print(f"{n:<12} | {standard_ms:<15.3f} | {pooled_ms:<15.3f} | {compression:<10.1f} | {speedup:>5.1f}x")

        results.append({
            'seq_len': n,
            'standard_ms': standard_ms,
            'pooled_ms': pooled_ms,
            'compression_ratio': compression,
            'speedup': speedup
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark Lineage Strategies and k-Slot Pooling")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    parser.add_argument('--num_turns', type=int, default=6)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--gain', type=float, default=0.5, help='Spectral norm of the synthetic per-turn translator')
    parser.add_argument('--json_out', type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Running benchmarks on device: {device}")

    # Run benchmarks
    lineage_results = benchmark_lineage_strategies(args, device)
    pooling_results = benchmark_kslot_pooling(device)

    # Save to JSON if requested
    if args.json_out:
        out_data = {
            'metadata': {
                'device': str(device),
                'num_turns': args.num_turns,
                'seq_len': args.seq_len
            },
            'lineage_strategies': lineage_results,
            'kslot_pooling': pooling_results
        }
        with open(args.json_out, 'w') as f:
            json.dump(out_data, f, indent=2)
        print(f"\nResults saved to {args.json_out}")

if __name__ == '__main__':
    main()
