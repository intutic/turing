"""
Empirical Benchmark: Turing Programmatic DSL Parallel Fork/Join vs Sequential Generation.
Measures latency speedup, prefix KV cache sharing efficiency, and branch throughput.
"""

import sys
import time
import argparse
import torch

from turing.config import TuringConfig
from turing.models.registry import get_model_config
from turing.models.causal_lm import SubspaceCausalLM
from turing.dsl import chain, gen, fork, join, select, LocalExecutor


def benchmark_dsl_fork_join(device: str = "auto", num_branches: int = 8, tokens_per_branch: int = 16):
    jcfg = TuringConfig(device=device)
    dev = jcfg.resolve_device()
    print(f"\n[*] Benchmarking Turing Programmatic DSL on device: {dev}")
    print(f"[*] Configuration: {num_branches} branches, {tokens_per_branch} tokens/branch\n")

    cfg = get_model_config("test-tiny")
    model = SubspaceCausalLM(cfg).to(dev).eval()
    executor = LocalExecutor(model=model, device=str(dev))

    # 1. Baseline: Sequential N independent generations (re-prefilling each time)
    start_seq = time.perf_counter()
    seq_outputs = []
    prefix_prompt = "Explain quantum computing fundamentals in detail: "
    for i in range(num_branches):
        full_p = f"{prefix_prompt} Branch {i}: "
        tokens = [ord(c) % cfg.vocab_size for c in full_p]
        text, g_tokens, _ = executor.generate(tokens=tokens, max_tokens=tokens_per_branch)
        seq_outputs.append(text)
    seq_time = time.perf_counter() - start_seq
    seq_tok_s = (num_branches * tokens_per_branch) / max(1e-5, seq_time)

    # 2. Turing DSL: Shared prefix fork() + join()
    @chain(executor=executor)
    def dsl_fork_join_workflow():
        gen(prefix_prompt, max_tokens=4)
        branches = fork(n=num_branches)
        for i, b in enumerate(branches):
            b.gen(f" Branch {i}: ", max_tokens=tokens_per_branch)
        winner = join(branches, strategy="best")
        return winner

    start_dsl = time.perf_counter()
    dsl_output = dsl_fork_join_workflow()
    dsl_time = time.perf_counter() - start_dsl
    dsl_tok_s = (num_branches * tokens_per_branch) / max(1e-5, dsl_time)

    speedup = seq_time / max(1e-5, dsl_time)

    print("=" * 68)
    print("🚀 Turing DSL Parallel Fork/Join Benchmark Results")
    print("=" * 68)
    print(f"  Sequential Baseline Latency : {seq_time*1000:8.2f} ms ({seq_tok_s:8.1f} tok/s)")
    print(f"  Turing DSL Fork/Join Latency: {dsl_time*1000:8.2f} ms ({dsl_tok_s:8.1f} tok/s)")
    print(f"  Execution Speedup           : {speedup:8.2f}x")
    print("=" * 68)
    return {
        "device": str(dev),
        "branches": num_branches,
        "tokens_per_branch": tokens_per_branch,
        "sequential_latency_ms": seq_time * 1000,
        "dsl_latency_ms": dsl_time * 1000,
        "speedup": speedup
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turing DSL Benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu, mps)")
    parser.add_argument("--branches", type=int, default=8, help="Number of parallel fork branches")
    parser.add_argument("--tokens", type=int, default=16, help="Tokens per branch")
    args = parser.parse_args()
    benchmark_dsl_fork_join(device=args.device, num_branches=args.branches, tokens_per_branch=args.tokens)
