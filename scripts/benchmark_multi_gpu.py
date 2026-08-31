"""
Empirical Benchmark: Distributed Multi-GPU & Pipeline Parallel Throughput Scaling.
Measures token throughput (tok/s), micro-batch bubble overhead, and pipeline speedup across GPU partitions.
"""

import sys
import time
import argparse
import torch

from turing.config import TuringConfig, ModelConfig
from turing.models.registry import get_model_config
from turing.models.causal_lm import SubspaceCausalLM
from turing.models.tensor_parallel import partition_model_for_pipeline, MicroBatchScheduler
from turing.serving.distributed import DistributedConfig, DistributedInferenceDriver, PlacementPolicy


def benchmark_distributed_scaling(
    model_name: str = "test-tiny",
    pp_stages: int = 2,
    num_micro_batches: int = 4,
    batch_size: int = 4,
    seq_len: int = 32,
    device: str = "auto"
):
    jcfg = TuringConfig(device=device)
    dev = jcfg.resolve_device()

    print(f"\n[*] Benchmarking Distributed Pipeline Parallelism on device: {dev}")
    print(f"[*] Model: {model_name}, PP Stages: {pp_stages}, Micro-Batches: {num_micro_batches}, Batch Size: {batch_size}\n")

    cfg = get_model_config(model_name)
    model = SubspaceCausalLM(cfg).to(dev).eval()

    # 1. Baseline: Single-Stage Forward Execution
    inputs = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=dev)
    
    # Warmup
    with torch.inference_mode():
        _ = model(inputs)
    
    start_single = time.perf_counter()
    num_iters = 10
    with torch.inference_mode():
        for _ in range(num_iters):
            logits_single, _ = model(inputs)
    single_time = (time.perf_counter() - start_single) / num_iters
    single_tok_s = (batch_size * seq_len) / max(1e-5, single_time)

    # 2. Pipeline Parallel Distributed Driver Execution
    dist_cfg = DistributedConfig(tp_size=1, pp_size=pp_stages)
    driver = DistributedInferenceDriver(model=model, config=cfg, dist_config=dist_cfg)

    # Warmup
    with torch.inference_mode():
        _ = driver.forward_distributed(inputs)

    start_dist = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_iters):
            logits_dist, _ = driver.forward_distributed(inputs)
    dist_time = (time.perf_counter() - start_dist) / num_iters
    dist_tok_s = (batch_size * seq_len) / max(1e-5, dist_time)

    scheduler = MicroBatchScheduler(num_stages=pp_stages, num_micro_batches=num_micro_batches)
    bubble_pct = scheduler.bubble_ratio * 100.0

    print("=" * 72)
    print("🚀 Turing Distributed Pipeline Parallelism Benchmark Results")
    print("=" * 72)
    print(f"  Single-Stage Baseline Latency : {single_time*1000:8.2f} ms ({single_tok_s:8.1f} tok/s)")
    print(f"  Distributed (PP={pp_stages}) Latency    : {dist_time*1000:8.2f} ms ({dist_tok_s:8.1f} tok/s)")
    print(f"  Micro-Batch Bubble Overhead   : {bubble_pct:8.2f} %")
    print(f"  Theoretical Efficiency        : {100.0 - bubble_pct:8.2f} %")
    print("=" * 72)

    return {
        "device": str(dev),
        "single_latency_ms": single_time * 1000,
        "dist_latency_ms": dist_time * 1000,
        "bubble_ratio_pct": bubble_pct,
        "single_tok_s": single_tok_s,
        "dist_tok_s": dist_tok_s
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turing Distributed Inference Benchmark")
    parser.add_argument("--model", type=str, default="test-tiny", help="Model target")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallelism stages")
    parser.add_argument("--micro-batches", type=int, default=4, help="Number of micro-batches")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=32, help="Sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device")
    args = parser.parse_args()

    benchmark_distributed_scaling(
        model_name=args.model,
        pp_stages=args.pp,
        num_micro_batches=args.micro_batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=args.device
    )
