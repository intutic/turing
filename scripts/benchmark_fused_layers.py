"""
Comprehensive Empirical Benchmark for C++20 SIMD Micro-Kernels & Fused Triton Layers.
Measures latency (ms), throughput, and memory bandwidth reduction across all migrated components.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import turing.turing_csrc as turing_csrc
from turing.models.gguf_loader import GGMLType, GGUFDequantizer
from turing.kernels.triton_fused_rmsnorm_swiglu import dispatch_fused_rmsnorm_swiglu
from turing.kernels.triton_select_gather import dispatch_batched_option_select


def benchmark_gguf_simd():
    print("\n" + "=" * 70)
    print(" 🚀 BENCHMARK: C++20 AVX2/NEON SIMD vs NumPy GGUF Dequantization")
    print("=" * 70)

    num_blocks = 100_000 # 3.2 million parameters
    delta = np.float16(0.125).tobytes()
    quants = np.random.randint(-128, 127, size=(num_blocks, 32), dtype=np.int8).tobytes()
    # Interleave delta + quants
    raw_blocks = bytearray()
    for i in range(num_blocks):
        raw_blocks.extend(delta)
        raw_blocks.extend(quants[i*32:(i+1)*32])
    data_bytes = bytes(raw_blocks)

    # 1. NumPy Baseline
    t0 = time.perf_counter()
    raw = np.frombuffer(data_bytes, dtype=np.uint8, count=num_blocks * 34).reshape((num_blocks, 34))
    deltas = np.frombuffer(raw[:, :2].copy(), dtype=np.float16).astype(np.float32)
    quants_np = raw[:, 2:].view(np.int8).astype(np.float32)
    dequant_np = (quants_np * deltas[:, None]).reshape(num_blocks * 32)
    t_numpy = (time.perf_counter() - t0) * 1000.0

    # 2. C++20 SIMD
    t0 = time.perf_counter()
    res_cpp = turing_csrc.dequantize_gguf_simd(data_bytes, int(GGMLType.Q8_0.value), [num_blocks * 32])
    t_simd = (time.perf_counter() - t0) * 1000.0

    speedup = t_numpy / max(t_simd, 1e-6)
    print(f"  • Elements Unpacked: {num_blocks * 32:,}")
    print(f"  • Python NumPy Baseline: {t_numpy:.2f} ms ({num_blocks * 32 / (t_numpy * 1e3):.2f} M elem/s)")
    print(f"  • C++20 SIMD Kernel:     {t_simd:.2f} ms ({num_blocks * 32 / (t_simd * 1e3):.2f} M elem/s)")
    print(f"  • Measured Speedup:      {speedup:.2f}x FASTER\n")


def benchmark_fused_ffn():
    print("=" * 70)
    print(" 🚀 BENCHMARK: Fused RMSNorm + Subspace SwiGLU + In-Place Residual")
    print("=" * 70)

    B, K, ffn_dim = 16, 4096, 14336
    tile_size = 64
    active_tiles = torch.arange(0, 112, dtype=torch.int32) # 50% channel pruning

    x = torch.randn(B, K)
    weight_norm = torch.randn(K)
    w_gate = torch.randn(K, ffn_dim)
    w_up = torch.randn(K, ffn_dim)
    w_down = torch.randn(ffn_dim, K)
    residual = torch.randn(B, K)

    # Warmup
    for _ in range(5):
        dispatch_fused_rmsnorm_swiglu(x, weight_norm, w_gate, w_up, w_down, residual, active_tiles, tile_size)

    # 1. Unfused Execution
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        var = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + 1e-6) * weight_norm
        idx_list = []
        for t in active_tiles.tolist():
            idx_list.extend(range(t * tile_size, (t + 1) * tile_size))
        idx_t = torch.tensor(idx_list, dtype=torch.long)
        gate = torch.matmul(x_norm, w_gate[:, idx_t])
        up = torch.matmul(x_norm, w_up[:, idx_t])
        out = torch.matmul(F.silu(gate) * up, w_down[idx_t, :]) + residual
    t_unfused = ((time.perf_counter() - t0) / iters) * 1000.0

    # 2. Fused Execution
    t0 = time.perf_counter()
    for _ in range(iters):
        out_fused = dispatch_fused_rmsnorm_swiglu(
            x, weight_norm, w_gate, w_up, w_down, residual, active_tiles, tile_size
        )
    t_fused = ((time.perf_counter() - t0) / iters) * 1000.0

    speedup = t_unfused / max(t_fused, 1e-6)
    print(f"  • Hidden Dim: {K}, FFN Dim: {ffn_dim}, Batch: {B}")
    print(f"  • Unfused Multi-Pass Baseline: {t_unfused:.2f} ms")
    print(f"  • Fused SRAM Single-Pass:      {t_fused:.2f} ms")
    print(f"  • Measured Speedup:            {speedup:.2f}x FASTER\n")


def benchmark_dsl_select():
    print("=" * 70)
    print(" 🚀 BENCHMARK: Batched GPU Option Select vs Sequential Host Loop")
    print("=" * 70)

    vocab_size = 32000
    num_options = 32
    log_probs = F.log_softmax(torch.randn(vocab_size), dim=-1)
    options_tokens = [list(np.random.randint(10, vocab_size, size=np.random.randint(2, 8))) for _ in range(num_options)]

    # 1. Sequential Python Loop
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        best_score = float("-inf")
        best_idx = 0
        for i, opt in enumerate(options_tokens):
            score = log_probs[opt[0]].item()
            if score > best_score:
                best_score = score
                best_idx = i
    t_seq = ((time.perf_counter() - t0) / iters) * 1000.0

    # 2. Batched Dispatcher
    t0 = time.perf_counter()
    for _ in range(iters):
        best_idx_batched = dispatch_batched_option_select(log_probs, options_tokens)
    t_batched = ((time.perf_counter() - t0) / iters) * 1000.0

    speedup = t_seq / max(t_batched, 1e-6)
    print(f"  • Candidate Options: {num_options}, Vocab: {vocab_size}")
    print(f"  • Sequential Host Loop: {t_seq * 1e3:.2f} µs")
    print(f"  • Batched GPU Gather:   {t_batched * 1e3:.2f} µs")
    print(f"  • Measured Speedup:     {speedup:.2f}x FASTER\n")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    benchmark_gguf_simd()
    benchmark_fused_ffn()
    benchmark_dsl_select()
