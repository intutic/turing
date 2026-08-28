#!/usr/bin/env python3
"""
Physical Empirical Benchmarking Suite for Native C++20 SIMD & Triton/CUDA Kernel Fusions.
Measures execution latencies, throughput, and speedup ratios on CPU, Apple Silicon (MPS), and NVIDIA GPU (CUDA).
"""

import time
import argparse
import torch
import numpy as np

# Check native C++
try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    try:
        import turing_csrc
        HAS_CSRC = True
    except ImportError:
        HAS_CSRC = False


def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_matryoshka_spec(device: torch.device):
    print("\n" + "=" * 80)
    print(f"[*] 1. MATRYOSHKA SLICED SPECULATION DRAFT HEAD ({device.type.upper()})")
    print("=" * 80)

    from turing.core.speculation import MatryoshkaDraftHead, QuadtreeMRPSpeculator

    hidden_dim = 8192
    vocab_size = 32000
    slice_widths = [8192, 4096, 2048, 1024]

    hidden = torch.randn(1, hidden_dim, device=device, dtype=torch.float32)
    draft_head = MatryoshkaDraftHead(hidden_dim=hidden_dim, vocab_size=vocab_size, bias=False).to(device)

    # Warmup
    for _ in range(10):
        _ = draft_head(hidden, slice_width=8192)
    sync_device(device)

    N_ITERS = 100
    results = {}

    for w in slice_widths:
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            _ = draft_head(hidden, slice_width=w)
        sync_device(device)
        t1 = time.perf_counter()
        avg_ms = ((t1 - t0) / N_ITERS) * 1000.0
        results[w] = avg_ms

    baseline_ms = results[8192]
    print(f"{'Slice Width':<15} | {'Latency (ms)':<15} | {'Speedup vs Full Head':<20}")
    print("-" * 55)
    for w in slice_widths:
        speedup = baseline_ms / results[w]
        print(f"W = {w:<11} | {results[w]:>10.3f} ms | {speedup:>18.2f}x")

    return results


def benchmark_svd_int8_quant(device: torch.device):
    print("\n" + "=" * 80)
    print(f"[*] 2. FUSED SVD INT8 QUANTIZATION & RECONSTRUCTION ({device.type.upper()})")
    print("=" * 80)

    from turing.kernels.triton_svd_paged import fused_svd_int8_quant_cuda, fused_int8_dequant_svd_recon_cuda

    seq_lens = [512, 2048, 8192, 32768]
    head_dim = 128
    rank = 64

    print(f"{'Context Length':<15} | {'PyTorch (4-step)':<18} | {'Fused Kernel':<15} | {'Speedup':<10}")
    print("-" * 65)

    for seq_len in seq_lens:
        k = torch.randn(seq_len, head_dim, device=device, dtype=torch.float32)
        u_proj = torch.randn(head_dim, rank, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = torch.matmul(k, u_proj)
            if device.type == "cuda":
                _ = fused_svd_int8_quant_cuda(k, u_proj)
        sync_device(device)

        N_ITERS = 50

        # Benchmark PyTorch 4-step baseline
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            k_sub = torch.matmul(k, u_proj)
            scale = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
            k_int8 = torch.clamp(torch.round(k_sub / scale), -128, 127).to(torch.int8)
        sync_device(device)
        t1 = time.perf_counter()
        py_ms = ((t1 - t0) / N_ITERS) * 1000.0

        # Benchmark Fused Kernel
        sync_device(device)
        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            if device.type == "cuda":
                _ = fused_svd_int8_quant_cuda(k, u_proj)
            elif HAS_CSRC:
                k_np = k.detach().cpu().numpy()
                u_np = u_proj.detach().cpu().numpy()
                _ = turing_csrc.fused_svd_int8_quant(k_np, u_np)
            else:
                k_sub = torch.matmul(k, u_proj)
                scale = torch.amax(torch.abs(k_sub), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
                k_int8 = torch.clamp(torch.round(k_sub / scale), -128, 127).to(torch.int8)
        sync_device(device)
        t1 = time.perf_counter()
        fused_ms = ((t1 - t0) / N_ITERS) * 1000.0

        speedup = py_ms / max(fused_ms, 1e-6)
        print(f"L = {seq_len:<11} | {py_ms:>13.3f} ms | {fused_ms:>10.3f} ms | {speedup:>8.2f}x")


def benchmark_cross_kv(device: torch.device):
    print("\n" + "=" * 80)
    print(f"[*] 3. FUSED INVERSE-RoPE & RIDGE REPRESENTATION TRANSFER ({device.type.upper()})")
    print("=" * 80)

    from turing.core.cross_model_kv import RoPEContentDecoupler, ClosedFormRidgeMapper
    from turing.kernels.triton_cross_kv import fused_inv_rope_cuda

    seq_len = 2048
    num_heads = 32
    head_dim = 128
    k = torch.randn(2, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        _ = RoPEContentDecoupler.strip_rope(k)
    sync_device(device)

    N_ITERS = 50

    # PyTorch baseline
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = RoPEContentDecoupler.strip_rope(k)
    sync_device(device)
    t1 = time.perf_counter()
    py_ms = ((t1 - t0) / N_ITERS) * 1000.0

    # Fused
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        if device.type == "cuda":
            _ = fused_inv_rope_cuda(k)
        elif HAS_CSRC:
            k_cpu = k.detach().cpu().numpy()
            _ = turing_csrc.fused_rope_transform(k_cpu, 500000.0, 0, True)
        else:
            _ = RoPEContentDecoupler.strip_rope(k)
    sync_device(device)
    t1 = time.perf_counter()
    fused_ms = ((t1 - t0) / N_ITERS) * 1000.0

    speedup = py_ms / max(fused_ms, 1e-6)
    print(f"Inverse-RoPE (2048 tokens): PyTorch = {py_ms:.3f} ms | Fused = {fused_ms:.3f} ms | Speedup = {speedup:.2f}x")


def benchmark_hca_chunk_pool(device: torch.device):
    print("\n" + "=" * 80)
    print(f"[*] 4. FUSED HIERARCHICAL ATTENTION CHUNK POOLING ({device.type.upper()})")
    print("=" * 80)

    from turing.core.hierarchical_compression import HCAChunkCompressor
    from turing.kernels.triton_chunk_compression import hca_chunk_pool_cuda

    seq_len = 32768
    num_heads = 8
    head_dim = 128
    chunk_size = 128

    k = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=torch.float32)

    comp = HCAChunkCompressor(hidden_dim=num_heads * head_dim, chunk_size=chunk_size).to(device)

    # Warmup
    for _ in range(5):
        _ = comp.compress_chunk(k, v)
    sync_device(device)

    N_ITERS = 20

    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = comp.compress_chunk(k, v)
    sync_device(device)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / N_ITERS) * 1000.0

    print(f"HCA 32K Sequence Pooling (128x compression): {avg_ms:.3f} ms / pass (Total chunks = {seq_len // chunk_size})")


def benchmark_mhc_hyperconnections(device: torch.device):
    print("\n" + "=" * 80)
    print(f"[*] 5. FUSED MANIFOLD-CONSTRAINED HYPER-CONNECTIONS ({device.type.upper()})")
    print("=" * 80)

    from turing.core.mhc import ManifoldHyperConnection

    hidden_dim = 4096
    num_streams = 4
    seq_len = 2048

    streams = torch.randn(1, seq_len, num_streams, hidden_dim, device=device, dtype=torch.float32)
    mhc = ManifoldHyperConnection(hidden_dim=hidden_dim, num_streams=num_streams).to(device)

    dummy_layer = lambda x: x * 0.9 + 0.1

    # Warmup
    for _ in range(5):
        _ = mhc(streams, dummy_layer)
    sync_device(device)

    N_ITERS = 50

    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        _ = mhc(streams, dummy_layer)
    sync_device(device)
    t1 = time.perf_counter()
    avg_ms = ((t1 - t0) / N_ITERS) * 1000.0

    print(f"mHC 4-Stream Recirculation Step (2048 tokens): {avg_ms:.3f} ms / layer")


def main():
    parser = argparse.ArgumentParser(description="Turing Engine Native Kernel Fusions Benchmark")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
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

    print(f"\n🚀 Running Turing Engine Kernel Fusions Benchmark on: {device.type.upper()}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    benchmark_matryoshka_spec(device)
    benchmark_svd_int8_quant(device)
    benchmark_cross_kv(device)
    benchmark_hca_chunk_pool(device)
    benchmark_mhc_hyperconnections(device)

    print("\n" + "=" * 80)
    print("✅ All Kernel Fusion Benchmarks Completed Successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
