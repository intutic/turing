"""
Empirical Benchmark Suite: Native C++20 AVX2 SIMD & In-SRAM Fused Triton GPU Kernels.
Measures physical micro-benchmark speedups and latency reductions.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from turing.serving.traffic import PrefixHashRouter
from turing.serving.spec_gate import SpecExactParityVerifier
from turing.kernels.triton_vram_hash import compute_fast_tensor_hash, fused_tensor_checksum_cuda
from turing.kernels.triton_quadtree_mrp import fused_quadtree_mrp_cuda
from turing.kernels.triton_chunk_filter import fused_chunk_context_filter_cuda
from turing.kernels.triton_gated_zero_identity import fused_gated_zero_identity_cuda
from turing.core.speculation import QuadtreeMRPSpeculator, MatryoshkaDraftHead
from turing.core.kslot_pooling import GatedZeroIdentityHead
from turing.core.hybrid_attention import ChunkContextScorer


def benchmark_prefix_token_hashing(num_iters: int = 50000):
    tokens = list(range(1000, 1128))
    tok_arr = np.array(tokens, dtype=np.int32)
    router = PrefixHashRouter(window=128)

    # 1. Pure Python baseline
    t0 = time.perf_counter()
    offset_basis = 0xcbf29ce484222325
    prime = 0x100000001b3
    for _ in range(num_iters):
        h = offset_basis
        for t in tokens:
            h ^= (t & 0xFF)
            h *= prime
            h &= 0xFFFFFFFFFFFFFFFF
    t_py = (time.perf_counter() - t0) / num_iters * 1e6 # in microseconds

    # 2. Native C++ AVX2 SIMD
    t0 = time.perf_counter()
    for _ in range(num_iters):
        h_cpp = turing_csrc.compute_prefix_hash_fast(tok_arr, 128)
    t_cpp = (time.perf_counter() - t0) / num_iters * 1e6 # in microseconds

    return t_py, t_cpp, t_py / max(t_cpp, 1e-9)


def benchmark_spec_parity_verification(num_iters: int = 20000, num_tokens: int = 256):
    spec = list(range(num_tokens))
    plain = list(range(num_tokens))
    s_arr = np.array(spec, dtype=np.int32)
    p_arr = np.array(plain, dtype=np.int32)

    # 1. Pure Python baseline
    t0 = time.perf_counter()
    for _ in range(num_iters):
        for i in range(num_tokens):
            if spec[i] != plain[i]:
                break
    t_py = (time.perf_counter() - t0) / num_iters * 1e6

    # 2. Native C++ AVX2 SIMD
    t0 = time.perf_counter()
    for _ in range(num_iters):
        passed, count, div = turing_csrc.verify_greedy_parity_fast(s_arr, p_arr)
    t_cpp = (time.perf_counter() - t0) / num_iters * 1e6

    return t_py, t_cpp, t_py / max(t_cpp, 1e-9)


def benchmark_lineage_tensor_hashing(device: str = "cpu", num_iters: int = 500):
    tensors = [torch.randn(32, 128, 64, device=device) for _ in range(16)] # 16 layers, 10MB KV cache
    
    # 1. Baseline Python tobytes + blake2b
    import hashlib
    t0 = time.perf_counter()
    for _ in range(num_iters):
        h = hashlib.blake2b()
        for t in tensors:
            b = t.detach().cpu().contiguous().numpy().tobytes()
            h.update(b)
        res = h.hexdigest()
    t_py = (time.perf_counter() - t0) / num_iters * 1e3 # in milliseconds

    # 2. Fast In-VRAM / Pointer Checksum
    t0 = time.perf_counter()
    for _ in range(num_iters):
        res_fast = compute_fast_tensor_hash(tensors)
    t_fast = (time.perf_counter() - t0) / num_iters * 1e3 # in milliseconds

    return t_py, t_fast, t_py / max(t_fast, 1e-9)


def benchmark_quadtree_speculator(device: str = "cpu", num_iters: int = 200):
    hidden_dim = 2048
    vocab_size = 32000
    slice_width = 1024
    
    speculator = QuadtreeMRPSpeculator(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        slice_widths=[1024, 2048],
    ).to(device)
    
    hidden = torch.randn(1, hidden_dim, device=device)
    
    # Warmup
    for _ in range(10):
        speculator.generate_speculative_tree(hidden, slice_width=slice_width)
        if device == "cuda":
            torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        speculator.generate_speculative_tree(hidden, slice_width=slice_width)
        if device == "cuda":
            torch.cuda.synchronize()
    t_elapsed = (time.perf_counter() - t0) / num_iters * 1e3 # in ms
    return t_elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"\n==========================================================================")
    print(f"⚡ Turing Engine Native SIMD & Fused Kernel Micro-Benchmark (Device: {device})")
    print(f"==========================================================================")

    # 1. Prefix Token Hashing
    t_py, t_cpp, speedup = benchmark_prefix_token_hashing()
    print(f"\n1. Prefix Token Hasher (128 Tokens):")
    print(f"   - Python Baseline:   {t_py:8.3f} µs / hash")
    print(f"   - Native C++ AVX2:   {t_cpp:8.3f} µs / hash")
    print(f"   - Measured Speedup:  {speedup:8.2f}x Faster")

    # 2. Speculative Parity Verifier
    t_py, t_cpp, speedup = benchmark_spec_parity_verification()
    print(f"\n2. Speculative Parity Verifier (256 Tokens):")
    print(f"   - Python Baseline:   {t_py:8.3f} µs / pass")
    print(f"   - Native C++ SIMD:   {t_cpp:8.3f} µs / pass")
    print(f"   - Measured Speedup:  {speedup:8.2f}x Faster")

    # 3. Lineage KV Cache Hashing
    t_py, t_fast, speedup = benchmark_lineage_tensor_hashing(device=device)
    print(f"\n3. Multi-Turn Lineage KV Hashing (16 Layers, 10MB Cache):")
    print(f"   - Python tobytes:    {t_py:8.3f} ms / turn")
    print(f"   - Fast Pointer/VRAM: {t_fast:8.3f} ms / turn")
    print(f"   - Measured Speedup:  {speedup:8.2f}x Faster")

    # 4. Quadtree Speculative Draft Latency
    t_spec = benchmark_quadtree_speculator(device=device)
    print(f"\n4. Quadtree MRP Speculative Draft Head (W_slice=1024, Vocab=32K, 21 Nodes):")
    print(f"   - Fused Candidate Latency: {t_spec:8.3f} ms / draft pass")

    print(f"\n==========================================================================")
    print(f"✅ Micro-Benchmark Suite Completed Successfully.")
    print(f"==========================================================================\n")


if __name__ == "__main__":
    main()
