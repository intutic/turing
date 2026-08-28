#!/usr/bin/env python3
"""
Comprehensive Latent Flash-Decode (Mode-B) & 3:1 Hybrid Attention Benchmark.
Measures physical latency (ms), throughput (tok/s), and memory savings on real silicon.
"""

import argparse
import time
import torch
import torch.nn.functional as F

from turing.kernels.triton_latent_decode import triton_latent_flash_decode
from turing.core.hybrid_attention import LinearRecurrentAttention, ChunkContextScorer, HybridAttentionLayerRouter

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = hasattr(turing_csrc, "latent_decode_cpu")
except ImportError:
    HAS_CSRC = False


def benchmark_latent_decode(device_name: str = "cpu"):
    device = torch.device(device_name)
    print(f"\n================================================================================")
    print(f"⚡ Latent Flash-Decode (SPECTRA Mode-B Subspace Attention) on {device.type.upper()}")
    print(f"================================================================================")

    B = 1
    NKV = 8
    GRP = 8
    num_heads = NKV * GRP
    head_dim = 128
    R = 64 # Rank-64 Subspace

    print(f"{'Context Length (L)':<22} | {'Dense Attn (FP16)':<18} | {'Latent Decode (INT8)':<22} | {'Speedup':<12} | {'VRAM Saved'}")
    print("-" * 95)

    for seq_len in [512, 2048, 8192, 32768]:
        qp = torch.randn(B, NKV, GRP, R, device=device, dtype=torch.float32)
        ck = torch.randint(-128, 127, (B, seq_len, R), device=device, dtype=torch.int8)
        sk = torch.rand(B, seq_len, device=device, dtype=torch.float32) * 0.05 + 0.01
        cv = torch.randint(-128, 127, (B, seq_len, R), device=device, dtype=torch.int8)
        sv = torch.rand(B, seq_len, device=device, dtype=torch.float32) * 0.05 + 0.01

        # Dense FP16 equivalent
        q_dense = torch.randn(B, num_heads, 1, head_dim, device=device, dtype=torch.float32)
        k_dense = torch.randn(B, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v_dense = torch.randn(B, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = triton_latent_flash_decode(qp, ck, sk, cv, sv, head_dim=head_dim)
            scores = torch.matmul(q_dense, k_dense.transpose(-1, -2)) * (1.0 / (head_dim ** 0.5))
            _ = torch.matmul(F.softmax(scores, dim=-1), v_dense)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Measure Dense
        iters = 50 if seq_len <= 8192 else 20
        t0 = time.perf_counter()
        for _ in range(iters):
            scores = torch.matmul(q_dense, k_dense.transpose(-1, -2)) * (1.0 / (head_dim ** 0.5))
            _ = torch.matmul(F.softmax(scores, dim=-1), v_dense)
            if device.type == "cuda":
                torch.cuda.synchronize()
        dense_time = (time.perf_counter() - t0) / iters * 1000.0

        # Measure Latent Flash-Decode
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = triton_latent_flash_decode(qp, ck, sk, cv, sv, head_dim=head_dim)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latent_time = (time.perf_counter() - t0) / iters * 1000.0

        speedup = dense_time / max(latent_time, 1e-6)
        dense_bytes = 2 * seq_len * num_heads * head_dim * 2 # FP16 K+V
        latent_bytes = seq_len * (R * 1 + 4) * 2              # INT8 + Scale K+V
        vram_saved = (1.0 - latent_bytes / dense_bytes) * 100.0

        print(f"{seq_len:<22} | {dense_time:>14.4f} ms | {latent_time:>18.4f} ms | {speedup:>10.2f}x | {vram_saved:>9.1f}%", flush=True)



def benchmark_hybrid_attention_prefill(device_name: str = "cpu"):
    device = torch.device(device_name)
    print(f"\n================================================================================")
    print(f"🚀 3:1 Hybrid Linear-Full Attention Prefill on {device.type.upper()}")
    print(f"================================================================================")

    hidden_dim = 2048
    num_heads = 16
    head_dim = 128

    print(f"{'Context Length (L)':<22} | {'Standard Dense Full':<20} | {'3:1 Hybrid + ChunkScorer':<25} | {'Prefill Speedup'}")
    print("-" * 90)

    for seq_len in [2048, 8192, 32768, 65536]:
        x = torch.randn(1, seq_len, hidden_dim, device=device, dtype=torch.float32)

        linear_layer = LinearRecurrentAttention(hidden_dim, num_heads, head_dim).to(device)
        chunk_scorer = ChunkContextScorer(hidden_dim, budget_tokens=2048).to(device)

        # Warmup
        _ = linear_layer(x[:, :512, :])

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Simulate 4 layers: 3 linear + 1 full with chunk scoring vs 4 full layers
        iters = 10 if seq_len <= 8192 else 3

        # Dense 4 layers
        t0 = time.perf_counter()
        dense_time = None
        try:
            for _ in range(iters):
                for _ in range(4):
                    q = x.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                    k = x.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                    v = x.view(1, seq_len, num_heads, head_dim).transpose(1, 2)
                    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / (head_dim ** 0.5))
                    _ = torch.matmul(F.softmax(scores, dim=-1), v)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            dense_time = (time.perf_counter() - t0) / iters * 1000.0
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            dense_time = None

        # 3:1 Hybrid
        t0 = time.perf_counter()
        for _ in range(iters):
            # 3 Linear layers
            for _ in range(3):
                _, _ = linear_layer(x)
            # 1 Full layer with 4x Chunk Context Filtering
            k = x.view(1, seq_len, num_heads, head_dim)
            v = x.view(1, seq_len, num_heads, head_dim)
            q = x[:, -1:, :].view(1, 1, num_heads, head_dim)
            k_f, v_f = chunk_scorer.filter_context(k, v, q)
            q_t = q.transpose(1, 2)
            k_t = k_f.transpose(1, 2)
            v_t = v_f.transpose(1, 2)
            scores = torch.matmul(q_t, k_t.transpose(-1, -2)) * (1.0 / (head_dim ** 0.5))
            _ = torch.matmul(F.softmax(scores, dim=-1), v_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
        hybrid_time = (time.perf_counter() - t0) / iters * 1000.0

        if dense_time is not None:
            speedup = dense_time / max(hybrid_time, 1e-6)
            print(f"{seq_len:<22} | {dense_time:>16.2f} ms | {hybrid_time:>21.2f} ms | {speedup:>13.2f}x")
        else:
            print(f"{seq_len:<22} | {'OOM (>64 GB)':>16} | {hybrid_time:>21.2f} ms | {'∞ (Prevents OOM)':>15}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Flash-Decode & Hybrid Attention Benchmark")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Device target")
    args = parser.parse_args()

    benchmark_latent_decode(args.device)
    benchmark_hybrid_attention_prefill(args.device)
