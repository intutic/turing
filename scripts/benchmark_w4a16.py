#!/usr/bin/env python3
"""
Where the fused int4 GEMM beats dequantize + cuBLAS fp16, by row count M (decode is M = batch, prefill M = tokens).
For each layer shape of the model and each M: the Triton W4A16 kernel, cuBLAS on fp16 weights, and QuantLinear's
large-M path (dequantize into scratch + cuBLAS), the int4 kernel without its K split, and Marlin (marlin_us) where
the optional extension is built (TURING_BUILD_MARLIN=1). GPU time per call from CUDA-graph replays (median of
--iters). The crossovers set turing.models.quant.INT4_MAX_ROWS and MARLIN_MIN_ROWS / MARLIN_MAX_ROWS. Writes
results_w4a16_crossover_cuda.json.

python scripts/benchmark_w4a16.py --shapes 2048x2048,2048x8192,8192x2048
"""
import _bootstrap  # noqa: F401,E402
import argparse
import json
import platform
import statistics

import torch
import torch.nn.functional as F

from turing.kernels.marlin import marlin_gemm, marlin_pack, marlin_supported
from turing.kernels.triton_w4a16 import dequantize_cuda, w4a16_gemm, w4a16_gemv
from turing.models.quant import dequantize, pack, rtn_quantize


def timed(fn, iters, reps=20):
    """GPU microseconds per call, from CUDA-graph replays of `reps` back-to-back calls (serving decode replays graphs;
    timing each launch from Python would measure the launcher once the kernel is faster than it)."""
    for _ in range(3):  # compilation and autotuning, outside the capture
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / reps)
    return statistics.median(ts)


def no_split(fn):
    """fn with the int4 kernel's K split forced to 1 (one program per output tile walks all of K)."""
    import turing.kernels.triton_sparse_gemv as sg

    def run():
        saved, sg._splits = sg._splits, (lambda *a: 1)
        try:
            return fn()
        finally:
            sg._splits = saved
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="1536x2048,1536x1536,1536x17920,8960x1536,1536x151936,2048x2048,2048x8192,8192x2048,4096x4096,4096x14336,14336x4096",
                    help="KxN of y = x W^T (in x out features)")
    ap.add_argument("--rows", default="1,2,4,8,16,32,64,128,256,512")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="benchmarks/results/results_w4a16_crossover_cuda.json")
    a = ap.parse_args()
    rows = []
    for shape in a.shapes.split(","):
        K, N = map(int, shape.split("x"))
        codes, s, z = rtn_quantize(torch.randn(N, K))
        q, s, z = pack(codes).cuda(), s.cuda(), z.cuda()
        w16 = dequantize(q, s, z, torch.float16)
        scratch = torch.empty_like(w16)
        mw = marlin_pack(q, s, z, torch.float16) if marlin_supported(N, K, 64, torch.float16, q.device) else None
        for M in map(int, a.rows.split(",")):
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            r = dict(K=K, N=N, M=M,
                     int4_kernel_us=timed(lambda: w4a16_gemm(x, q, s, z), a.iters),
                     int4_kernel_no_split_us=timed(no_split(lambda: w4a16_gemm(x, q, s, z)), a.iters),
                     fp16_cublas_us=timed(lambda: F.linear(x, w16), a.iters),
                     dequant_cublas_us=timed(lambda: F.linear(x, dequantize_cuda(q, s, z, scratch)), a.iters))
            if M <= 16:  # the few-row GEMV kernel (sets GEMV_MAX_ROWS)
                r["gemv_us"] = timed(lambda: w4a16_gemv(x, q, s, z), a.iters)
            if mw is not None:
                r["marlin_us"] = timed(lambda: marlin_gemm(x, mw), a.iters)
            print(json.dumps(r), flush=True)
            rows.append(r)
    env = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(0), platform=platform.platform())
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
