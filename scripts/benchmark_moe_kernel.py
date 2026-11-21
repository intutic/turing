#!/usr/bin/env python3
"""
The grouped MoE kernels (turing/kernels/triton_moe.py) alone, on one layer of expert weights: device time per call
(a CUDA-graph replay, as in serving) for each block and tile size, the per-expert loop it replaced and the per-expert
GEMM path used for prefill-sized inputs (both eager: they sync with the host), and the read rate reached: bytes of the experts hit over the time, next to the GPU's measured read
bandwidth. Weights and routing are random (uniform top-k, the most experts a batch can hit); the shapes default to
OLMoE-1B-7B's (64 experts, top-8, hidden 2048, expert width 1024). --w4 times the kernels for experts packed in
4 bits (round to nearest), against the loop on the dequantized weights.

python scripts/benchmark_moe_kernel.py --tokens 1,8,32,64,512
python scripts/benchmark_moe_kernel.py --w4 --block-k 32,64
"""
import _bootstrap  # noqa: F401,E402
import argparse
import itertools
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from turing.core.probe import read_bandwidth_gb_s
from turing.kernels.triton_moe import fused_moe, fused_moe_w4, gemm_moe


def loop(x, gate_up, down, weights, experts):
    """SparseMoE's per-expert loop."""
    out = torch.zeros_like(x)
    for e in torch.unique(experts).tolist():
        token, slot = torch.where(experts == e)
        gate, up = F.linear(x[token], gate_up[e]).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, down[e])
        out.index_add_(0, token, (y * weights[token, slot, None]).to(out.dtype))
    return out


def median_ms(run, iters):
    ts = []
    for _ in range(5):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) / iters * 1e3)
    return statistics.median(ts)


def graph_ms(fn, iters):
    """(ms per replay of fn captured in a CUDA graph, fn's output)."""
    fn()
    fn()  # compilation and autotuning happen outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    for _ in range(3):
        graph.replay()
    return median_ms(graph.replay, iters), out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", default="1,8,32,64,512", help="token counts: decode batch sizes and a 512-token prefill")
    ap.add_argument("--experts", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--block-m", default="16,32,64")
    ap.add_argument("--block-n", default="32,64,128")
    ap.add_argument("--block-k", default="32,64,128")
    ap.add_argument("--warps", default="4,8")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--layers", type=int, default=1,
                    help="copies of the layer run one after another in each call, as a model's layers are; with a few, "
                         "the weights a call reads no longer fit the GPU's cache (4-bit experts at small batch do)")
    ap.add_argument("--w4", action="store_true", help="experts packed in 4 bits (block-k must divide 64 for blocks of 16 rows or more)")
    ap.add_argument("--out", default=None, help="default benchmarks/results/results_moe_kernel_sweep[_w4]_cuda.json")
    a = ap.parse_args()
    a.out = a.out or f"benchmarks/results/results_moe_kernel_sweep{'_w4' if a.w4 else ''}_cuda.json"
    dev = torch.device("cuda")
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    E, K, H, Fd = a.experts, a.top_k, a.hidden, a.ffn
    g = torch.Generator(device="cuda").manual_seed(0)
    gate_up = torch.randn(E, 2 * Fd, H, device=dev, dtype=dtype, generator=g) * 0.02
    down = torch.randn(E, H, Fd, device=dev, dtype=dtype, generator=g) * 0.02
    expert_bytes = (gate_up[0].numel() + down[0].numel()) * gate_up.element_size()
    packed = None
    if a.w4:  # pack, and keep the dequantized weights for the reference loop
        from turing.models.quant import dequantize, pack, rtn_quantize
        packed = []
        for w in (gate_up, down):
            parts = []
            for i in range(0, E, 8):
                c, sc, z = rtn_quantize(w[i:i + 8].float())
                parts.append((pack(c), sc, z))
                w[i:i + 8] = dequantize(*parts[-1], dtype)
            packed.append(tuple(torch.cat(t) for t in zip(*parts)))
        expert_bytes = sum(t[0].numel() * t.element_size() for trio in packed for t in trio)
    # every layer its own memory (the same values: only the first is checked against the loop)
    dense = [(gate_up, down)] + [(gate_up.clone(), down.clone()) for _ in range(a.layers - 1)] if not a.w4 else None
    packs = [packed] + [[tuple(t.clone() for t in trio) for trio in packed] for _ in range(a.layers - 1)] if a.w4 else None
    bw = read_bandwidth_gb_s(dev)
    ints = lambda s: [int(v) for v in s.split(",")]  # noqa: E731
    # the few-row 4-bit kernels (block_m < 16) keep block_m x block_n x block_k / 2 sums in registers
    configs = [c for c in itertools.product(ints(a.block_m), ints(a.block_n), ints(a.block_k), ints(a.warps))
               if c[0] >= 16 or c[0] * c[1] * c[2] <= 8192]
    rows = []
    for T in ints(a.tokens):
        x = torch.randn(T, H, device=dev, dtype=dtype, generator=g)
        experts = torch.rand(T, E, device=dev, generator=g).topk(K, dim=-1).indices
        weights = torch.rand(T, K, device=dev, generator=g)
        weights = weights / weights.sum(-1, keepdim=True)
        hit = int(experts.unique().numel())
        ref = loop(x, gate_up, down, weights, experts)
        loop_ms = median_ms(lambda: loop(x, gate_up, down, weights, experts), max(3, a.iters // 5))
        gemm_ms = median_ms(lambda: gemm_moe(x, weights, experts, E, lambda e: (gate_up[e], down[e]), F.silu),
                            max(3, a.iters // 5))
        base = dict(tokens=T, pairs=T * K, experts_hit=hit, loop_eager_ms=round(loop_ms, 4),
                    gemm_eager_ms=round(gemm_ms, 4))
        best = None
        default_ms, _ = graph_ms((lambda: [fused_moe_w4(x, q[0], q[1], weights, experts, 64) for q in packs][0])
                                 if a.w4 else (lambda: [fused_moe(x, gu, dn, weights, experts) for gu, dn in dense][0]),
                                 a.iters)  # the block and tile sizes the library picks
        for bm, bn, bk, w in configs:
            row = dict(base, block_m=bm, block_n=bn, block_k=bk, num_warps=w)
            try:
                kw = dict(block_m=bm, block_n=bn, block_k=bk, num_warps=w)
                ms, out = graph_ms((lambda: [fused_moe_w4(x, q[0], q[1], weights, experts, 64, **kw) for q in packs][0])
                                   if a.w4 else
                                   (lambda: [fused_moe(x, gu, dn, weights, experts, **kw) for gu, dn in dense][0]), a.iters)
                ms /= a.layers
                row.update(ms=round(ms, 4), read_gb_per_s=round(hit * expert_bytes / 1e9 / (ms / 1e3), 1),
                           max_abs_diff_vs_loop=float((out.float() - ref.float()).abs().max()))
                if best is None or ms < best["ms"]:
                    best = row
            except Exception as e:  # a tile shape Triton rejects or that runs out of memory is recorded, not hidden
                row.update(error=f"{type(e).__name__}: {str(e)[:200]}")
            rows.append(row)
            torch.cuda.empty_cache()
        rows.append(dict(base, default=True, ms=round(default_ms / a.layers, 4)))
        print(json.dumps(dict(tokens=T, experts_hit=hit, loop_eager_ms=base["loop_eager_ms"],
                              gemm_eager_ms=base["gemm_eager_ms"],
                              default_ms=rows[-1]["ms"], best=best)), flush=True)
    env = dict(platform=platform.platform(), torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
               read_bandwidth_gb_s=bw, dtype=a.dtype, experts=E, top_k=K, hidden=H, ffn=Fd, expert_bytes=expert_bytes,
               layers=a.layers,
               weights="4-bit" if a.w4 else a.dtype)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
