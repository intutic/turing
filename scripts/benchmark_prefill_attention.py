#!/usr/bin/env python3
"""
Paged prefill attention alone (speed plan P5.2, decision D10): Turing's Triton kernel (triton_kv_prefill) against the
gather + SDPA path it replaced and, when the interpreter has it, FlashInfer's paged prefill on the same native fp16
pages. One request, a chunk of --chunk queries at each start offset (the cached prefix before it), Qwen2.5-1.5B's
attention shape by default. Milliseconds per layer call, median of --iters.

python scripts/benchmark_prefill_attention.py                       # Turing kernel and the old path
~/vllm-venv/bin/python scripts/benchmark_prefill_attention.py --out results_prefill_attention_flashinfer_cuda.json
"""
import _bootstrap  # noqa: F401,E402
import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from turing.core.kv_cache import KVCache, KVPool


def timed(fn, iters):
    for _ in range(3):
        fn()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heads", default="12,2,128", help="query heads, KV heads, head dim")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--starts", default="0,6144,14336")
    ap.add_argument("--kv", default="native,int8,int4")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="results_prefill_attention_cuda.json")
    a = ap.parse_args()
    H, Hkv, D = map(int, a.heads.split(","))
    from turing.kernels.triton_kv_prefill import triton_pool_prefill
    try:
        import flashinfer
    except ImportError:
        flashinfer = None
    rows = []
    g = torch.Generator(device="cuda").manual_seed(0)
    for kv in a.kv.split(","):
        for start in map(int, a.starts.split(",")):
            total = start + a.chunk
            pool = KVPool(1, Hkv, D, total // 16 + 8, torch.float16, "cuda", kv)
            c = KVCache(pool)
            c.reserve(total)
            pool.append(0, c, torch.randn(1, Hkv, total, D, device="cuda", generator=g).half(),
                        torch.randn(1, Hkv, total, D, device="cuda", generator=g).half())
            q = torch.randn(1, H, a.chunk, D, device="cuda", generator=g).half()
            row = dict(kv_cache_dtype=kv, start=start, chunk=a.chunk, heads=[H, Hkv, D])
            row["turing_ms"] = timed(lambda: triton_pool_prefill(pool, 0, q, c.slot, start, D ** -0.5), a.iters)

            def old():
                k, v = pool.read(0, c, total)
                k, v = k.repeat_interleave(H // Hkv, 1), v.repeat_interleave(H // Hkv, 1)
                qpos = start + torch.arange(a.chunk, device="cuda")[:, None]
                mask = torch.arange(total, device="cuda") <= qpos
                return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            row["gather_sdpa_ms"] = timed(old, a.iters)
            if flashinfer is not None and kv == "native":
                try:  # FlashInfer's NHD paged layout: [pages, 2, 16, Hkv, D] (a copy of our pages, untimed)
                    n_pages = -(-total // 16)
                    phys = pool.page_table[c.slot, :n_pages].long()
                    kc, _, _, _ = pool.layer_fields(0, "k")
                    vc, _, _, _ = pool.layer_fields(0, "v")
                    paged = torch.stack((kc[phys], vc[phys]), 1).permute(0, 1, 3, 2, 4).contiguous()
                    ws = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
                    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
                    i32 = dict(dtype=torch.int32, device="cuda")
                    w.plan(torch.tensor([0, a.chunk], **i32), torch.tensor([0, n_pages], **i32),
                           torch.arange(n_pages, **i32), torch.tensor([total - (n_pages - 1) * 16], **i32),
                           H, Hkv, D, 16, causal=True, q_data_type=torch.float16)
                    qf = q[0].transpose(0, 1).contiguous()  # [chunk, H, D]
                    row["flashinfer_ms"] = timed(lambda: w.run(qf, paged), a.iters)
                    row["flashinfer"] = flashinfer.__version__
                except Exception as e:  # an API change between versions: record it, keep the other numbers
                    row["flashinfer_error"] = f"{type(e).__name__}: {e}"[:300]
            print(json.dumps(row), flush=True)
            rows.append(row)
            del pool, c
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, rows=rows,
                       note="milliseconds per call (one layer), median; causal chunk after `start` cached tokens"),
                  f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
