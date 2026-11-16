#!/usr/bin/env python3
"""
W6.3: k-slot cache pooling (turing/core/kslot_pooling.py KSlotCachePooler) against int8 / int4 KV and against
dropping the oldest tokens, at matched cache bytes, by next-token NLL of a WikiText-2 continuation.

Each window: C context tokens, then D continuation tokens; every arm scores the same D - 1 next-token predictions
inside the continuation, on top of a different representation of the context:
  exact          the model's own prefill, native precision
  int8 / int4    the prefill through the paged cache in int8 / int4 (kv_cache_dtype; the continuation's own K/V too)
  kslot_k        KSlotCachePooler(num_slots=k) over the exact cache: RoPE stripped from K (content space), pooled
                 per layer and KV head into k slots, RoPE re-applied at positions 0 .. k-1, written into the paged
                 cache; the continuation follows at position k. The pooler is used as shipped: its slot queries are a
                 seeded random init (the library has no training procedure for them).
  last_m         only the last m context tokens, prefilled at positions 0 .. m-1 (the oldest C - m dropped)
  none           no context
Budgets: k = m = C * bytes_per_token(int8) / bytes_per_token(fp16) (the int8 cache's bytes), the same for int4, and the
small budgets in --budgets. Native caches are counted at fp16 (the CUDA serving dtype; these runs compute in fp32).

python scripts/eval_kslot.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import json

import torch

from eval_xkv_transfer import (cache_from_kv, continuation_nll, env_info, load_model, mean_se, pick_device,
                               prefill_kv, token_windows)
from turing.core.cross_model_kv import RoPEContentDecoupler
from turing.core.kslot_pooling import KSlotCachePooler
from turing.core.kv_cache import KVCache, kv_bytes_per_token


@torch.inference_mode()
def pooled_kv(pooler, keys, values, theta):
    """keys (after RoPE), values: per layer [1, N, H, D] -> per layer [1, k, H, D], keys re-rotated at 0 .. k-1."""
    content = [RoPEContentDecoupler.strip_rope(k, base=theta) for k in keys]
    sk = torch.stack(content, 1).transpose(2, 3)          # [1, L, H, N, D]
    sv = torch.stack(values, 1).transpose(2, 3)
    pk, pv = pooler(sk, sv)                                # [1, L, H, k, D]
    pk, pv = pk.transpose(2, 3), pv.transpose(2, 3)        # [1, L, k, H, D]
    L = len(keys)
    return ([RoPEContentDecoupler.apply_rope(pk[:, i].contiguous(), base=theta) for i in range(L)],
            [pv[:, i].contiguous() for i in range(L)])


@torch.inference_mode()
def quantized_nll(model, kv_dtype, ctx, cont):
    model.config.kv_cache_dtype = kv_dtype
    cache = KVCache(model.kv_pool())
    model(ctx, past_key_values=cache, logits_to_keep=1)
    out = continuation_nll(model, cache, cont)
    cache.free()
    model.config.kv_cache_dtype = "native"
    return out


@torch.inference_mode()
def run_model(name, args, device):
    model, tok = load_model(name, device)
    cfg = model.config
    C = args.context
    per = lambda dt: kv_bytes_per_token(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, dt, 2)  # noqa: E731
    budgets = {"int8": round(C * per("int8") / per("native")), "int4": round(C * per("int4") / per("native"))}
    sizes = sorted({budgets["int8"], budgets["int4"], *[int(x) for x in args.budgets.split(",") if x]}, reverse=True)
    torch.manual_seed(args.seed)
    poolers = {k: KSlotCachePooler(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, num_slots=k).to(device).eval()
               for k in sizes}
    wins = token_windows(tok, "test", C + args.cont, args.windows)
    arms = {}
    for w in wins:
        ctx, cont = w[None, :C].to(device), w[None, C:].to(device)
        res = {}
        cache, keys, values = prefill_kv(model, ctx)
        res["exact"] = continuation_nll(model, cache, cont)
        res["none"] = continuation_nll(model, None, cont)
        for dt in ("int8", "int4"):
            res[dt] = quantized_nll(model, dt, ctx, cont)
        for k in sizes:
            c = cache_from_kv(model, *pooled_kv(poolers[k], keys, values, cfg.rope_theta))
            res[f"kslot_{k}"] = continuation_nll(model, c, cont)
            c.free()
            res[f"last_{k}"] = _last(model, ctx[:, C - k:], cont)
        for a, v in res.items():
            arms.setdefault(a, []).append(v)
    rows = []
    native_bytes = C * per("native")
    for a, v in arms.items():
        m, se = mean_se(v)
        d, dse = mean_se([x - y for x, y in zip(v, arms["exact"])])
        if a in ("int8", "int4"):
            nbytes = C * per(a)
        elif a.startswith(("kslot_", "last_")):
            nbytes = int(a.split("_")[1]) * per("native")
        else:
            nbytes = native_bytes if a == "exact" else 0
        rows.append(dict(model=name, arm=a, nll=m, nll_se=se, delta_vs_exact=d, delta_se=dse, n=len(v),
                         context_bytes=nbytes, bytes_vs_fp16=nbytes / native_bytes))
    print(f"\n{name}: {len(wins)} windows, context {C}, continuation {args.cont}; matched budgets {budgets}")
    for r in rows:
        print(f"  {r['arm']:<11} bytes {r['bytes_vs_fp16']:6.3f}x  nll {r['nll']:.4f} ± {r['nll_se']:.4f}  "
              f"delta {r['delta_vs_exact']:+.4f} ± {r['delta_se']:.4f}")
    del model
    return dict(model=name, matched_budgets=budgets, rows=rows)


def _last(model, tail, cont):
    """Prefill only the kept tail at positions 0 .. m-1, then score the continuation on top of it."""
    cache = KVCache(model.kv_pool())
    model(tail, past_key_values=cache, logits_to_keep=1)
    out = continuation_nll(model, cache, cont)
    cache.free()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--budgets", default="64,4", help="extra slot / token budgets besides the int8 and int4 matches")
    ap.add_argument("--seed", type=int, default=0, help="seed of the poolers' random slot queries")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = pick_device(a.device)
    results = [run_model(m, a, device) for m in a.models.split(",") if m]
    out = a.out or f"results_kslot_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env_info(device), args=vars(a), dtype="float32",
                       text="WikiText-2 raw test, consecutive windows", models=results), f, indent=2)
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
