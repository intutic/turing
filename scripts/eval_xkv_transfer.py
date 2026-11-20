#!/usr/bin/env python3
"""
W6.1: cross-model KV transfer (turing/experimental/cross_model_kv.py CrossModelKVPipeline, closed-form ridge W* with RoPE
content decoupling) on real models that share a tokenizer, measured by what the receiver does with the result.

Calibration: n_calib windows of C tokens from WikiText-2 train. Both models prefill each window through the Turing
paged KV cache; their caches (K after RoPE, V) are the paired data, and CrossModelKVPipeline.fit solves every target
layer's K and V maps in closed form (on the CPU, fp32).

Evaluation: n_eval windows of C + D tokens from WikiText-2 test. The receiver scores the D continuation tokens (the
D - 1 next-token predictions inside the continuation, identical targets in every arm) on top of
  exact     its own prefill of the C context tokens
  transfer  the sender's prefill of the context, mapped by W* and written into the receiver's paged cache
  none      no context (the continuation alone, from position 0)
plus `sender` (the sender scoring the continuation on its own prefill; context for how much the sender knows).
Reported: mean NLL ± SE over windows, paired deltas vs exact, the fraction of the context benefit kept
(none - transfer) / (none - exact) (ratio of window means; SE by the delta method), the relative error of the
transferred K and V against the receiver's own cache, and wall times (shared, loaded machine: context only).

python scripts/eval_xkv_transfer.py --pairs HuggingFaceTB/SmolLM2-360M:HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-0.5B:Qwen/Qwen2.5-1.5B
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import gc
import json
import math
import platform
import subprocess
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.experimental.cross_model_kv import CrossModelKVPipeline
from turing.core.kv_cache import KVCache
from turing.models.hf_loader import RealHuggingFaceLoader


# ---------------------------------------------------------------- shared helpers (eval_lineage / eval_kslot import them)

def pick_device(choice="auto"):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(choice)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def env_info(device):
    chip = platform.processor()
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return dict(device=device.type, torch=torch.__version__, platform=platform.platform(),
                processor=torch.cuda.get_device_name(0) if device.type == "cuda" else chip,
                note="shared, loaded machine: wall times are context only")


def load_model(name, device, dtype=torch.float32, kv_pool_gib=1.0):
    """Turing model with HF's exact logits (RealHuggingFaceLoader, no pruning, no recirculation) and its tokenizer."""
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, low_cpu_mem_usage=True).eval()
    model = RealHuggingFaceLoader.convert_hf_model(hf, device=str(device), dtype=dtype, name=name).eval()
    del hf
    gc.collect()
    model.config.kv_pool_gib = kv_pool_gib
    return model, tok


def token_windows(tok, split, T, n, offset=0):
    """n consecutive windows of T tokens from WikiText-2 raw `split`, skipping the first `offset` windows."""
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    n = min(n, len(ids) // T - offset)
    return ids[offset * T:(offset + n) * T].view(n, T)


@torch.inference_mode()
def prefill_kv(model, ids):
    """Prefill ids [1, L] through a fresh paged cache; returns that cache and per-layer K (after RoPE), V as
    [1, L, Hkv, D] in the model dtype."""
    cache = KVCache(model.kv_pool())
    model(ids, past_key_values=cache, logits_to_keep=1)
    pool, L = cache.pool, ids.shape[1]
    kv = [pool.read(layer, cache, L) for layer in range(model.config.num_layers)]
    return cache, [k.transpose(1, 2) for k, _ in kv], [v.transpose(1, 2) for _, v in kv]


@torch.inference_mode()
def cache_from_kv(model, keys, values):
    """A paged cache holding the given per-layer K (after RoPE, positions 0 .. L-1), V [1, L, Hkv, D]."""
    pool = model.kv_pool()
    cache = KVCache(pool)
    L = keys[0].shape[1]
    cache.reserve(L)
    for layer, (k, v) in enumerate(zip(keys, values)):
        pool.append(layer, cache, k.transpose(1, 2).to(pool.dtype), v.transpose(1, 2).to(pool.dtype))
    cache.length = L
    return cache


@torch.inference_mode()
def continuation_nll(model, cache, cont):
    """Mean NLL of cont[1:] given cont[:-1] on top of `cache` (None = no context). cont [1, D]."""
    logits, _ = model(cont, past_key_values=cache)
    return F.cross_entropy(logits[0, :-1].float(), cont[0, 1:]).item()


def mean_se(v):
    v = torch.tensor(v, dtype=torch.float64)
    return v.mean().item(), (v.std().item() / math.sqrt(len(v)) if len(v) > 1 else None)


def retained(none, exact, arm):
    """(mean(none) - mean(arm)) / (mean(none) - mean(exact)) and its delta-method SE from the per-window values."""
    n = torch.tensor(none, dtype=torch.float64)
    e = torch.tensor(exact, dtype=torch.float64)
    a = torch.tensor(arm, dtype=torch.float64)
    num, den = (n - a), (n - e)
    r = num.mean() / den.mean()
    infl = (num - r * den) / den.mean()  # influence function of a ratio of means
    return r.item(), (infl.std() / math.sqrt(len(n))).item()


def rel_err(pred, ref):
    num = sum(((p.float() - r.float()) ** 2).sum() for p, r in zip(pred, ref))
    den = sum((r.float() ** 2).sum() for r in ref)
    return math.sqrt(num / den)


# ---------------------------------------------------------------- W6.1

@torch.inference_mode()
def fit_pipeline(src, tgt, tok, args, device):
    """Paired calibration caches from WikiText-2 train -> a fitted CrossModelKVPipeline (maps on the CPU)."""
    cal = token_windows(tok, "train", args.context, args.calib_windows)
    n, C = cal.shape

    def buffers(m):  # half precision on the host, filled in place: the 1.7B's 32 KV heads are big
        c = m.config
        return [[torch.empty(n, C, c.num_kv_heads, c.head_dim, dtype=torch.float16) for _ in range(c.num_layers)]
                for _ in range(2)]
    (sk, sv), (tk, tv) = buffers(src), buffers(tgt)
    for i, w in enumerate(cal):
        ids = w[None].to(device)
        for model, ks, vs in ((src, sk, sv), (tgt, tk, tv)):
            cache, k, v = prefill_kv(model, ids)
            for layer in range(model.config.num_layers):
                ks[layer][i] = k[layer][0].half().cpu()
                vs[layer][i] = v[layer][0].half().cpu()
            cache.free()
    pipe = CrossModelKVPipeline(src.config, tgt.config, top_k_layers=args.top_k, ridge_lambda=args.ridge_lambda)
    t0 = time.perf_counter()
    pipe.fit(sk, sv, tk, tv)
    fit_s = time.perf_counter() - t0
    for m in list(pipe.layer_mappers_k.values()) + list(pipe.layer_mappers_v.values()):
        m.to(device)  # the maps were solved on the CPU; apply them where the caches are
    return pipe, dict(calib_windows=len(cal), calib_tokens=cal.numel(), fit_seconds=fit_s)


@torch.inference_mode()
def run_pair(src_name, tgt_name, args, device):
    src, tok = load_model(src_name, device)
    tgt, tok_t = load_model(tgt_name, device)
    probe = "The quick brown fox, 1234 été."
    assert tok(probe).input_ids == tok_t(probe).input_ids, "sender and receiver must share a tokenizer"
    pipe, fit_info = fit_pipeline(src, tgt, tok, args, device)
    print(f"[{src_name} -> {tgt_name}] fit on {fit_info['calib_tokens']} tokens in {fit_info['fit_seconds']:.1f}s")

    wins = token_windows(tok, "test", args.context + args.cont, args.eval_windows)
    C = args.context
    arms = {a: [] for a in ("exact", "transfer", "none", "sender")}
    errs_k, errs_v, t_recv, t_send, t_map = [], [], [], [], []
    for w in wins:
        ids, cont = w[None, :C].to(device), w[None, C:].to(device)
        sync(device)
        t0 = time.perf_counter()
        cache_r, kr, vr = prefill_kv(tgt, ids)
        sync(device)
        t_recv.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cache_s, ks, vs = prefill_kv(src, ids)
        sync(device)
        t1 = time.perf_counter()
        kt, vt = pipe.transfer_cache(ks, vs)
        cache_t = cache_from_kv(tgt, kt, vt)
        sync(device)
        t2 = time.perf_counter()
        t_send.append(t1 - t0)
        t_map.append(t2 - t1)

        errs_k.append(rel_err(kt, kr))
        errs_v.append(rel_err(vt, vr))
        arms["exact"].append(continuation_nll(tgt, cache_r, cont))
        arms["transfer"].append(continuation_nll(tgt, cache_t, cont))
        arms["none"].append(continuation_nll(tgt, None, cont))
        arms["sender"].append(continuation_nll(src, cache_s, cont))
        for c in (cache_r, cache_s, cache_t):
            c.free()

    rows = []
    for arm, v in arms.items():
        m, se = mean_se(v)
        rows.append(dict(arm=arm, metric="nll", value=m, se=se, n=len(v)))
    for arm in ("transfer", "none"):
        m, se = mean_se([a - b for a, b in zip(arms[arm], arms["exact"])])
        rows.append(dict(arm=arm, metric="delta_nll_vs_exact", value=m, se=se, n=len(wins)))
    r, rse = retained(arms["none"], arms["exact"], arms["transfer"])
    rows.append(dict(arm="transfer", metric="context_benefit_retained", value=r, se=rse, n=len(wins)))
    for name, v in (("rel_err_K", errs_k), ("rel_err_V", errs_v)):
        m, se = mean_se(v)
        rows.append(dict(arm="transfer", metric=name, value=m, se=se, n=len(v)))
    times = dict(receiver_prefill_s=mean_se(t_recv)[0], sender_prefill_s=mean_se(t_send)[0],
                 transfer_and_inject_s=mean_se(t_map)[0])
    print(f"  {len(wins)} windows: context {C}, continuation {args.cont} ({args.cont - 1} scored)")
    for x in rows:
        print(f"  {x['arm']:<9} {x['metric']:<26} {x['value']:+.4f} ± {x['se']:.4f}")
    print("  times (s, mean per window): " + ", ".join(f"{k}={v:.3f}" for k, v in times.items()))
    del src, tgt, pipe
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return dict(sender=src_name, receiver=tgt_name, fit=fit_info, rows=rows, wall_times=times)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="HuggingFaceTB/SmolLM2-360M:HuggingFaceTB/SmolLM2-1.7B,"
                                       "Qwen/Qwen2.5-0.5B:Qwen/Qwen2.5-1.5B")
    ap.add_argument("--context", type=int, default=512, help="context tokens C (calibration windows have C tokens)")
    ap.add_argument("--cont", type=int, default=64, help="continuation tokens D")
    ap.add_argument("--calib-windows", type=int, default=64)
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8, help="source layers per target layer (library default 8)")
    ap.add_argument("--ridge-lambda", type=float, default=0.01, help="library default 0.01")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    device = pick_device(a.device)
    torch.manual_seed(0)
    results = [run_pair(*p.split(":"), a, device) for p in a.pairs.split(",") if p]
    out = a.out or f"results_xkv_transfer_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env_info(device), args=vars(a), dtype="float32",
                       text="WikiText-2 raw: calibration on train, evaluation on test", pairs=results), f, indent=2)
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
