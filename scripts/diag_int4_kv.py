#!/usr/bin/env python3
"""
Where int4 KV loses information: per layer, the relative int4 reconstruction error of K (after RoPE, with and without
key smoothing) and of V on real text, and how far real-text channel maxima of K exceed the random-token calibration.

python scripts/diag_int4_kv.py --models Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B --tokens 1024 --device cpu
"""
import _bootstrap  # noqa: F401,E402
import argparse
import json

import torch

from turing.core.kv_cache import INT4_GROUP, _dequantize, _quantize
from turing.models.hf_loader import RealHuggingFaceLoader


def rel_err(x, bits=4, group=INT4_GROUP):
    c, s, z = _quantize(x, bits, group)
    return ((_dequantize(c, s, z, group, torch.float32) - x.float()).norm() / x.float().norm()).item()


@torch.no_grad()
def run(name, T, device):
    from datasets import load_dataset
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(name, device=device, dtype="fp32")
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:400])
    ids = tok(text, return_tensors="pt").input_ids[:, :T].to(device)
    model.calibrate_kv_smoothing()
    attns = [layer.self_attn for layer in model.layers]
    got = {}
    for i, a in enumerate(attns):
        orig = a._attend

        def cap(q, k, v, mask, causal, _i=i, _orig=orig):
            got[_i] = (k.detach(), v.detach())
            return _orig(q, k, v, mask, causal=causal)
        a._attend = cap
    model._body(ids, 0, None, None)
    rows = []
    for i, a in enumerate(attns):
        k_s, v = got[i]  # [B, Hkv, T, D], keys already divided by k_smooth
        s = a.k_smooth.float()
        k = k_s.float() * s
        real_max = k.abs().amax(dim=(0, 2))  # [Hkv, D]
        med = real_max.median(dim=-1, keepdim=True).values
        rows.append(dict(layer=i, k_err_raw=rel_err(k), k_err_smoothed=rel_err(k_s), v_err=rel_err(v),
                         k_absmax=real_max.max().item(), k_median_channel_max=med.median().item(),
                         k_smoothed_absmax=k_s.abs().max().item(), v_absmax=v.abs().max().item(),
                         v_median_channel_max=v.abs().amax(dim=(0, 2)).median().item(),
                         smooth_max=s.max().item(),
                         # real-text max / (calibration scale * median): >1 means calibration under-shrank a channel
                         worst_undershrink=(real_max / (s[0, :, 0, :] * med)).max().item()))
    return dict(model=name, head_dim=attns[0].head_dim, tokens=int(ids.shape[1]), rows=rows)


def fq(x):
    c, s, z = _quantize(x, 4, INT4_GROUP)
    return _dequantize(c, s, z, INT4_GROUP, x.dtype)


@torch.no_grad()
def ablate(name, T, device, windows):
    """NLL of the last quarter of each window with int4 round-trips applied to chosen layers' K and/or V inside a
    stateless forward (keys after RoPE and smoothing, as the pool stores them)."""
    from datasets import load_dataset
    import torch.nn.functional as F
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(name, device=device, dtype="fp32")
    model.calibrate_kv_smoothing()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0][: windows * T].view(windows, T).to(device)
    attns = [layer.self_attn for layer in model.layers]
    cfg = {}
    for i, a in enumerate(attns):
        orig = a._attend

        def hook(q, k, v, mask, causal, _i=i, _orig=orig):
            c = cfg.get(_i)
            if c:
                k2, v2 = (fq(k) if "k" in c else k), (fq(v) if "v" in c else v)
                if "keep0" in c:  # first token exact
                    k2, v2 = torch.cat([k[:, :, :1], k2[:, :, 1:]], 2), torch.cat([v[:, :, :1], v2[:, :, 1:]], 2)
                k, v = k2, v2
            return _orig(q, k, v, mask, causal=causal)
        a._attend = hook
    s = T * 3 // 4

    def nll():
        out = []
        for w in ids:
            logits = model._body(w[None], 0, None, None).float()
            out.append(F.cross_entropy(logits[0, s - 1:-1], w[s:]).item())
        return sum(out) / len(out)
    L = len(attns)
    configs = [("none", {}), ("all_kv", {i: "kv" for i in range(L)}), ("all_k", {i: "k" for i in range(L)}),
               ("all_v", {i: "v" for i in range(L)}), ("all_kv_keep0", {i: "kv keep0" for i in range(L)})]
    configs += [(f"layer{i}_kv", {i: "kv"}) for i in range(L)]
    res = {}
    for nm, c in configs:
        cfg.clear(); cfg.update(c)
        res[nm] = nll()
        print(f"{nm}: {res[nm]:.4f} ({res[nm] - res.get('none', res[nm]):+.4f})", flush=True)
    return dict(model=name, ablation=res)


@torch.no_grad()
def sensitivity(name, T, device):
    """Per layer: relative error of the attention output when K and V take an int4 / int8 round-trip, on the
    random-token calibration batch (4 x 256, what calibrate_kv_smoothing sees) and on 1 x T tokens of real text."""
    from datasets import load_dataset
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(name, device=device, dtype="fp32")
    model.calibrate_kv_smoothing()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    real = tok(text, return_tensors="pt").input_ids[:, :T].to(device)
    rand = torch.randint(0, model.config.vocab_size, (4, 256), generator=torch.Generator().manual_seed(0)).to(device)
    attns = [layer.self_attn for layer in model.layers]
    errs = {}
    for i, a in enumerate(attns):
        orig = a._attend

        def hook(q, k, v, mask, causal, _i=i, _orig=orig):
            out = _orig(q, k, v, mask, causal=causal)
            for bits, group in ((4, INT4_GROUP), (8, k.shape[-1])):
                def rt(x):
                    c, s, z = _quantize(x, bits, group)
                    return _dequantize(c, s, z, group, x.dtype)
                o2 = _orig(q, rt(k), rt(v), mask, causal=causal)
                errs[(cur["src"], _i, bits)] = ((o2 - out).norm() / out.norm()).item()
            return out
        a._attend = hook
    cur = {}
    for src, ids in (("random", rand), ("real", real)):
        cur["src"] = src
        model._body(ids, 0, None, None)
    rows = [dict(layer=i, **{f"{src}_int{b}": errs[(src, i, b)] for src in ("random", "real") for b in (4, 8)})
            for i in range(len(attns))]
    return dict(model=name, rows=rows)


@torch.no_grad()
def kl_select(name, T, device):
    """Per layer: mean KL(exact || one layer's K and V int4 / int8 round-tripped) of the next-token distribution, on
    random tokens (4 x 256) and on 1 x T tokens of real text: can calibration on random tokens find the fragile
    layers the NLL ablation finds?"""
    from datasets import load_dataset
    import torch.nn.functional as F
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(name, device=device, dtype="fp32")
    model.calibrate_kv_smoothing()
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    real = tok(text, return_tensors="pt").input_ids[:, :T].to(device)
    rand = torch.randint(0, model.config.vocab_size, (4, 256), generator=torch.Generator().manual_seed(0)).to(device)
    attns = [layer.self_attn for layer in model.layers]
    cfg = {}
    for i, a in enumerate(attns):
        orig = a._attend

        def hook(q, k, v, mask, causal, _i=i, _orig=orig):
            if cfg.get("layer") == _i:
                bits, group = cfg["bits"], (INT4_GROUP if cfg["bits"] == 4 else k.shape[-1])
                rt = lambda x: _dequantize(*_quantize(x, bits, group), group, x.dtype)  # noqa: E731
                k, v = rt(k), rt(v)
            return _orig(q, k, v, mask, causal=causal)
        a._attend = hook
    rows = []
    for src, ids in (("random", rand), ("real", real)):
        cfg.clear()
        ref = F.log_softmax(model._body(ids, 0, None, None).float(), -1)
        for bits in (4, 8):
            for i in range(len(attns)):
                cfg.update(layer=i, bits=bits)
                lp = F.log_softmax(model._body(ids, 0, None, None).float(), -1)
                kl = (ref.exp() * (ref - lp)).sum(-1).mean().item()
                rows.append(dict(src=src, bits=bits, layer=i, kl=kl))
        print(src, "done", flush=True)
    return dict(model=name, rows=rows)


@torch.no_grad()
def smooth_kl(name, T, device):
    """KL(exact || every layer's K and V int4 / int8 round-tripped at once), with and without key smoothing, on the
    calibration random tokens and on real text: does the random-token batch predict when smoothing helps?"""
    from datasets import load_dataset
    import torch.nn.functional as F
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(name, device=device, dtype="fp32")
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    real = tok(text, return_tensors="pt").input_ids[:, :T].to(device)
    rand = torch.randint(0, model.config.vocab_size, (4, 256), generator=torch.Generator().manual_seed(0)).to(device)
    attns = [layer.self_attn for layer in model.layers]
    model.calibrate_kv_smoothing()
    smooth = [(a.k_smooth, a.q_smooth) for a in attns]
    out = {}
    for smoothed in (False, True):
        for a, (ks, qs) in zip(attns, smooth):
            a.k_smooth, a.q_smooth = (ks, qs) if smoothed else (None, None)
        for src, ids in (("random", rand), ("real", real)):
            for a in attns:
                a._kv_roundtrip = None
            ref = F.log_softmax(model._body(ids, 0, None, None).float(), -1)
            for bits in (4, 8):
                for a in attns:
                    a._kv_roundtrip = (bits, INT4_GROUP if bits == 4 else a.head_dim)
                lp = F.log_softmax(model._body(ids, 0, None, None).float(), -1)
                out[f"{src}_int{bits}_{'smooth' if smoothed else 'raw'}"] = (ref.exp() * (ref - lp)).sum(-1).mean().item()
            for a in attns:
                a._kv_roundtrip = None
    print(name, json.dumps({k: round(v, 5) for k, v in out.items()}), flush=True)
    return dict(model=name, kl=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="diag_int4_kv.json")
    ap.add_argument("--ablate", action="store_true", help="per-layer / K-only / V-only / keep-token-0 NLL ablation")
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--sensitivity", action="store_true", help="per-layer attention-output error, random vs real text")
    ap.add_argument("--kl", action="store_true", help="per-layer KL of the next-token distribution, random vs real")
    ap.add_argument("--smooth-kl", action="store_true", help="all-layer KL with and without smoothing")
    a = ap.parse_args()
    if a.smooth_kl:
        json.dump([smooth_kl(m, a.tokens, a.device) for m in a.models.split(",")], open(a.out, "w"), indent=1)
        return
    if a.kl:
        res = [kl_select(m, a.tokens, a.device) for m in a.models.split(",")]
        json.dump(res, open(a.out, "w"), indent=1)
        for r in res:
            print("==", r["model"])
            for src in ("random", "real"):
                for bits in (4, 8):
                    xs = sorted(((x["kl"], x["layer"]) for x in r["rows"] if x["src"] == src and x["bits"] == bits),
                                reverse=True)
                    print(f"  {src} int{bits} top: " + ", ".join(f"L{l}={k:.4f}" for k, l in xs[:4])
                          + f"  median={xs[len(xs) // 2][0]:.5f}")
        return
    if a.sensitivity:
        res = [sensitivity(m, a.tokens, a.device) for m in a.models.split(",")]
        json.dump(res, open(a.out, "w"), indent=1)
        for r in res:
            print("==", r["model"])
            for x in r["rows"]:
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in x.items()))
        return
    if a.ablate:
        res = [ablate(m, a.tokens, a.device, a.windows) for m in a.models.split(",")]
        json.dump(res, open(a.out, "w"), indent=1)
        return
    res = [run(m, a.tokens, a.device) for m in a.models.split(",")]
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(f"== {r['model']} (head_dim {r['head_dim']})")
        for x in r["rows"]:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in x.items()))


if __name__ == "__main__":
    main()
