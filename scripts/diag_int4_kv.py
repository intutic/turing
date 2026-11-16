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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="diag_int4_kv.json")
    a = ap.parse_args()
    res = [run(m, a.tokens, a.device) for m in a.models.split(",")]
    json.dump(res, open(a.out, "w"), indent=1)
    for r in res:
        print(f"== {r['model']} (head_dim {r['head_dim']})")
        for x in r["rows"]:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in x.items()))


if __name__ == "__main__":
    main()
