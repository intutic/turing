#!/usr/bin/env python3
"""
Quality of Turing Engine serving settings on real weights: next-token NLL against the exact model, paired per window.

Method (after cift F6): consecutive windows of T tokens from WikiText-2 test. The first T - D tokens are the prompt;
the last D positions are scored. Arms in "forward" mode score them in one teacher-forced pass; arms in "decode" mode
prefill the prompt, then feed one token per step through the KV cache, exactly as serving does; arms in "chunk" mode
prefill the prompt, then score the last D tokens in one pass over the cache. For a low-bit cache "chunk" equals
"decode" (each query reads every key after quantization; tests/test_kv_cache.py checks this), and it is far faster.
Quest page selection acts only on one-token steps, so its arms run in "decode" mode. Each arm's NLL is
compared window by window against its reference arm (mean delta and its standard error over windows).

Sanity rows: `exact_forward` vs HuggingFace's own model (`hf_native`), and `exact_decode` vs `exact_forward`.

python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-135M --arms exact_forward,served_default
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import gc
import json
import math
import platform
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.models.hf_loader import RealHuggingFaceLoader

# name -> (mode, loader settings, KV settings set on model.config, reference arm)
ARMS = {
    "hf_native": ("forward", None, {}, "exact_forward"),
    "exact_forward": ("forward", dict(sparsity_ratio=0.0, enable_recirculation=False), {}, None),
    "sparsity_50": ("forward", dict(sparsity_ratio=0.5, enable_recirculation=False), {}, "exact_forward"),
    "recirculation": ("forward", dict(sparsity_ratio=0.0, enable_recirculation=True), {}, "exact_forward"),
    "served_default": ("forward", dict(sparsity_ratio=0.5, enable_recirculation=True), {}, "exact_forward"),
    "exact_decode": ("decode", dict(sparsity_ratio=0.0, enable_recirculation=False), {}, "exact_forward"),
}
_EXACT = dict(sparsity_ratio=0.0, enable_recirculation=False)
ARMS |= {  # KV cache settings through the cache; low-bit arms vs exact_forward, Quest arms vs exact_decode
    "int8": ("chunk", _EXACT, dict(kv_cache_dtype="int8"), "exact_forward"),
    "int4": ("chunk", _EXACT, dict(kv_cache_dtype="int4"), "exact_forward"),
    "int8_nosmooth": ("chunk", _EXACT, dict(kv_cache_dtype="int8", kv_smoothing=False), "exact_forward"),
    "int4_nosmooth": ("chunk", _EXACT, dict(kv_cache_dtype="int4", kv_smoothing=False), "exact_forward"),
    "quest25": ("decode", _EXACT, dict(kv_page_budget=0.25), "exact_decode"),
    "quest10": ("decode", _EXACT, dict(kv_page_budget=0.10), "exact_decode"),
    "int8_quest25": ("decode", _EXACT, dict(kv_cache_dtype="int8", kv_page_budget=0.25), "exact_decode"),
    # 4-bit weights (turing/models/quant.py): round to nearest, and GPTQ calibrated on 128 x 2048 WikiText-2 train tokens
    "rtn4": ("forward", dict(_EXACT, quant="rtn"), {}, "exact_forward"),
    "gptq4": ("forward", dict(_EXACT, quant="gptq"), {}, "exact_forward"),
    # TEAL-style activation sparsity (turing/models/activation_sparsity.py), thresholds from WikiText-2 train
    "teal25": ("forward", dict(_EXACT, teal=0.25), {}, "exact_forward"),
    "teal40": ("forward", dict(_EXACT, teal=0.40), {}, "exact_forward"),
    "teal50": ("forward", dict(_EXACT, teal=0.50), {}, "exact_forward"),
}


def windows(tokenizer, T, n):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = min(n, len(ids) // T)
    return ids[: n * T].view(n, T)


def nll(logits, targets):
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1)).item()


@torch.inference_mode()
def score(model, win, s, mode, hf=False):
    """Mean NLL of targets s+1 .. T-1 for one window [1, T]."""
    if mode == "forward":
        logits = model(win).logits if hf else model(win)[0]
        return nll(logits[:, s:-1], win[:, s + 1:])
    _, past = model(win[:, :s], logits_to_keep=1)
    if mode == "chunk":
        logits, _ = model(win[:, s:-1], past_key_values=past)
        return nll(logits, win[:, s + 1:])
    out = []
    for t in range(s, win.shape[1] - 1):
        logits, past = model(win[:, t:t + 1], past_key_values=past)
        out.append(logits[:, -1])
    return nll(torch.stack(out, 1), win[:, s + 1:])


def mean_se(v):
    v = torch.tensor(v, dtype=torch.float64)
    return v.mean().item(), (v.std().item() / math.sqrt(len(v)) if len(v) > 1 else None)


def run_model(name, arms, T, D, n_win, device, dtype_name=None):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype_name] if dtype_name else (
        torch.float16 if device.type == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    # eager attention on MPS: its SDPA reuses one cached graph per shape with the first call's scale (PyTorch 2.9)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **({"attn_implementation": "eager"}
                                                                     if device.type == "mps" else {})).to(device).eval()
    wins = windows(tok, T, n_win).to(device)
    s = T - D
    per_arm, rows = {}, []
    for arm in arms:
        mode, load, kv, _ = ARMS[arm]
        load = dict(load) if load is not None else None
        quant = load.pop("quant", None) if load else None
        teal = load.pop("teal", None) if load else None
        model = hf if load is None else RealHuggingFaceLoader.convert_hf_model(hf, device=str(device), **load)
        if quant:  # replaces modules only: the HF weights this model shares stay as they are for the other arms
            from turing.models.quant import calibration_windows, quantize_model
            model = quantize_model(model, quant, calibration_windows(tok) if quant == "gptq" else None)
        if teal:
            from turing.models.activation_sparsity import calibrate_and_apply
            from turing.models.quant import calibration_windows
            calibrate_and_apply(model, calibration_windows(tok, 16, 512), teal)
        for k, v in kv.items():
            setattr(model.config, k, v)
        t0 = time.perf_counter()
        per_arm[arm] = [score(model, wins[i:i + 1], s, mode, hf=load is None) for i in range(len(wins))]
        secs = time.perf_counter() - t0
        m, se = mean_se(per_arm[arm])
        rows.append(dict(model=name, arm=arm, mode=mode, metric="nll", value=m, se=se, n=len(wins), seconds=secs, dtype=str(dtype)[6:]))
        if model is not hf:
            del model
            gc.collect()  # free its KV pool before the next arm sizes one
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for arm in arms:
        ref = ARMS[arm][3]
        if ref in per_arm:
            d, se = mean_se([a - b for a, b in zip(per_arm[arm], per_arm[ref])])
            rows.append(dict(model=name, arm=arm, metric="delta_nll_vs_" + ref, value=d, se=se, n=len(wins), dtype=str(dtype)[6:]))
    print(f"\n{name}  ({len(wins)} windows of {T}, last {D} positions scored, {device}, {str(dtype)[6:]})")
    for r in rows:
        se = f" ± {r['se']:.5f}" if r.get("se") is not None else ""
        print(f"  {r['arm']:<16} {r['metric']:<28} {r['value']:+.5f}{se}")
    del hf
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--arms", default="hf_native,exact_forward,sparsity_50,recirculation,served_default")
    ap.add_argument("--context", type=int, default=2048, help="window length T")
    ap.add_argument("--decode", type=int, default=256, help="positions scored at the end of each window")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default=None, choices=["fp16", "bf16", "fp32"], help="default: fp16 on CUDA, fp32 elsewhere")
    ap.add_argument("--out", default=None, help="results JSON (default results_kv_quality_<device>.json)")
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    unknown = [x for x in arms if x not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(ARMS)}")
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(a.device)
    rows = []
    for name in [m for m in a.models.split(",") if m]:
        rows += run_model(name, arms, a.context, a.decode, a.windows, device, a.dtype)
    env = dict(device=a.device, torch=torch.__version__, platform=platform.platform(),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor())
    out = a.out or f"results_kv_quality_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env, args=vars(a), text="WikiText-2 raw test, consecutive windows", rows=rows), f, indent=2)
    print(f"\n[+] wrote {out}")


if __name__ == "__main__":
    main()
