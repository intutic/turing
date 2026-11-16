#!/usr/bin/env python3
"""
Batch-1 decode speed and weight memory of each weight format on real weights: fp16, 4-bit RTN/GPTQ (QuantLinear and
its fused int4 kernel), and TEAL-style activation sparsity on fp16 and on 4-bit weights. One WikiText-2 prompt of
--context tokens is prefilled, then --steps greedy decode steps are timed (device synchronized). Writes
results_weight_formats_<device>.json.

python scripts/benchmark_weight_formats.py --model HuggingFaceTB/SmolLM2-1.7B --settings fp16,rtn4,teal40,rtn4_teal40
"""
import _bootstrap  # noqa: F401,E402
import argparse
import copy
import json
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.models.activation_sparsity import calibrate_and_apply
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.models.quant import calibration_windows, quantize_model, weight_bytes


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def decode_rate(model, ids, steps, dev):
    model.config.cuda_graphs = False  # same path for every format (QuantLinear and sparse kernels take eager decode)
    for _ in range(2):  # warm-up: kernel compilation and autotuning
        _, past = model(ids, logits_to_keep=1)
        tok = torch.tensor([[1]], device=dev)
        for _ in range(4):
            logits, past = model(tok, past_key_values=past)
    _, past = model(ids, logits_to_keep=1)
    tok = torch.tensor([[1]], device=dev)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, past = model(tok, past_key_values=past)
        tok = logits[:, -1:].argmax(-1)
    sync(dev)
    return steps / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--settings", default="fp16,rtn4,gptq4,teal25,teal40,teal50,rtn4_teal40")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).eval()
    base = RealHuggingFaceLoader.convert_hf_model(hf, device=a.device, dtype=dtype)
    del hf
    ids = calibration_windows(tok, 1, a.context, seed=7).to(dev)
    rows = []
    for name in [s for s in a.settings.split(",") if s]:
        model = copy.deepcopy(base)
        quant = "gptq" if "gptq4" in name else "rtn" if "rtn4" in name else None
        if quant:
            quantize_model(model, quant, calibration_windows(tok, 32, 2048) if quant == "gptq" else None)
        if "teal" in name:
            calibrate_and_apply(model, calibration_windows(tok, 16, 512), int(name.split("teal")[1]) / 100)
        rate = decode_rate(model, ids, a.steps, dev)
        row = dict(model=a.model, setting=name, tokens_per_second=round(rate, 2), weight_bytes=weight_bytes(model),
                   context=a.context, steps=a.steps)
        print(json.dumps(row), flush=True)
        rows.append(row)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    env = dict(device=a.device, torch=torch.__version__, platform=platform.platform(),
               gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else platform.processor())
    out = a.out or f"results_weight_formats_{dev.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env, rows=rows), f, indent=2)
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
