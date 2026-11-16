#!/usr/bin/env python3
"""
Real-checkpoint parity: does the Turing model reproduce HuggingFace's logits for each supported family?

For each model: HuggingFace's logits for one WikiText-2 test window (T tokens), then the converted Turing model's
logits from a full forward and from an incremental run (prefill half, then one token at a time through the paged KV
cache). Reports the largest absolute logit difference, top-1 agreement and both models' NLL on the window, and the
same comparison of HuggingFace's own cached decode against its full forward (the dtype's noise floor: in bf16 the two
HF paths already differ).

python scripts/eval_parity.py --models gpt2,facebook/opt-125m --device mps
"""
import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # this checkout's turing, not an installed one

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.models.hf_loader import RealHuggingFaceLoader, resolve_dtype


def window(tok, T):
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"][:400])
    return tok(text, return_tensors="pt").input_ids[:, :T]


def nll(logits, ids):
    return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:]).item()


@torch.inference_mode()
def run(name, device, dtype_name, T):
    dev = torch.device(device)
    dtype = resolve_dtype(dtype_name, dev) if dtype_name else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    ids = window(tok, T).to(dev)
    t0 = time.perf_counter()
    # eager attention for the reference: HF's SDPA path skips Gemma 2's attention soft-capping, and PyTorch 2.9's MPS
    # SDPA reuses one cached graph per shape with the first call's scale
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="eager").to(dev).eval()
    ref = hf(ids).logits.float()
    # HF against itself, prefill half then one token at a time: the noise floor of this dtype (bf16 reductions differ
    # between a full forward and cached decode)
    out = hf(ids[:, :T // 2], use_cache=True)
    hf_inc, past_hf = [out.logits], out.past_key_values
    for t in range(T // 2, T):
        out = hf(ids[:, t:t + 1], past_key_values=past_hf, use_cache=True)
        hf_inc.append(out.logits)
        past_hf = out.past_key_values
    hf_inc = torch.cat(hf_inc, 1).float()
    del past_hf, out
    params = sum(p.numel() for p in hf.parameters())
    model = RealHuggingFaceLoader.convert_hf_model(hf, device=device, dtype=dtype)
    arch = model.config.arch
    del hf
    gc.collect()
    full = model(ids)[0].float()
    half = T // 2
    logits, past = model(ids[:, :half])
    outs = [logits]
    for t in range(half, T):
        logits, past = model(ids[:, t:t + 1], past_key_values=past)
        outs.append(logits)
    inc = torch.cat(outs, 1).float()
    del past, model
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    row = dict(model=name, arch=arch, params=params, dtype=str(dtype)[6:], tokens=T,
               max_abs_diff_full=(full - ref).abs().max().item(), max_abs_diff_incremental=(inc - ref).abs().max().item(),
               max_rel_diff=((inc - ref).abs().max() / ref.abs().max()).item(),  # diff relative to the largest logit
               top1_agreement=(inc.argmax(-1) == ref.argmax(-1)).float().mean().item(),
               hf_self_max_abs_diff=(hf_inc - ref).abs().max().item(),
               hf_self_top1_agreement=(hf_inc.argmax(-1) == ref.argmax(-1)).float().mean().item(),
               nll_hf=nll(ref, ids), nll_turing=nll(inc, ids), seconds=round(time.perf_counter() - t0, 1))
    print(json.dumps(row), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True, help="comma-separated HF repo ids")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="fp32 (default), bf16 or fp16")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default=None, help="results JSON (default results_parity_<device>.json; rows are merged)")
    a = ap.parse_args()
    out = Path(a.out or f"results_parity_{torch.device(a.device).type}.json")
    old = json.loads(out.read_text())["rows"] if out.exists() else []
    rows = {(r["model"], r["dtype"]): r for r in old}
    for name in [m for m in a.models.split(",") if m]:
        try:
            r = run(name, a.device, a.dtype, a.tokens)
        except Exception as e:  # keep going; the failure is recorded, not hidden
            r = dict(model=name, dtype=a.dtype or "float32", error=f"{type(e).__name__}: {e}")
            print(json.dumps(r), flush=True)
        rows[(r["model"], r["dtype"])] = r
        env = dict(device=a.device, torch=torch.__version__, platform=platform.platform(),
                   gpu=torch.cuda.get_device_name(0) if torch.device(a.device).type == "cuda" else platform.processor())
        out.write_text(json.dumps(dict(env=env, text="WikiText-2 raw test, first window", rows=list(rows.values())),
                                  indent=2))
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
