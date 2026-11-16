#!/usr/bin/env python3
"""
Which projections make TEAL activation sparsity expensive on a model: per-window NLL with TEAL thresholds (calibrated
once, as turing does) switched on for all projections, attention inputs only, MLP gate/up inputs only, or MLP down
inputs only, against the dense model, in the dtype given.

python scripts/diag_teal.py --model HuggingFaceTB/SmolLM2-1.7B --sparsity 0.4 --dtype fp32 --device cpu
"""
import _bootstrap  # noqa: F401,E402
import argparse
import json

import torch
import torch.nn.functional as F

from turing.models.activation_sparsity import SparseInputLinear, calibrate_and_apply
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.models.quant import calibration_windows

GROUPS = {"all": lambda n: True,
          "attn": lambda n: ".self_attn." in n,
          "mlp_in": lambda n: n.endswith(("gate_proj", "up_proj", "gate_up_proj", "fc1")),
          "mlp_down": lambda n: n.endswith(("down_proj", "fc2"))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--sparsity", type=float, default=0.4)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--out", default="diag_teal.json")
    a = ap.parse_args()
    from datasets import load_dataset
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=a.device, dtype=a.dtype)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    wins = ids[: a.windows * a.tokens].view(a.windows, a.tokens).to(a.device)
    thresholds = calibrate_and_apply(model, calibration_windows(tok, 16, 512), a.sparsity)
    mods = {n: m for n, m in model.named_modules() if isinstance(m, SparseInputLinear)}

    def nll(w):
        logits = model(w[None])[0].float()
        return F.cross_entropy(logits[0, :-1], w[1:]).item()

    res = dict(model=a.model, sparsity=a.sparsity, dtype=a.dtype, thresholds=thresholds, arms={})
    for arm, keep in [("dense", lambda n: False)] + list(GROUPS.items()):
        for n, m in mods.items():
            m.threshold = thresholds[n] if keep(n) else 0.0
        res["arms"][arm] = [nll(w) for w in wins]
        print(arm, [round(x, 3) for x in res["arms"][arm]], flush=True)
    for n, m in mods.items():  # which layers' thresholds stand out
        m.threshold = thresholds[n]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
