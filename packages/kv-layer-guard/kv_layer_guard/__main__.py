# Copyright 2026 Ishan Gupta
# SPDX-License-Identifier: Apache-2.0
"""python -m kv_layer_guard MODEL [--kv-cache-dtype fp8]: prints the layers to keep in full precision, as JSON."""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM

from .guard import VLLM_FORMATS, choose_layers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("--kv-cache-dtype", default="fp8", choices=sorted(VLLM_FORMATS))
    ap.add_argument("--threshold", type=float, default=0.02, help="KL in nats above which a layer is kept")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(a.device).eval()
    layers, kls = choose_layers(model, VLLM_FORMATS[a.kv_cache_dtype], threshold=a.threshold)
    print(json.dumps({"model": a.model, "kv_cache_dtype": a.kv_cache_dtype, "kv_cache_dtype_skip_layers":
                      [str(i) for i in layers], "layer_kl": [round(k, 6) for k in kls]}))


if __name__ == "__main__":
    main()
