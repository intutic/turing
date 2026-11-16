#!/usr/bin/env python3
"""
Decode throughput of the serving path by batch, context and KV cache setting, on real weights.

For each (cache setting, context C, batch B): one WikiText-2 prompt of C tokens is prefilled (in 2,048-token
chunks for Turing, as the engine does; HF in one pass), its cache is copied
B - 1 times into independent pages (as B separate requests would hold; no copy-on-write sharing), then N batched decode
steps run through SubspaceCausalLM.forward with a list of per-request caches, as the engine does. The KV pool takes
--pool-fraction of free GPU memory; a batch that does not fit in it, or in memory, is reported as not fitting. Reported: generated tokens per second
(B * N / time, device synchronized), the cache bytes held, and whether the batch fit in memory. Contexts beyond the
model's trained length measure speed only; the model's quality there is not claimed.

python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --contexts 2048,8192,32768 --batches 1,8,32,64
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import gc
import json
import platform
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from turing.core.kv_cache import KVPoolExhausted
from turing.models.hf_loader import RealHuggingFaceLoader

PREFILL_CHUNK = 2048  # Turing prefills in chunks (the engine default is 512); HF prefills the prompt in one pass
SETTINGS = {"native_eager": dict(kv_cache_dtype="native", kv_page_budget=1.0, cuda_graphs=False),
            "native": dict(kv_cache_dtype="native", kv_page_budget=1.0),
            "int8": dict(kv_cache_dtype="int8", kv_page_budget=1.0),
            "int4": dict(kv_cache_dtype="int4", kv_page_budget=1.0),
            "int8_quest25": dict(kv_cache_dtype="int8", kv_page_budget=0.25),
            "hf_transformers": None}  # the same weights through HuggingFace transformers with its own DynamicCache


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run(model, prompt, B, steps, device, hf=None):
    tok = prompt[:, -1:].expand(B, 1).contiguous()
    if hf is not None:
        out = hf(prompt, use_cache=True)
        cache = out.past_key_values
        cache.batch_repeat_interleave(B)
        step = lambda t: hf(t, past_key_values=cache, use_cache=True).logits
        nbytes = lambda: sum(x.numel() * x.element_size() for layer in cache.layers for x in (layer.keys, layer.values))
    else:
        base = None
        for s in range(0, prompt.shape[1], PREFILL_CHUNK):  # chunked, as the engine prefills
            _, base = model(prompt[:, s:s + PREFILL_CHUNK], past_key_values=base, logits_to_keep=1)
        caches = [base] + [base.clone(share=False) for _ in range(B - 1)]
        step = lambda t: model(t, past_key_values=caches)[0]
        nbytes = lambda: sum(c.nbytes() for c in caches)
    step(tok)  # warm-up step (kernel compilation, allocator)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        tok = step(tok)[:, -1].argmax(-1, keepdim=True)
    sync(device)
    dt = time.perf_counter() - t0
    return B * steps / dt, nbytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--settings", default="native,int8,int8_quest25")
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pool-fraction", type=float, default=0.85, help="share of free GPU memory for the KV pool")
    ap.add_argument("--quest-min-tokens", type=int, default=None, help="override kv_select_min_tokens (0 = always)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(a.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype).eval()
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device=a.device, enable_recirculation=False)
    hf = hf.to(device) if "hf_transformers" in a.settings else None
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    cfg = model.config
    cfg.kv_pool_fraction = a.pool_fraction
    if a.quest_min_tokens is not None:
        cfg.kv_select_min_tokens = a.quest_min_tokens
    rows = []
    print(f"{a.model} on {a.device} ({str(dtype)[6:]}), {cfg.num_layers} layers, {cfg.num_heads}/{cfg.num_kv_heads} "
          f"heads, head_dim {cfg.head_dim}, trained context {cfg.max_position_embeddings}")
    for name in [s for s in a.settings.split(",") if s]:
        cfg.cuda_graphs = True
        for k, v in (SETTINGS[name] or {}).items():
            setattr(cfg, k, v)
        model._kv_pool = None  # each setting sizes its own pool from the memory the previous one leaves free
        model._graphs.graphs.clear()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for C in [int(x) for x in a.contexts.split(",")]:
            prompt = ids[:, :C]
            for B in [int(x) for x in a.batches.split(",")]:
                row = dict(setting=name, context=C, batch=B, steps=a.steps,
                           beyond_trained_context=C > cfg.max_position_embeddings)
                try:
                    tps, nbytes = run(model, prompt, B, a.steps, device, hf if name == "hf_transformers" else None)
                    row |= dict(tokens_per_s=tps, kv_cache_bytes=nbytes, fit=True)
                    print(f"  {name:<13} C={C:<6} B={B:<3} {tps:9.1f} tok/s   KV {nbytes / 2**30:6.2f} GiB")
                except (torch.OutOfMemoryError, KVPoolExhausted) as e:
                    row |= dict(fit=False, limit="kv pool full" if isinstance(e, KVPoolExhausted) else "cuda out of memory")
                    print(f"  {name:<13} C={C:<6} B={B:<3} {row['limit']}")
                if name != "hf_transformers" and getattr(model, "_kv_pool", None) is not None:
                    row["kv_pool_tokens"] = model._kv_pool.num_pages * 16
                rows.append(row)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if not row["fit"]:
                    break  # larger batches will not fit either
        if name == "hf_transformers":  # free the baseline's weights before the Turing pool is sized
            hf = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
    env = dict(device=a.device, torch=torch.__version__, platform=platform.platform(),
               gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor())
    out = a.out or f"results_kv_decode_{device.type}.json"
    with open(out, "w") as f:
        json.dump(dict(env=env, args=vars(a), model_config=vars(cfg), rows=rows), f, indent=2)
    print(f"[+] wrote {out}")


if __name__ == "__main__":
    main()
