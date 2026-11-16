#!/usr/bin/env python3
"""
End-to-end batch generation, Turing Engine against vLLM on the same GPU with identical requests: B prompts of C tokens
(WikiText-2 windows), each generating exactly N tokens (EOS ignored), all submitted at once; reported: time to finish
the batch, output tokens per second, and time to first token. Turing runs through ContinuousBatchEngine (paged pool,
CUDA graphs, prefix caching off so no request reuses another's pages); vLLM through vllm.LLM.generate (run it in its
own virtualenv: --engine vllm). Greedy decoding on both.

python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-360M --batches 1,8,32,64
python scripts/benchmark_e2e_generation.py --engine vllm   --model HuggingFaceTB/SmolLM2-360M --batches 1,8,32,64
"""
import argparse
import asyncio
import json
import os
import platform
import sys
import time


def prompts(tokenizer, n, C, seed=0):
    from datasets import load_dataset
    import random
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text).input_ids
    rng = random.Random(seed)
    return [ids[s:s + C] for s in (rng.randrange(0, len(ids) - C) for _ in range(n))]


def run_turing(a, batches):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import torch
    from transformers import AutoTokenizer
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device="cuda", dtype=a.dtype)
    model.config.kv_cache_dtype = a.kv_cache_dtype
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=max(batches)), model=model,
                                   tokenizer=tok, prefix_caching=False, draft_tokens=0, prefill_chunk_size=2048)
    rows = []

    async def batch(ps):
        t0 = time.perf_counter()
        reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0,
                                         stop_token_ids=set()) for p in ps]
        ttft = []

        async def drain(r):
            n = 0
            async for _ in engine.stream_request(r):
                if n == 0:
                    ttft.append(time.perf_counter() - t0)
                n += 1
            return n
        counts = await asyncio.gather(*[drain(r) for r in reqs])
        return time.perf_counter() - t0, sum(counts), ttft

    async def main():
        await engine.start()
        try:
            for B in batches:
                await batch(prompts(tok, B, a.context, seed=99))  # warm-up at this batch size: CUDA graph capture
            for B in batches:
                secs, n, ttft = await batch(prompts(tok, B, a.context))
                rows.append(dict(engine="turing", model=a.model, batch=B, context=a.context, new_tokens=a.new_tokens,
                                 seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1),
                                 mean_ttft_s=round(sum(ttft) / len(ttft), 4), kv_cache_dtype=a.kv_cache_dtype))
                print(json.dumps(rows[-1]), flush=True)
        finally:
            await engine.stop()
    asyncio.run(main())
    return rows, torch.__version__


def run_vllm(a, batches):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    kv = {"native": "auto", "int8": "fp8"}.get(a.kv_cache_dtype, "auto")  # vLLM's 8-bit KV is fp8
    llm = LLM(a.model, dtype=a.dtype or "float16", kv_cache_dtype=kv, enable_prefix_caching=False,
              disable_log_stats=False,  # request metrics (first-token time) are recorded only with stats on
              max_model_len=a.context + a.new_tokens + 16, gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens, ignore_eos=True)
    for B in batches:  # warm-up at every batch size, as for Turing
        llm.generate([{"prompt_token_ids": p} for p in prompts(tok, B, a.context, seed=99)], sp, use_tqdm=False)
    rows = []
    for B in batches:
        ps = prompts(tok, B, a.context)
        t0 = time.perf_counter()
        outs = llm.generate([{"prompt_token_ids": p} for p in ps], sp, use_tqdm=False)
        secs = time.perf_counter() - t0
        n = sum(len(o.outputs[0].token_ids) for o in outs)
        ttft = [o.metrics.first_token_time - o.metrics.arrival_time for o in outs
                if getattr(o, "metrics", None) and getattr(o.metrics, "first_token_time", None)]
        rows.append(dict(engine="vllm", model=a.model, batch=B, context=a.context, new_tokens=a.new_tokens,
                         seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1),
                         mean_ttft_s=round(sum(ttft) / len(ttft), 4) if ttft else None, kv_cache_dtype=kv))
        print(json.dumps(rows[-1]), flush=True)
    import vllm
    return rows, f"vllm {vllm.__version__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["turing", "vllm"], required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--kv-cache-dtype", default="native", choices=["native", "int8"])
    ap.add_argument("--out", default="results_e2e_generation_cuda.json")
    a = ap.parse_args()
    if a.engine == "vllm":
        a.dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(a.dtype, a.dtype)
    batches = [int(b) for b in a.batches.split(",")]
    rows, version = (run_vllm if a.engine == "vllm" else run_turing)(a, batches)
    old = json.load(open(a.out))["rows"] if os.path.exists(a.out) else []
    keep = [r for r in old if not (r["engine"] == a.engine and r["model"] == a.model and r["context"] == a.context
                                   and r.get("kv_cache_dtype") in (a.kv_cache_dtype, {"native": "auto"}.get(a.kv_cache_dtype)))]
    env = dict(platform=platform.platform(), runtime=version)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=keep + rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
