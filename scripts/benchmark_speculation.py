#!/usr/bin/env python3
"""
Prompt-lookup speculative decoding, measured: output tokens per second with and without drafting on three workloads
that differ in how much of the output repeats the prompt, for Turing (per-request draft tokens, with the speculation
gate `turing serve` installs) and for vLLM's n-gram drafter, on the same chat prompts with greedy decoding and prefix
caching off. With --draft-head both engines draft with that EAGLE-3 head instead:

  code     add type hints to a Python function from this repository (the output repeats most of the prompt)
  rewrite  correct a WikiText-2 paragraph and return it in full (the same)
  chat     solve a GSM8K problem step by step (little repetition)

Turing rows also give drafted and accepted tokens, tokens emitted per forward pass, and how many requests produced
exactly the tokens of the run without drafting.

python scripts/benchmark_speculation.py --engine turing --draft-tokens 0,4,8
python scripts/benchmark_speculation.py --engine vllm --draft-tokens 0    # in a vLLM environment, one run per value
python scripts/benchmark_speculation.py --engine vllm --draft-tokens 4
"""
import argparse
import ast
import asyncio
import json
import os
import platform
import sys
import textwrap
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def code_prompts(n):
    """Functions of 12 to 45 lines from the turing package, evenly spaced over the sorted source files."""
    found = []
    for dirpath, _, files in sorted(os.walk(os.path.join(ROOT, "turing"))):
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            src = open(os.path.join(dirpath, name), encoding="utf-8").read()
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, ast.FunctionDef) and 12 <= node.end_lineno - node.lineno + 1 <= 45:
                    found.append(textwrap.dedent("\n".join(src.splitlines()[node.lineno - 1:node.end_lineno])))
    step = max(1, len(found) // n)
    return ["Add type hints to every parameter that lacks one. Change nothing else and reply with the complete "
            f"function only.\n\n```python\n{f}\n```" for f in found[::step][:n]]


def rewrite_prompts(n):
    from datasets import load_dataset
    paragraphs = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
                  if 400 <= len(t.strip()) <= 900 and not t.strip().startswith("=")]
    return ["Correct any spelling or punctuation mistakes in this passage. Reply with the full corrected passage "
            f"only.\n\n{p}" for p in paragraphs[:n]]


def chat_prompts(n):
    from datasets import load_dataset
    return [f"{q}\nSolve it step by step." for q in load_dataset("openai/gsm8k", "main", split="test")["question"][:n]]


WORKLOADS = {"code": code_prompts, "rewrite": rewrite_prompts, "chat": chat_prompts}


def tokenized(tok, contents):
    # enable_thinking: Qwen3's template opens a thinking block unless told not to; other templates ignore it
    return [tok(tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False), add_special_tokens=False).input_ids for c in contents]


def run_turing(a, work):
    sys.path.insert(0, ROOT)
    import torch
    from transformers import AutoTokenizer
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    from turing.serving.spec_gate import SpeculationGatePolicy
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=a.device, dtype=a.dtype)
    model.config.kv_pool_fraction = None  # as turing serve: all free memory but a reserve
    head = None
    if a.draft_head:
        from turing.models.eagle3 import load_eagle3
        head = load_eagle3(a.draft_head, model.config.hidden_dim, model.config.num_layers)
    engine = ContinuousBatchEngine(model.config, TuringConfig(device=a.device, max_batch_size=max(a.concurrency)),
                                   model=model, tokenizer=tok, prefix_caching=False, draft_tokens=0,
                                   spec_gate=SpeculationGatePolicy(), prefill_chunk_size=2048, draft_head=head)
    engine.ride_prefill = not a.no_ride_prefill
    rows = []

    async def cell(prompts, k, c):
        async def one(p):
            r = await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0, draft_tokens=k)
            return [t async for t in engine.stream_request(r)]
        d0, a0 = engine.spec_drafted, engine.spec_accepted
        t0 = time.perf_counter()
        outs = []
        for i in range(0, len(prompts), c):  # c requests at a time, the next c when all of them are done: as for vLLM
            outs += await asyncio.gather(*[one(p) for p in prompts[i:i + c]])
        return time.perf_counter() - t0, outs, engine.spec_drafted - d0, engine.spec_accepted - a0

    async def main():
        await engine.start()
        try:
            for name, contents in work.items():
                prompts = tokenized(tok, contents)
                for k in a.draft_tokens:  # CUDA graphs and the verify forward's shapes, before anything is timed
                    await cell(prompts[:4], k, max(a.concurrency))
                for c in a.concurrency:
                    plain = None
                    for k in a.draft_tokens:
                        secs, outs, drafted, accepted = await cell(prompts, k, c)
                        n = sum(len(o) for o in outs)
                        plain = outs if k == 0 else plain
                        rows.append(dict(
                            engine="turing", model=a.model, dtype=a.dtype, drafter=a.draft_head or "prompt lookup",
                            workload=name, requests=len(prompts),
                            concurrency=c, draft_tokens=k, max_new_tokens=a.new_tokens, output_tokens=n,
                            seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1), drafted_tokens=drafted,
                            accepted_tokens=accepted, acceptance_rate=round(accepted / drafted, 4) if drafted else None,
                            tokens_per_pass=round(n / (n - accepted), 3),
                            same_tokens_as_no_draft=sum(o == p for o, p in zip(outs, plain)) if k and plain else None))
                        print(json.dumps(rows[-1]), flush=True)
        finally:
            await engine.stop()
    asyncio.run(main())
    return rows, torch.__version__


def run_vllm(a, work):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    assert len(a.draft_tokens) == 1, "vLLM: one --draft-tokens value per run (the drafter is fixed at load)"
    k = a.draft_tokens[0]
    tok = AutoTokenizer.from_pretrained(a.model)
    spec = {}
    if k and a.draft_head:
        spec = dict(speculative_config={"method": "eagle3", "model": a.draft_head, "num_speculative_tokens": k})
    elif k:
        spec = dict(speculative_config={"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": 3,
                                        "prompt_lookup_min": 1})
    llm = LLM(a.model, dtype={"fp16": "float16", "bf16": "bfloat16"}[a.dtype], enable_prefix_caching=False,
              max_model_len=4096, gpu_memory_utilization=a.gpu_memory_utilization, **spec)
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens)
    rows = []
    for name, contents in work.items():
        prompts = [{"prompt_token_ids": p} for p in tokenized(tok, contents)]
        llm.generate(prompts[:4], sp, use_tqdm=False)
        for c in a.concurrency:
            t0 = time.perf_counter()
            n = 0
            for i in range(0, len(prompts), c):  # c requests at a time, the next c when all of them are done
                n += sum(len(o.outputs[0].token_ids) for o in llm.generate(prompts[i:i + c], sp, use_tqdm=False))
            secs = time.perf_counter() - t0
            rows.append(dict(engine="vllm", model=a.model, dtype=a.dtype,
                             drafter=a.draft_head or "n-gram", workload=name, requests=len(prompts),
                             concurrency=c, draft_tokens=k, max_new_tokens=a.new_tokens, output_tokens=n,
                             seconds=round(secs, 3), output_tok_per_s=round(n / secs, 1)))
            print(json.dumps(rows[-1]), flush=True)
    import vllm
    return rows, f"vllm {vllm.__version__}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["turing", "vllm"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda", help="Turing only")
    ap.add_argument("--workloads", default="code,rewrite,chat")
    ap.add_argument("--requests", type=int, default=60, help="requests per workload")
    ap.add_argument("--new-tokens", type=int, default=256, help="output limit per request (generation stops at EOS)")
    ap.add_argument("--draft-tokens", default="0,4,8", help="0 = no drafting")
    ap.add_argument("--draft-head", default=None,
                    help="an EAGLE-3 head for the model (repository or directory): it drafts instead of prompt lookup")
    ap.add_argument("--concurrency", default="1,2", help="requests in flight at a time")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--no-ride-prefill", action="store_true",
                    help="turing: decode steps of their own while prompts are prefilled (the engine before riders)")
    ap.add_argument("--out", default="benchmarks/results/results_speculation_cuda.json")
    a = ap.parse_args()
    a.draft_tokens = [int(v) for v in a.draft_tokens.split(",")]
    a.concurrency = [int(v) for v in a.concurrency.split(",")]
    work = {w: WORKLOADS[w](a.requests) for w in a.workloads.split(",")}
    rows, version = (run_turing if a.engine == "turing" else run_vllm)(a, work)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    key = lambda r: (r["engine"], r["model"], r.get("drafter"), r["workload"], r["concurrency"],  # noqa: E731
                     r["draft_tokens"])
    keep = [r for r in old.get("rows", []) if key(r) not in {key(n) for n in rows}]
    import torch
    env = dict(platform=platform.platform(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, versions=dict(old.get("versions", {}), **{a.engine: version}), rows=keep + rows), f,
                  indent=2)
    print(f"[+] wrote {a.out}", flush=True)
    if a.engine == "vllm":
        os._exit(0)  # vLLM's helper processes keep the interpreter alive


if __name__ == "__main__":
    main()
