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


CODE_MODULES = ("argparse", "ast", "bisect", "calendar", "cmd", "configparser", "csv", "difflib", "fractions",
                "heapq", "json.decoder", "json.encoder", "shlex", "statistics", "string", "textwrap")


def code_prompts(n):
    """Functions of 12 to 45 lines from Python's standard library (CODE_MODULES), evenly spaced: the same prompts
    for every engine run under one Python version. (They came from the turing package itself until 2026-11-21, so
    two runs on either side of a code change had different prompts: 5,244 against 4,993 output tokens without
    drafts in results_eagle3_cuda.json's code rows.)"""
    import importlib
    import inspect
    found = []
    for name in CODE_MODULES:
        src = inspect.getsource(importlib.import_module(name))
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
    from turing.models.causal_lm import PrefillGraphs
    head = None
    if a.draft_head:
        from turing.models.eagle3 import load_eagle3
        head = load_eagle3(a.draft_head, model.config.hidden_dim, model.config.num_layers)
    engine = ContinuousBatchEngine(model.config, TuringConfig(device=a.device, max_batch_size=max(a.concurrency)),
                                   model=model, tokenizer=tok, prefix_caching=False, draft_tokens=0,
                                   spec_gate=SpeculationGatePolicy(), prefill_chunk_size=2048, draft_head=head)
    engine.ride_prefill = not a.no_ride_prefill
    if a.head_reach is not None:  # the draft head's chain rule (ContinuousBatchEngine.HEAD_REACH); 0: always full
        engine.HEAD_REACH = a.head_reach or 1e-30
    rows = []

    async def cell(prompts, k, c):
        async def one(p):
            r = await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0, draft_tokens=k)
            return [t async for t in engine.stream_request(r)]
        d0, a0, v0 = engine.spec_drafted, engine.spec_accepted, engine.spec_verifies
        t0 = time.perf_counter()
        outs, groups = [], []
        for i in range(0, len(prompts), c):  # c requests at a time, the next c when all of them are done: as for vLLM
            t = time.perf_counter()
            outs += await asyncio.gather(*[one(p) for p in prompts[i:i + c]])
            groups.append(round(time.perf_counter() - t, 4))
        cell.groups, cell.verifies = groups, engine.spec_verifies - v0
        return time.perf_counter() - t0, outs, engine.spec_drafted - d0, engine.spec_accepted - a0

    async def main():
        await engine.start()
        try:
            for name, contents in work.items():
                prompts = tokenized(tok, contents)
                for k in a.draft_tokens:  # CUDA graphs and the verify forward's shapes, before anything is timed
                    await cell(prompts[:4], k, max(a.concurrency))
                # and the prefill graph of every prompt length a request arriving alone will have: captured at a
                # length bucket's second use and judged on the two replays after it (PrefillGraphs)
                first = {}
                for p in prompts:
                    first.setdefault(PrefillGraphs.bucket(len(p)), p)
                for p in first.values():
                    for _ in range(3):
                        r = await engine.add_request(prompt_tokens=p, max_new_tokens=1, temperature=0.0, draft_tokens=0)
                        [t async for t in engine.stream_request(r)]
                for c in a.concurrency:
                    plain = None
                    for k in a.draft_tokens:
                        runs = [await cell(prompts, k, c) for _ in range(a.passes)]  # the last pass is the row
                        secs, outs, drafted, accepted = runs[-1]
                        n = sum(len(o) for o in outs)
                        plain = outs if k == 0 else plain
                        rows.append(dict(
                            engine="turing", model=a.model, dtype=a.dtype, drafter=a.draft_head or "prompt lookup",
                            head_reach=engine.HEAD_REACH if a.draft_head else None,
                            workload=name, requests=len(prompts),
                            concurrency=c, draft_tokens=k, max_new_tokens=a.new_tokens, output_tokens=n,
                            seconds=round(secs, 3), pass_seconds=[round(r[0], 3) for r in runs],
                            group_seconds=cell.groups, group_tokens=[sum(len(o) for o in outs[i:i + c])
                                                                     for i in range(0, len(outs), c)],
                            output_tok_per_s=round(n / secs, 1), drafted_tokens=drafted,
                            accepted_tokens=accepted, acceptance_rate=round(accepted / drafted, 4) if drafted else None,
                            # tokens a target pass as vLLM's row counts them, by the steps that verified a request's
                            # drafts (its accepted count holds tokens past a stop token); a stop token is not output
                            tokens_per_pass=round((n - len(prompts)) / cell.verifies, 3) if cell.verifies else None,
                            verify_steps=cell.verifies,
                            same_tokens_as_no_draft=sum(o == p for o, p in zip(outs, plain)) if k and plain else None))
                        print(json.dumps(rows[-1]), flush=True)
        finally:
            await engine.stop()
    asyncio.run(main())
    return rows, torch.__version__


def run_vllm(a, work):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, GenerationConfig
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
              max_model_len=4096, gpu_memory_utilization=a.gpu_memory_utilization, disable_log_stats=False, **spec)

    def spec_counts():
        """vLLM's speculative-decoding counters so far (drafts, draft tokens, accepted tokens), or {} if this version
        does not report them."""
        try:
            return {m.name.split(":")[-1]: m.value for m in llm.get_metrics()
                    if "spec_decode" in m.name and isinstance(getattr(m, "value", None), (int, float))}
        except Exception:
            return {}
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens)
    # vLLM's token_ids end with the stop token of a request that stopped on one; Turing's stream leaves it out (a
    # stop token is not output). Counted the same way for both: without it (24 tokens of 4,274 on the code workload)
    eos = GenerationConfig.from_pretrained(a.model).eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}

    def output_tokens(o):
        ids = o.outputs[0].token_ids
        return len(ids) - bool(ids and o.outputs[0].finish_reason == "stop" and ids[-1] in stops)
    rows = []
    for name, contents in work.items():
        prompts = [{"prompt_token_ids": p} for p in tokenized(tok, contents)]
        llm.generate(prompts[:4], sp, use_tqdm=False)
        for c in a.concurrency:
            passes = []
            for _ in range(a.passes):  # the last pass is the row
                before = spec_counts()
                t0 = time.perf_counter()
                n = dropped = 0
                groups, group_tokens = [], []
                for i in range(0, len(prompts), c):  # c requests at a time, the next c when all of them are done
                    t = time.perf_counter()
                    outs = llm.generate(prompts[i:i + c], sp, use_tqdm=False)
                    groups.append(round(time.perf_counter() - t, 4))
                    group_tokens.append(sum(output_tokens(o) for o in outs))
                    n += group_tokens[-1]
                    dropped += sum(len(o.outputs[0].token_ids) - output_tokens(o) for o in outs)
                passes.append(time.perf_counter() - t0)
                counts = {k: v - before.get(k, 0) for k, v in spec_counts().items()}
            secs = passes[-1]
            drafted, accepted = counts.get("spec_decode_num_draft_tokens"), counts.get("spec_decode_num_accepted_tokens")
            steps = counts.get("spec_decode_num_drafts")
            # tokens a target pass by vLLM's own count of its drafting steps: its accepted count also holds drafts
            # accepted past a request's last token (30 to 64 tokens a cell on Qwen3-8B), so n / (n - accepted) reads
            # 2 to 6% high. Each request's first token comes from its prefill and its stop token from a step
            rows.append(dict(engine="vllm", model=a.model, dtype=a.dtype,
                             drafter=a.draft_head or "n-gram", workload=name, requests=len(prompts),
                             concurrency=c, draft_tokens=k, max_new_tokens=a.new_tokens, output_tokens=n,
                             stop_tokens_not_counted=dropped, seconds=round(secs, 3),
                             pass_seconds=[round(x, 3) for x in passes], group_seconds=groups,
                             group_tokens=group_tokens, output_tok_per_s=round(n / secs, 1), spec_decode=counts,
                             drafted_tokens=drafted, accepted_tokens=accepted,
                             acceptance_rate=round(accepted / drafted, 4) if drafted else None,
                             tokens_per_pass=round((n + dropped - len(prompts)) / steps, 3) if steps else None))
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
    ap.add_argument("--passes", type=int, default=1,
                    help="passes over the prompts per cell; the last is reported (pass_seconds holds all): with 2, "
                         "what an engine builds at a shape's first use (a graph, a kernel) is outside the row")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--head-reach", type=float, default=None,
                    help="Turing: a draft head's chain stops where the chance of reaching a draft falls below this "
                         "(default the engine's; 0: every chain at --draft-tokens, as vLLM drafts)")
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
    env = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, versions=dict(old.get("versions", {}), **{a.engine: version}), rows=keep + rows), f,
                  indent=2)
    print(f"[+] wrote {a.out}", flush=True)
    if a.engine == "vllm":
        os._exit(0)  # vLLM's helper processes keep the interpreter alive


if __name__ == "__main__":
    main()
