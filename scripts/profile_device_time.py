#!/usr/bin/env python3
"""
Where a serving engine's GPU time goes, for Turing and for vLLM (each under its own python): the device's kernels
during one run under torch.profiler (device activity only), as

  busy     the time at least one kernel was running (kernels on two streams at once count once)
  idle     the rest of the span from the first kernel to the last: the device waiting for the host
  summed   every kernel's own time; what exceeds busy ran beside another kernel

with the kernels and the idle gaps that take most. The wall time is measured first, without the profiler.

Two workloads: B requests of a --context-token prompt (64: a model's decode step; longer: a burst that prefills)
decoding N tokens each, or, with --draft-head, --groups groups of C requests of one of
scripts/benchmark_speculation.py's workloads (code by default), each group at once, with an EAGLE-3 head.

python scripts/profile_device_time.py --engine turing --model ~/tgate4/Qwen1.5-MoE-A2.7B-Chat --batch 1
python scripts/profile_device_time.py --engine vllm --model Qwen/Qwen3-8B --draft-head RedHatAI/Qwen3-8B-speculator.eagle3
"""
import _bootstrap  # noqa: F401,E402  (this repo's root first on sys.path)
import argparse
import asyncio
import json
import os
import platform
import re
import tempfile
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # vLLM's engine in this process: its kernels are profiled


def turing_run(a, prompts, tok):
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import TGATE4_FILE, load_quantized
    from turing.serving.engine import ContinuousBatchEngine
    from turing.serving.spec_gate import SpeculationGatePolicy
    if os.path.isfile(os.path.join(a.model, TGATE4_FILE)):  # a directory written by `turing convert`
        model, _ = load_quantized(a.model, device="cuda", dtype=a.dtype)
    else:
        model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device="cuda", dtype=a.dtype)
    model.config.kv_pool_fraction = None  # as turing serve: all free memory but a reserve
    model.config.kv_cache_dtype = a.kv_cache_dtype
    if a.kv_pool_gib:
        model.config.kv_pool_gib = a.kv_pool_gib
    head = None
    if a.draft_head:
        from turing.models.eagle3 import load_eagle3
        head = load_eagle3(a.draft_head, model.config.hidden_dim, model.config.num_layers)
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=a.group), model=model,
                                   tokenizer=tok, prefix_caching=False, draft_tokens=0,
                                   spec_gate=SpeculationGatePolicy(), prefill_chunk_size=a.step_tokens or 2048,
                                   draft_head=head)
    if a.head_reach is not None:  # the draft head's chain rule (ContinuousBatchEngine.HEAD_REACH); 0: always full
        engine.HEAD_REACH = a.head_reach or 1e-30

    async def cell():
        async def one(p):
            r = await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0,
                                         draft_tokens=a.draft_tokens if head else 0,
                                         **({} if head else {"stop_token_ids": set()}))
            return len([t async for t in engine.stream_request(r)])
        v0, n = engine.spec_verifies, 0
        for i in range(0, len(prompts), a.group):  # a group at once, the next when all of it is done
            n += sum(await asyncio.gather(*[one(p) for p in prompts[i:i + a.group]]))
        a.passes = engine.spec_verifies - v0
        return n

    loop = asyncio.new_event_loop()
    loop.run_until_complete(engine.start())
    return lambda: loop.run_until_complete(cell())


def vllm_run(a, prompts, tok):
    from vllm import LLM, SamplingParams
    spec = dict(speculative_config={"method": "eagle3", "model": a.draft_head,
                                    "num_speculative_tokens": a.draft_tokens}) if a.draft_head else {}
    llm = LLM(a.model, dtype={"fp16": "float16", "bf16": "bfloat16"}[a.dtype], enable_prefix_caching=False,
              kv_cache_dtype={"native": "auto", "int8": "fp8", "fp8": "fp8"}[a.kv_cache_dtype],
              **({"max_num_batched_tokens": a.step_tokens} if a.step_tokens else {}),
              max_model_len=max(4096, a.context + a.new_tokens + 16),
              gpu_memory_utilization=a.gpu_memory_utilization, disable_log_stats=not a.draft_head, **spec)
    sp = SamplingParams(temperature=0.0, max_tokens=a.new_tokens, ignore_eos=not a.draft_head)

    def drafts():  # vLLM's count of a request's drafting steps so far
        return sum(m.value for m in llm.get_metrics() if m.name.endswith("spec_decode_num_drafts"))

    def run():
        d0 = drafts() if a.draft_head else 0
        n = sum(len(o.outputs[0].token_ids) for i in range(0, len(prompts), a.group)  # a group at once
                for o in llm.generate([{"prompt_token_ids": p} for p in prompts[i:i + a.group]], sp, use_tqdm=False))
        a.passes = drafts() - d0 if a.draft_head else None
        return n
    return run


def device_time(events, top_n=12):
    """events: the device's kernels and copies as (start us, duration us, stream, name), sorted by start."""
    busy, end, last, gaps, kernels, streams = 0.0, None, "", {}, {}, set()
    for start, dur, stream, name in events:
        k = kernels.setdefault(name, [0, 0.0])
        k[0] += 1
        k[1] += dur
        streams.add(stream)
        if end is None:
            end = start
        if start > end:
            if start - end > 20:  # a gap worth a line: over 20 us with nothing running
                g = gaps.setdefault((last, name), [0, 0.0])
                g[0] += 1
                g[1] += start - end
            busy += dur
            end, last = start + dur, name
        elif start + dur > end:
            busy += start + dur - end
            end, last = start + dur, name
    span = end - events[0][0]
    top = lambda d, n: sorted(([round(ms / 1e3, 2), c, *([k] if isinstance(k, str) else k)]  # noqa: E731
                               for k, (c, ms) in d.items()), reverse=True)[:n]
    return dict(span_ms=round(span / 1e3, 1), busy_ms=round(busy / 1e3, 1), idle_ms=round((span - busy) / 1e3, 1),
                summed_ms=round(sum(v[1] for v in kernels.values()) / 1e3, 1), launches=len(events),
                streams=len(streams), top_kernels=top(kernels, top_n), top_gaps=top(gaps, 8))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["turing", "vllm"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16"])
    ap.add_argument("--batch", type=int, default=1, help="requests at once (the decode workload)")
    ap.add_argument("--new-tokens", type=int, default=None, help="default: 96, or 256 with a draft head")
    ap.add_argument("--context", type=int, default=64, help="prompt tokens of each request (the decode workload)")
    ap.add_argument("--kv-cache-dtype", default="native", choices=["native", "int8", "fp8"],
                    help="KV storage (vLLM has fp8 for both 1-byte ones)")
    ap.add_argument("--step-tokens", type=int, default=None,
                    help="prompt tokens a scheduler step takes: Turing's prefill chunk (default 2,048), vLLM's "
                         "max_num_batched_tokens (default: its own, 8,192 on an L4)")
    ap.add_argument("--kv-pool-gib", type=float, default=None, help="Turing only: the KV pool's size")
    ap.add_argument("--prompts", default="random", choices=["random", "wikitext"],
                    help="the decode workload's prompts: random token ids, or WikiText windows as the benchmark's "
                         "(two engines with different quantized weights continue random ids differently)")
    ap.add_argument("--top", type=int, default=12, help="kernels listed")
    ap.add_argument("--draft-head", default=None, help="an EAGLE-3 head: the --workload's prompts instead")
    ap.add_argument("--draft-tokens", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4, help="requests at once (with a draft head)")
    ap.add_argument("--workload", default="code", choices=["code", "rewrite", "chat"],
                    help="with a draft head: scripts/benchmark_speculation.py's prompts")
    ap.add_argument("--groups", type=int, default=1, help="with a draft head: groups of --concurrency requests")
    ap.add_argument("--head-reach", type=float, default=None, help="Turing only: the draft head's chain rule")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM only")
    ap.add_argument("--code", default=None, help="recorded with the row: the commit of the tree that ran")
    ap.add_argument("--out", default="benchmarks/results/results_device_time_cuda.json")
    a = ap.parse_args()
    a.new_tokens = a.new_tokens or (256 if a.draft_head else 96)
    import torch
    from torch.profiler import ProfilerActivity, profile
    tok = None
    if a.draft_head:  # the benchmark's prompts, ended by EOS as there
        import benchmark_speculation as spec
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = spec.tokenized(tok, spec.WORKLOADS[a.workload](24))[:a.concurrency * a.groups]
    elif a.prompts == "wikitext":  # benchmark_e2e_generation.py's prompts: text both engines continue alike
        import benchmark_e2e_generation as bench
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model)
        prompts = bench.prompts(tok, a.batch, a.context, seed=0)
    else:
        g = torch.Generator().manual_seed(0)
        prompts = torch.randint(1000, 30000, (a.batch, a.context), generator=g).tolist()
    a.group, a.passes = a.concurrency if a.draft_head else len(prompts), None
    run = (turing_run if a.engine == "turing" else vllm_run)(a, prompts, tok)
    for _ in range(3):  # CUDA graphs and kernels built
        tokens = run()
    walls = []
    for _ in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        run()
        torch.cuda.synchronize()
        walls.append(round(time.perf_counter() - t, 4))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.cuda.synchronize()
        t = time.perf_counter()
        run()
        torch.cuda.synchronize()
        profiled = time.perf_counter() - t
    with tempfile.TemporaryDirectory() as tmp:  # the trace names each kernel's stream; prof.events() does not
        prof.export_chrome_trace(os.path.join(tmp, "trace.json"))
        trace = json.load(open(os.path.join(tmp, "trace.json")))["traceEvents"]
    events = sorted((e["ts"], e["dur"], e["args"].get("stream", 0),
                     re.sub(r"void |at::native::|\(anonymous namespace\)::", "", e["name"])[:64])
                    for e in trace if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    name = os.path.basename(os.path.normpath(a.model)) if os.path.isdir(a.model) else a.model
    row = dict(engine=a.engine, model=name, dtype=a.dtype, code=a.code,
               workload=f"eagle3 {a.workload}" if a.draft_head else "decode" if a.context == 64 else
               f"{a.context}-token prompts, {a.kv_cache_dtype} KV, {a.step_tokens or 'default'} tokens a step",
               requests=len(prompts), new_tokens=a.new_tokens, draft_head=a.draft_head,
               draft_tokens=a.draft_tokens if a.draft_head else 0, concurrency=a.group,
               head_reach=a.head_reach, verify_passes=a.passes, output_tokens=tokens, wall_s=walls,
               profiled_wall_s=round(profiled, 4), **device_time(events, a.top))
    print(json.dumps({k: v for k, v in row.items() if not k.startswith("top_")}), flush=True)
    old = json.load(open(a.out)) if os.path.exists(a.out) else {}
    key = lambda r: (r["engine"], r["model"], r["code"], r["workload"], r["requests"], r["draft_tokens"],  # noqa: E731
                     r.get("concurrency"), r.get("head_reach"))
    keep = [r for r in old.get("rows", []) if key(r) != key(row)]
    env = dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0))
    version = torch.__version__
    if a.engine == "vllm":
        import vllm
        version = f"vllm {vllm.__version__}"
    with open(a.out, "w") as f:
        json.dump(dict(env=env, versions=dict(old.get("versions", {}), **{a.engine: version}), rows=keep + [row]), f,
                  indent=2)
    print(f"[+] wrote {a.out}", flush=True)
    os._exit(0)  # an engine's helper threads keep the interpreter alive


if __name__ == "__main__":
    main()
