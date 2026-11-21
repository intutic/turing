#!/usr/bin/env python3
"""
Where greedy output with speculative drafting differs from plain greedy decoding, and how close the model's choice
was there. A verify pass and a decode step run different kernels, so in fp16 their logits differ in the last bits: a
different token at a near-tie is rounding, a different token at a clear margin would be a bug. For every request
whose two outputs differ, this reports the first differing position and, from one forward over the common prefix,
the gap between the two tokens' logits and whether they are the model's top two.

python scripts/diag_spec_divergence.py --workloads chat --requests 60
"""
import _bootstrap  # noqa: F401,E402
import argparse
import asyncio
import json
import platform
import statistics

import torch

from benchmark_speculation import WORKLOADS, tokenized
from turing.config import TuringConfig
from turing.models.hf_loader import RealHuggingFaceLoader
from turing.serving.engine import ContinuousBatchEngine
from turing.serving.spec_gate import SpeculationGatePolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workloads", default="chat")
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--new-tokens", type=int, default=256)
    ap.add_argument("--draft-tokens", type=int, default=4)
    ap.add_argument("--draft-head", default=None, help="an EAGLE-3 head that drafts instead of prompt lookup")
    ap.add_argument("--out", default="benchmarks/results/results_spec_divergence_cuda.json")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device=a.device, dtype=a.dtype)
    head = None
    if a.draft_head:
        from turing.models.eagle3 import load_eagle3
        head = load_eagle3(a.draft_head, model.config.hidden_dim, model.config.num_layers)
    engine = ContinuousBatchEngine(model.config, TuringConfig(device=a.device, max_batch_size=1), model=model,
                                   tokenizer=tok, prefix_caching=False, draft_tokens=0,
                                   spec_gate=SpeculationGatePolicy(), draft_head=head)
    engine.SPEC_PROBE = 10 ** 9  # never rest: every step that finds a draft verifies it

    async def generate(prompts, k):
        outs = []
        for p in prompts:
            r = await engine.add_request(prompt_tokens=p, max_new_tokens=a.new_tokens, temperature=0.0, draft_tokens=k)
            outs.append([t async for t in engine.stream_request(r)])
        return outs

    async def run():
        await engine.start()
        try:
            rows = []
            for name in a.workloads.split(","):
                prompts = tokenized(tok, WORKLOADS[name](a.requests))
                plain, drafted = await generate(prompts, 0), await generate(prompts, a.draft_tokens)
                for i, (p, x, y) in enumerate(zip(prompts, plain, drafted)):
                    if x == y:
                        continue
                    at = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                    row = dict(workload=name, request=i, first_difference=at, plain_tokens=len(x), drafted_tokens=len(y))
                    if at < min(len(x), len(y)):  # else one output is a prefix of the other (a stop token on one path)
                        with torch.inference_mode():
                            logits, cache = model(torch.tensor([p + x[:at]], device=a.device), logits_to_keep=1)
                        cache.free()
                        last = logits[0, -1].float()
                        top = last.topk(2)
                        row.update(gap=float((last[x[at]] - last[y[at]]).abs()),
                                   top2_gap=float(top.values[0] - top.values[1]),
                                   both_in_top2={x[at], y[at]} == set(top.indices.tolist()))
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                gaps = [r["gap"] for r in rows if r["workload"] == name and "gap" in r]
                print(json.dumps(dict(workload=name, requests=len(prompts), differ=sum(r["workload"] == name for r in rows),
                                      median_gap=statistics.median(gaps) if gaps else None,
                                      max_gap=max(gaps) if gaps else None,
                                      all_in_top2=all(r.get("both_in_top2", True) for r in rows if r["workload"] == name))),
                      flush=True)
            return rows
        finally:
            await engine.stop()
    rows = asyncio.run(run())
    env = dict(platform=platform.platform(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, model=a.model, dtype=a.dtype,
               draft_tokens=a.draft_tokens, drafter=a.draft_head or "prompt lookup", requests=a.requests, new_tokens=a.new_tokens)
    with open(a.out, "w") as f:
        json.dump(dict(env=env, rows=rows), f, indent=2)
    print(f"[+] wrote {a.out}")


if __name__ == "__main__":
    main()
