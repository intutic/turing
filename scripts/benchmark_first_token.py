"""Time to the first token of one prompt arriving alone, by prompt length, with its prefill replayed from a CUDA graph
(PrefillGraphs, the default on CUDA up to 1,024 tokens) and run eagerly. --kv-cache-dtype int8 measures it on the
pool `turing serve` uses by default on CUDA.

Each length: 12 random prompts per pass, four passes alternating eager and graph on one engine; the first two prompts
of a pass are dropped (a bucket's first use is eager and its second captures the graph) and the median of the other
20 per variant is reported, with each pass's own median. The first four tokens of every prompt are compared between
the two variants (greedy).

python scripts/benchmark_first_token.py --models Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-360M
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turing.config import TuringConfig  # noqa: E402
from turing.models.causal_lm import PrefillGraphs  # noqa: E402
from turing.models.hf_loader import RealHuggingFaceLoader  # noqa: E402
from turing.serving.engine import ContinuousBatchEngine  # noqa: E402

PROMPTS, DROPPED = 12, 2


async def first_tokens(engine, ids):
    """(ms to the first token, the tokens generated) for each prompt, one request at a time."""
    ms, toks = [], []
    for row in ids.tolist():
        t = time.perf_counter()
        r = await engine.add_request(prompt_tokens=row, max_new_tokens=4, temperature=0.0, stop_token_ids=set())
        first, got = None, []
        async for x in engine.stream_request(r):
            first = first or time.perf_counter() - t
            got.append(x)
        ms.append(first * 1e3)
        toks.append(got)
    return ms, toks


async def run_model(model_id, lengths, dtype, kv):
    model, _ = RealHuggingFaceLoader.load_hf_model_into_turing(model_id, device="cuda", dtype=dtype)
    model.config.kv_pool_gib, model.config.kv_cache_dtype = 2.0, kv
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                   prefix_caching=False, draft_tokens=0)
    g = torch.Generator().manual_seed(0)
    eligible, rows = PrefillGraphs.eligible, []
    await engine.start()
    try:
        for n in lengths:
            ids = torch.randint(0, model.config.vocab_size, (PROMPTS, n), generator=g)
            ms, toks = {"eager": [], "graph": []}, {}
            for variant in ("eager", "graph", "eager", "graph"):
                PrefillGraphs.eligible = eligible if variant == "graph" else (lambda *a, **kw: False)
                got, t = await first_tokens(engine, ids)
                ms[variant].append(got[DROPPED:])
                toks.setdefault(variant, t)
            same = sum(a == b for a, b in zip(toks["eager"], toks["graph"]))
            for variant, passes in ms.items():
                rows.append(dict(model=model_id, dtype=dtype, kv_cache_dtype=kv, tokens=n, prefill=variant,
                                 graph_tokens=PrefillGraphs.bucket(n) if variant == "graph" else None,
                                 first_token_ms=round(statistics.median(passes[0] + passes[1]), 2),
                                 n=len(passes[0]) + len(passes[1]),
                                 pass_medians_ms=[round(statistics.median(p), 2) for p in passes],
                                 prompts_with_the_same_tokens=same, prompts=PROMPTS))
                print(json.dumps(rows[-1]), flush=True)
    finally:
        PrefillGraphs.eligible = eligible
        await engine.stop()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--lengths", default="64,200,512,1024", help="prompt tokens")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--kv-cache-dtype", default="native", choices=["native", "int8", "fp8"])
    ap.add_argument("--out", default="benchmarks/results/results_first_token_cuda.json")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("PrefillGraphs is CUDA-only: nothing to compare here")
    rows = []
    for name in a.models.split(","):
        rows += asyncio.run(run_model(name, [int(x) for x in a.lengths.split(",")], a.dtype, a.kv_cache_dtype))
        torch.cuda.empty_cache()
    json.dump(dict(env=dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__), args=vars(a), rows=rows),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
