#!/usr/bin/env python3
"""
Where a batch-1 request's time to first token goes inside ContinuousBatchEngine (speed plan P4.1): from add_request to
the first scheduler step, the prefill forward (synchronized), the sample, the rest of that step (the decode it also
runs), and from the step's end to the consumer receiving the token. Each phase is the median over --runs requests after
warm-up; the model-level prefill of the same prompt is measured alongside, so engine overhead = TTFT - that.

python scripts/profile_engine_ttft.py --model Qwen/Qwen2.5-1.5B-Instruct --context 512
"""
import _bootstrap  # noqa: F401,E402
import argparse
import asyncio
import json
import statistics
import time

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--quantization", default="none", choices=["none", "w4a16-rtn"])
    ap.add_argument("--max-batch-size", type=int, default=8, help="the engine's (the e2e benchmark uses its largest batch)")
    ap.add_argument("--decode-batch", type=int, default=0,
                    help="instead of TTFT: B requests decoding together; per decode step, the synchronized forward "
                         "against the whole step (speed plan P6.2), and the forward's kernels by device time")
    ap.add_argument("--out", default="benchmarks/results/results_engine_ttft_cuda.json")
    a = ap.parse_args()
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    model, tok = RealHuggingFaceLoader.load_hf_model_into_turing(a.model, device="cuda", dtype="fp16")
    if a.quantization != "none":
        from turing.models.quant import quantize_model
        quantize_model(model, "rtn")
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=a.max_batch_size), model=model,
                                   tokenizer=tok, prefix_caching=False, draft_tokens=0, prefill_chunk_size=2048)
    g = torch.Generator().manual_seed(0)
    marks = {}

    fwd, step = engine._forward, engine._step_batch

    def timed_forward(ids, caches, adapters=None, logits_to_keep=0):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fwd(ids, caches, adapters, logits_to_keep)
        torch.cuda.synchronize()
        if ids.shape[1] > 1:
            marks.setdefault("prefill_forward", time.perf_counter() - t)
        return out

    def timed_step():
        marks.setdefault("first_step_start", time.perf_counter())
        step()
        marks.setdefault("first_step_end", time.perf_counter())

    engine._forward, engine._step_batch = timed_forward, timed_step
    emit = engine._emit

    def timed_emit(req, token, now):
        marks.setdefault("emit", time.perf_counter())
        return emit(req, token, now)
    engine._emit = timed_emit

    rows = []
    if a.decode_batch:
        return decode_batch(a, engine, model, g)

    async def one(prompt):
        marks.clear()
        t0 = time.perf_counter()
        req = await engine.add_request(prompt_tokens=prompt, max_new_tokens=2, temperature=0.0, stop_token_ids=set())
        marks["added"] = time.perf_counter()
        first = None
        async for _ in engine.stream_request(req):
            if first is None:
                first = time.perf_counter()
        m = dict(marks)
        return dict(ttft=first - t0, add=m["added"] - t0, wait=m["first_step_start"] - t0,
                    prefill_forward=m.get("prefill_forward"), sample=m["emit"] - m["first_step_start"] - m.get("prefill_forward", 0),
                    rest_of_step=m["first_step_end"] - m["emit"], to_consumer=first - m["first_step_end"])

    async def run():
        await engine.start()
        try:
            for _ in range(3):
                await one(torch.randint(0, model.config.vocab_size, (a.context,), generator=g).tolist())
            for _ in range(a.runs):
                rows.append(await one(torch.randint(0, model.config.vocab_size, (a.context,), generator=g).tolist()))
        finally:
            await engine.stop()
    asyncio.run(run())

    ids = torch.randint(0, model.config.vocab_size, (1, a.context), generator=g).cuda()
    ts = []
    with torch.inference_mode():
        for _ in range(5):
            torch.cuda.synchronize()
            t = time.perf_counter()
            model(ids, logits_to_keep=1)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    med["model_prefill"] = statistics.median(ts[1:])
    print(json.dumps({k: round(v * 1e3, 2) for k, v in med.items()}, indent=1), "(ms)")
    with open(a.out, "w") as f:
        json.dump(dict(model=a.model, context=a.context, quantization=a.quantization, runs=a.runs,
                       max_batch_size=a.max_batch_size, median_s=med, rows=rows,
                       note="phases of one batch-1 request's TTFT inside the engine"), f, indent=2)


def decode_batch(a, engine, model, g):
    """Steps where every one of a.decode_batch requests decodes: wall time of the step and of its forward (both
    synchronized), so step - forward is the engine's own per-step work; then one such forward under torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    B, fwd, step = a.decode_batch, engine._forward, engine._step_batch
    cur, steps, prof = {}, [], {}

    def timed_forward(ids, caches, adapters=None, logits_to_keep=0):
        torch.cuda.synchronize()
        t = time.perf_counter()
        if ids.shape == (B, 1) and len(steps) == 40 and not prof:
            with profile(activities=[ProfilerActivity.CUDA]) as p:
                out = fwd(ids, caches, adapters, logits_to_keep)
                torch.cuda.synchronize()
            prof["p"] = p
        else:
            out = fwd(ids, caches, adapters, logits_to_keep)
        torch.cuda.synchronize()
        cur["fwd"] = cur.get("fwd", 0) + time.perf_counter() - t
        cur["shape"] = tuple(ids.shape)
        return out

    def timed_step():
        cur.clear()
        torch.cuda.synchronize()
        t = time.perf_counter()
        step()
        torch.cuda.synchronize()
        if cur.get("shape") == (B, 1):  # a step of pure batched decode
            steps.append((time.perf_counter() - t, cur["fwd"]))
    engine._forward, engine._step_batch = timed_forward, timed_step

    async def run():
        await engine.start()
        try:
            for n in (16, 128):  # warm-up (graph capture at this batch size), then the measured run
                steps.clear()
                ps = [torch.randint(0, model.config.vocab_size, (a.context,), generator=g).tolist() for _ in range(B)]
                reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=n, temperature=0.0,
                                                 stop_token_ids=set()) for p in ps]
                await asyncio.gather(*[_drain(engine, r) for r in reqs])
        finally:
            await engine.stop()
    asyncio.run(run())
    us = {}
    for e in prof["p"].events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            us[e.name[:90]] = us.get(e.name[:90], 0) + e.time_range.elapsed_us()
    res = dict(model=a.model, quantization=a.quantization, batch=B, context=a.context, steps=len(steps),
               step_ms=statistics.median(s for s, _ in steps) * 1e3,
               forward_ms=statistics.median(f for _, f in steps) * 1e3,
               forward_device_ms=sum(us.values()) / 1e3,
               top_kernels=sorted(([k, round(v, 1)] for k, v in us.items()), key=lambda x: -x[1])[:25],
               note="medians over steps where all B requests decode; step - forward = the engine's own work")
    print(json.dumps({k: v for k, v in res.items() if k != "top_kernels"}, indent=1))
    rows = []
    try:
        with open(a.out) as f:
            rows = json.load(f).get("rows", [])
    except (OSError, ValueError):
        pass
    with open(a.out, "w") as f:
        json.dump(dict(rows=rows + [res]), f, indent=2)


async def _drain(engine, req):
    async for _ in engine.stream_request(req):
        pass


if __name__ == "__main__":
    main()
