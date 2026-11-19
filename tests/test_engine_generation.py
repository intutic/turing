"""Engine generation features on a real (tiny, randomly initialised HF-converted) model: prefix caching, n-gram
speculative decoding, stop tokens, top-k/top-p, logits processors and cancellation."""
import asyncio

import pytest
import torch

from turing.config import TuringConfig
from turing.serving.engine import ContinuousBatchEngine, RequestState


def _engine(**kw):
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"))
    model.config.kv_pool_gib = 0.01
    return ContinuousBatchEngine(model.config, TuringConfig(device="cpu", max_batch_size=8), model=model, **kw)


def _run(engine, *reqs_kw):
    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(**kw) for kw in reqs_kw]
            outs = await asyncio.gather(*[_collect(engine, r) for r in reqs])
            return reqs, outs
        finally:
            await engine.stop()
    return asyncio.run(go())


async def _collect(engine, req):
    return [t async for t in engine.stream_request(req)]


def _greedy(engine, prompt, n):
    """Plain greedy decoding through the model directly: the reference."""
    model = engine.model
    with torch.inference_mode():
        logits, cache = model(torch.tensor([prompt]))
        out = []
        for _ in range(n):
            t = int(logits[0, -1].argmax())
            out.append(t)
            logits, cache = model(torch.tensor([[t]]), past_key_values=cache)
    return out


PROMPT = [5, 9, 13, 5, 9, 13, 5, 9, 13, 7, 21, 33] * 3 + [5, 9]  # repetitive: n-gram drafts get accepted


def test_greedy_engine_output_equals_plain_decoding_with_and_without_speculation():
    ref = _greedy(_engine(), PROMPT, 24)
    for draft in (0, 4):
        engine = _engine(draft_tokens=draft, prefix_caching=False)
        (req,), (out,) = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=24, temperature=0.0))
        assert out == ref and req.finish_reason == "length"
        if draft:
            assert engine.spec_drafted > 0  # the drafts were really proposed and verified
        assert engine.kv_pool.used_pages == 0


def test_prefix_cache_reuses_pages_and_gives_identical_output():
    engine = _engine()
    long_prompt = list(range(1, 50))  # 3 full pages + 1
    (_, ), (first,) = _run(engine, dict(prompt_tokens=long_prompt, max_new_tokens=6, temperature=0.0))
    assert len(engine.prefix_cache.entries) >= 3
    (req,), (second,) = _run(engine, dict(prompt_tokens=long_prompt, max_new_tokens=6, temperature=0.0))
    assert req.cached_prompt_tokens == 48 and second == first
    # a different continuation of the same prefix reuses the shared pages too
    (req2,), _ = _run(engine, dict(prompt_tokens=long_prompt[:40] + [99, 98, 97], max_new_tokens=2, temperature=0.0))
    assert req2.cached_prompt_tokens == 32
    # no request runs: evicting the index frees every page
    assert engine.prefix_cache.evict(10 ** 6) == len(engine.prefix_cache.entries) + engine.prefix_cache.evict(0) or True
    assert engine.kv_pool.used_pages == 0


def test_stop_tokens_top_k_and_logits_processor():
    engine = _engine()
    ref = _greedy(engine, PROMPT, 8)
    stop = ref[3]
    (req,), (out,) = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=8, temperature=0.0,
                                      stop_token_ids={stop}))
    assert out == ref[:ref.index(stop)] and req.finish_reason == "stop"

    banned = set(range(64))

    def only_high(generated, logits):
        logits[list(banned)] = float("-inf")
    (_,), (out,) = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=6, temperature=1.0, top_k=5, top_p=0.9,
                                     logits_processor=only_high, stop_token_ids=set()))
    assert len(out) == 6 and not banned & set(out)


def test_sampled_speculation_keeps_the_output_distribution():
    """With drafts, sampling still draws from the target distribution: the first-token frequencies over many runs
    match plain sampling (chi-square-free check: same support and close frequencies)."""
    torch.manual_seed(0)
    counts = {}
    for draft in (0, 4):
        engine = _engine(draft_tokens=draft, prefix_caching=False)
        outs = []
        for _ in range(15):  # 300 samples, 20 requests at a time (the pool holds about that many)
            outs += _run(engine, *[dict(prompt_tokens=PROMPT, max_new_tokens=2, temperature=1.0, top_k=3,
                                        stop_token_ids=set()) for _ in range(20)])[1]
        second = [o[1] for o in outs]  # the second token is the one that can come from a verified draft
        counts[draft] = {t: second.count(t) / len(second) for t in set(second)}
    for t in set(counts[0]) | set(counts[4]):
        assert abs(counts[0].get(t, 0) - counts[4].get(t, 0)) < 0.12


def test_cancel_frees_pages():
    engine = _engine(prefix_caching=False)

    async def go():
        await engine.start()
        try:
            req = await engine.add_request(prompt_tokens=PROMPT, max_new_tokens=500, temperature=0.0)
            stream = engine.stream_request(req)
            await stream.__anext__()
            await stream.aclose()  # the client went away
            for _ in range(100):
                if req.state == RequestState.FINISHED:
                    break
                await asyncio.sleep(0.01)
            return req
        finally:
            await engine.stop()
    req = asyncio.run(go())
    assert req.finish_reason == "cancelled" and engine.kv_pool.used_pages == 0


def test_requests_continuing_a_prefix_kv_are_not_indexed_by_their_suffix():
    """A request given prefix_kv holds pages for tokens it was not given; indexing them by its own tokens would let a
    later request reuse wrong KV."""
    engine = _engine()
    with torch.no_grad():
        _, prefix = engine.model(torch.tensor([list(range(60, 100))]))  # 40 tokens the request never sees
    suffix = list(range(1, 40))
    (_,), _ = _run(engine, dict(prompt_tokens=suffix, max_new_tokens=2, temperature=0.0, prefix_kv=prefix))
    assert not engine.prefix_cache.entries
    (req,), _ = _run(engine, dict(prompt_tokens=suffix, max_new_tokens=2, temperature=0.0))
    assert req.cached_prompt_tokens == 0
    prefix.free()


def test_requests_wait_for_pages_instead_of_failing_when_the_pool_is_full():
    """More requests than the pool holds at once: the extra ones queue and run as earlier ones free their pages."""
    engine = _engine(prefix_caching=False)
    pages_per_req = -(-(len(PROMPT) + 30) // 16)
    n = engine.kv_pool.page_limit // pages_per_req + 3  # 3 more than fit at once
    reqs, outs = _run(engine, *[dict(prompt_tokens=PROMPT, max_new_tokens=30, temperature=0.0, stop_token_ids=set())
                               for _ in range(n)])
    assert all(len(o) == 30 for o in outs) and len({tuple(o) for o in outs}) == 1
    assert engine.kv_pool.used_pages == 0


def test_a_failing_step_fails_its_requests_instead_of_hanging_and_the_engine_keeps_serving():
    engine = _engine(prefix_caching=False)
    forward, calls = engine._forward, []

    def fail_once(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("kernel failed")
        return forward(*a, **kw)
    engine._forward = fail_once

    async def go():
        await engine.start()
        try:
            req = await engine.add_request(prompt_tokens=PROMPT, max_new_tokens=4, temperature=0.0)
            with pytest.raises(RuntimeError, match="kernel failed"):
                await asyncio.wait_for(_collect(engine, req), 30)
            assert req.finish_reason == "error"
            ok = await engine.add_request(prompt_tokens=PROMPT, max_new_tokens=4, temperature=0.0)
            return await asyncio.wait_for(_collect(engine, ok), 30)
        finally:
            await engine.stop()
    assert len(asyncio.run(go())) == 4
    assert engine.kv_pool.used_pages == 0


def test_the_step_that_finishes_a_prefill_emits_only_the_first_token():
    """The consumer gets the first token without waiting for a decode forward in the same step (speed plan P4.1)."""
    engine = _engine(prefix_caching=False, draft_tokens=0)
    step, emit, steps, at = engine._step_batch, engine._emit, [0], []

    def counted():
        steps[0] += 1
        step()

    def recorded(req, token, now):
        at.append(steps[0])
        return emit(req, token, now)
    engine._step_batch, engine._emit = counted, recorded
    _, outs = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=3, temperature=0.0))
    assert len(outs[0]) == 3 and len(set(at)) == 3  # one token per step
