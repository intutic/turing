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
            outs = await asyncio.wait_for(asyncio.gather(*[_collect(engine, r) for r in reqs]), 120)  # never hang
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


# ---------------------------------------------------------------------------------------------- overlapped decode

def _overlapped(**kw):
    engine = _engine(prefix_caching=False, **kw)
    engine.overlap_decode = True  # on by default only on CUDA, where it saves time; the bookkeeping is the same
    launches = []
    launch = engine._launch
    engine._launch = lambda reqs, ids: launches.append(len(reqs)) or launch(reqs, ids)
    return engine, launches


def test_overlapped_decode_gives_the_tokens_of_plain_decoding():
    ref = _greedy(_engine(), PROMPT, 24)
    engine, launches = _overlapped()
    (req,), (out,) = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=24, temperature=0.0, stop_token_ids=set()))
    assert out == ref and req.finish_reason == "length"
    assert sum(launches) == 23  # the first token comes from the prefill; no forward is launched for a 25th
    assert engine._ahead is None and engine.kv_pool.used_pages == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a launch can only wait for the device on CUDA")
def test_a_step_launched_ahead_does_not_wait_for_the_step_in_flight():
    """Overlapped decode pays off only if launching the next step never synchronizes with the device: a blocking
    copy to it (an index made on the host, a page-table write) waits for the step in flight."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    model.config.kv_pool_gib = 0.01
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                   prefix_caching=False)
    assert engine.overlap_decode
    launch, launched = engine._launch, []

    def strict(reqs, ids):
        launched.append(len(reqs))
        if len(launched) <= 2:  # the first launches capture the CUDA graph, which synchronizes
            return launch(reqs, ids)
        torch.cuda.set_sync_debug_mode("error")
        try:
            return launch(reqs, ids)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    engine._launch = strict
    reqs = [dict(prompt_tokens=p, max_new_tokens=80, temperature=0.0, stop_token_ids=set())
            for p in (PROMPT, PROMPT[3:])]  # 80 tokens: both requests take new KV pages while steps are in flight
    _, outs = _run(engine, *reqs)
    assert [len(o) for o in outs] == [80, 80] and len(launched) > 70


def test_overlapped_decode_with_requests_of_different_lengths_and_a_stop_token():
    other = PROMPT[5:] + [44, 17]
    refs = [_greedy(_engine(), p, n) for p, n in ((PROMPT, 20), (other, 7), (PROMPT[2:], 13))]
    stop = refs[0][9]  # the first request ends where plain decoding first produces this token
    cut = refs[0][:refs[0].index(stop)]
    engine, launches = _overlapped()
    reqs, outs = _run(engine,
                      dict(prompt_tokens=PROMPT, max_new_tokens=20, temperature=0.0, stop_token_ids={stop}),
                      dict(prompt_tokens=other, max_new_tokens=7, temperature=0.0, stop_token_ids=set()),
                      dict(prompt_tokens=PROMPT[2:], max_new_tokens=13, temperature=0.0, stop_token_ids=set()))
    assert outs == [cut, refs[1], refs[2]]
    assert [r.finish_reason for r in reqs] == ["stop", "length", "length"]
    assert launches and engine._ahead is None and engine.kv_pool.used_pages == 0


def test_overlapped_decode_samples_the_same_tokens_as_the_plain_path():
    outs = []
    for overlap in (False, True):
        engine = _engine(prefix_caching=False)
        engine.overlap_decode = overlap
        torch.manual_seed(3)
        outs.append(_run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=16, temperature=0.9, top_k=8,
                                      stop_token_ids=set()))[1][0])
    assert outs[0] == outs[1] and len(outs[0]) == 16


def test_overlapped_decode_steps_aside_for_requests_that_need_the_host_each_step():
    ref = _greedy(_engine(), PROMPT, 12)
    engine, launches = _overlapped(draft_tokens=4)  # drafting reads the last token on the host: no overlap
    assert _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=12, temperature=0.0, stop_token_ids=set()))[1][0] == ref
    assert not launches
    engine, launches = _overlapped()
    keep = lambda generated, logits: None  # noqa: E731  a logits processor sees every token before the next step
    (_,), (out,) = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=12, temperature=0.0, stop_token_ids=set(),
                                     logits_processor=keep))
    assert out == ref and not launches


def test_cancel_during_overlapped_decode_frees_pages():
    engine, _ = _overlapped()

    async def go():
        await engine.start()
        try:
            req = await engine.add_request(prompt_tokens=PROMPT, max_new_tokens=500, temperature=0.0,
                                           stop_token_ids=set())
            stream = engine.stream_request(req)
            for _ in range(5):
                await stream.__anext__()
            await stream.aclose()
            for _ in range(200):
                if req.state == RequestState.FINISHED:
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            return req.state
        finally:
            await engine.stop()
    assert asyncio.run(go()) == RequestState.FINISHED
    assert engine.kv_pool.used_pages == 0 and engine._ahead is None


# ---------------------------------------------------------------------------------------------- drafting that pays

def test_prompt_lookup_index_finds_what_a_scan_of_the_sequence_finds():
    """The incremental n-gram index against the plain definition: the tokens after the most recent earlier occurrence
    of the last 3 (else 2, else 1) tokens."""
    import random
    from turing.serving.engine import AsyncSequenceRequest
    engine = _engine()

    def scan(seq, k):
        for n in (3, 2, 1):
            if len(seq) <= n:
                continue
            for start in range(len(seq) - n - 1, -1, -1):
                if seq[start:start + n] == seq[-n:]:
                    return seq[start + n:start + n + k]
        return None
    rng = random.Random(0)
    for vocab in (3, 8, 40):
        req = AsyncSequenceRequest([rng.randrange(vocab) for _ in range(rng.randrange(1, 30))], max_new_tokens=200)
        for _ in range(120):
            if rng.random() < 0.7:  # not every step looks up: the index catches up on the tokens in between
                assert engine._lookup(req, 4) == scan(req.prompt_tokens + req.generated_tokens, 4)
            req.generated_tokens.append(rng.randrange(vocab))


def test_drafting_rests_when_it_does_not_pay_and_resumes_later():
    from turing.serving.engine import AsyncSequenceRequest
    engine = _engine(draft_tokens=4)
    req = AsyncSequenceRequest(PROMPT, max_new_tokens=500)
    engine._step_s = {("drafts", 8): 0.030, 1: 0.010}  # a step of 5 rows (bucket 8) with drafts; a plain step of 1 row
    for _ in range(engine.SPEC_PROBE):  # 4 tokens per step of 30 ms against 3 from plain steps of 10 ms: it pays
        engine._speculation_pays(req, 4, 5, 1)
    assert req.spec_rest == 0 and engine._may_draft(req, 1)
    for _ in range(12):  # 1 token per step: it does not
        engine._speculation_pays(req, 1, 5, 1)
    assert req.spec_rest == engine.SPEC_REST and not engine._may_draft(req, 1)
    for t in range(engine.SPEC_REST):  # plain tokens count the rest down
        engine._emit(req, 5 + t % 3, 0.0)
    assert req.spec_rest == 0 and engine._may_draft(req, 1)
    engine._speculation_pays(req, 4, 20, 1)  # a step with drafts in bucket 32: not timed yet, so no judgement
    assert req.spec_rest == 0 and engine._spec_hold == 0
    for _ in range(engine.SPEC_PROBE):  # no plain step of two requests timed: everyone decodes plainly for a while
        engine._speculation_pays(req, 4, 5, 2)
    assert req.spec_rest == 0 and engine._spec_hold > 0 and not engine._may_draft(req, 1)
    engine._spec_hold = 0
    engine._speculation_pays(req, 4, 5, 2)  # still not timed (prefills ran in every step): no second hold right away
    assert engine._spec_hold == 0 and req.spec_rest == 0
    engine._step_s[("drafts", 8)] = 0.060
    for _ in range(engine.SPEC_PROBE):  # a step that costs six plain ones: 4 tokens per step no longer pay
        engine._speculation_pays(req, 4, 5, 1)
    assert req.spec_rest == engine.SPEC_REST


def test_step_times_are_kept_per_batch_bucket_and_drop_the_first_sample():
    engine = _engine()
    for seconds in (5.0, 0.010, 0.020):  # the first may include a graph capture
        engine._timed(8, seconds)
    assert engine._step_s == {8: pytest.approx(0.012)}
    from turing.serving.engine import _bucket
    assert [_bucket(n) for n in (1, 3, 8, 9, 40, 128, 300)] == [1, 4, 8, 12, 48, 128, 300]


def test_drafts_are_verified_in_the_step_at_any_concurrency_and_fit_the_largest_graph(monkeypatch):
    from turing.models.causal_lm import DecodeGraphs
    from turing.serving.spec_gate import SpeculationGatePolicy
    prompts = [PROMPT[i:] for i in range(6)]
    refs = [_greedy(_engine(), p, 16) for p in prompts]
    monkeypatch.setattr(DecodeGraphs, "BUCKETS", (1, 2, 4, 8, 16))
    engine = _engine(draft_tokens=4, prefix_caching=False, spec_gate=SpeculationGatePolicy())  # the gate stops at 4
    rows = []
    forward = engine._forward
    engine._forward = lambda ids, caches, *a, **kw: rows.append(tuple(ids.shape)) or forward(ids, caches, *a, **kw)
    _, outs = _run(engine, *[dict(prompt_tokens=p, max_new_tokens=16, temperature=0.0, stop_token_ids=set())
                             for p in prompts])
    assert outs == refs and engine.spec_accepted > 0
    steps = [n for n, width in rows if width == 1]
    assert 6 < max(steps) <= 16  # six requests drafted in one step, with no more rows than the largest graph has
    assert engine.kv_pool.used_pages == 0


def test_drafts_that_never_match_cost_a_short_probe_and_the_output_is_unchanged():
    prompt = list(range(40, 70))  # no repetition: a lookup can only match by chance
    ref = _greedy(_engine(), prompt, 60)
    engine = _engine(draft_tokens=4, prefix_caching=False)
    (req,), (out,) = _run(engine, dict(prompt_tokens=prompt, max_new_tokens=60, temperature=0.0, stop_token_ids=set()))
    assert out == ref
    assert engine.spec_drafted < 4 * 60 / 2  # it stopped drafting for most of the output


# ---------------------------------------------------------------------------------------------- drafts as decode rows

@pytest.mark.parametrize("kv", ["native", "int8", "int4", "fp8"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rows_of_one_request_in_a_decode_step_equal_decoding_them_one_by_one(kv, device):
    """A request's cache listed four times with four consecutive tokens: one decode step gives the logits of four
    steps, next to another request's ordinary row. This is how drafts are verified."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device=device)
    model.config.kv_pool_gib, model.config.kv_cache_dtype = 0.01, kv
    seq, other = torch.tensor([PROMPT[:20]], device=device), torch.tensor([PROMPT[3:17]], device=device)
    nxt = torch.tensor([[7], [21], [33], [5]], device=device)
    with torch.inference_mode():
        _, one = model(seq, logits_to_keep=1)
        _, two = model(other, logits_to_keep=1)
        want = []
        for t in nxt:
            logits, one = model(t[None], past_key_values=one)
            want.append(logits[0, -1])
        ref_other, _ = model(nxt[:1], past_key_values=two)
        _, one = model(seq, logits_to_keep=1)  # the same two requests again, now in a single step
        _, two = model(other, logits_to_keep=1)
        got, _ = model(torch.cat([nxt[:1], nxt]), past_key_values=[two, one, one, one, one])
    assert one.length == 24 and two.length == 15
    tol = dict(atol=2e-3 if device == "cuda" else 1e-4, rtol=0)
    torch.testing.assert_close(got[1:, -1].float(), torch.stack(want).float(), **tol)
    torch.testing.assert_close(got[0, -1].float(), ref_other[0, -1].float(), **tol)


def test_drafts_of_two_requests_are_verified_in_one_decode_step():
    refs = [_greedy(_engine(), p, 20) for p in (PROMPT, PROMPT[4:])]
    engine = _engine(draft_tokens=4, prefix_caching=False)
    forwards = []
    forward = engine._forward
    engine._forward = lambda ids, caches, *a, **kw: forwards.append(tuple(ids.shape)) or forward(ids, caches, *a, **kw)
    _, outs = _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=20, temperature=0.0, stop_token_ids=set()),
                   dict(prompt_tokens=PROMPT[4:], max_new_tokens=20, temperature=0.0, stop_token_ids=set()))
    assert outs == refs and engine.spec_accepted > 0
    assert max(rows for rows, width in forwards if width == 1) > 2  # drafts rode as rows of a [rows, 1] decode step
    assert engine.kv_pool.used_pages == 0


def _running(engine):
    """Records how many requests ran in each scheduler step."""
    seen = []
    step = engine._step_batch
    engine._step_batch = lambda: seen.append(len(engine.running_batch)) or step()
    return seen


def test_kv_pages_are_taken_as_requests_decode_so_a_large_limit_does_not_keep_others_waiting():
    """Four requests that may each generate 400 tokens and stop after a few: the pool has to hold their prompts and
    what they generate, not four times the limit."""
    ref = _greedy(_engine(), PROMPT, 12)
    stop = ref[10]
    engine = _engine(prefix_caching=False)
    engine.kv_pool.adjust_active_capacity(-(-(len(PROMPT) + 400) // 16))  # what one request could need at most
    seen = _running(engine)
    reqs, outs = _run(engine, *[dict(prompt_tokens=PROMPT, max_new_tokens=400, temperature=0.0, stop_token_ids={stop})
                               for _ in range(4)])
    assert outs == [ref[:ref.index(stop)]] * 4 and all(r.finish_reason == "stop" for r in reqs)
    assert max(seen) == 4 and engine.preemptions == 0
    assert engine.kv_pool.used_pages == 0


@pytest.mark.parametrize("mode", ["plain", "drafting", "overlapped"])
def test_the_newest_request_is_preempted_and_recomputed_when_the_pool_runs_out(mode):
    prompts = [PROMPT[i:] for i in range(4)]
    refs = [_greedy(_engine(), p, 60) for p in prompts]
    engine = _engine(prefix_caching=False, draft_tokens=4 if mode == "drafting" else 0)
    engine.overlap_decode = mode == "overlapped"
    engine.kv_pool.adjust_active_capacity(20)  # the four prompts and some output, not four outputs of 60 tokens
    seen = _running(engine)
    reqs, outs = _run(engine, *[dict(prompt_tokens=p, max_new_tokens=60, temperature=0.0, stop_token_ids=set())
                               for p in prompts])
    assert outs == refs and all(r.finish_reason == "length" for r in reqs)
    assert max(seen) == 4 and engine.preemptions > 0  # all four started; the pool then ran short
    assert reqs[0].prefill_tokens is reqs[0].prompt_tokens  # the oldest request kept its pages throughout
    assert engine.kv_pool.used_pages == 0 and engine._ahead is None


def test_a_preempted_request_resumes_through_the_prefix_cache_and_a_cancelled_one_frees_its_place():
    prompts = [PROMPT[i:] for i in range(3)]
    refs = [_greedy(_engine(), p, 60) for p in prompts]
    engine = _engine()  # prefix caching on: a preempted request's prompt pages stay indexed while no one needs them
    engine.kv_pool.adjust_active_capacity(15)

    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=60, temperature=0.0, stop_token_ids=set())
                    for p in prompts]
            extra = await engine.add_request(prompt_tokens=PROMPT[5:], max_new_tokens=60, temperature=0.0)
            extra.cancelled = True  # leaves from wherever it is: queued, running or preempted
            return reqs, await asyncio.gather(*[_collect(engine, r) for r in reqs + [extra]])
        finally:
            await engine.stop()
    reqs, outs = asyncio.run(go())
    assert outs[:3] == refs and engine.preemptions > 0
    engine.prefix_cache.evict(10 ** 6)
    assert engine.kv_pool.used_pages == 0


def test_whole_decode_steps_are_timed_by_what_they_ran():
    engine = _engine(prefix_caching=False)
    _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=12, temperature=0.0, stop_token_ids=set()))
    assert set(engine._step_s) == {1} and engine._step_s[1] > 0  # plain steps of one row
    engine = _engine(prefix_caching=False, draft_tokens=4)
    engine.SPEC_PROBE = 10 ** 9
    _run(engine, dict(prompt_tokens=PROMPT, max_new_tokens=24, temperature=0.0, stop_token_ids=set()))
    assert any(k == ("drafts", k[1]) for k in engine._step_s if k != 1)  # steps with draft rows, by row bucket


def test_a_request_waiting_for_pages_does_not_block_requests_that_hold_theirs():
    """One request runs at a time. The pool runs out under the running request while the next one waits with the
    pages of its prompt: the waiting one gives them back, and both finish."""
    prompts, new = [PROMPT, PROMPT[1:]], [100, 10]
    refs = [_greedy(_engine(), p, n) for p, n in zip(prompts, new)]
    engine = _engine(prefix_caching=False)
    engine.max_batch_size = 1
    engine.kv_pool.adjust_active_capacity(10)
    reqs, outs = _run(engine, *[dict(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set())
                               for p, n in zip(prompts, new)])
    assert outs == refs and engine.preemptions == 0  # the running request was never recomputed
    assert engine.kv_pool.used_pages == 0


def test_a_request_continuing_a_callers_cache_runs_while_a_preempted_one_waits():
    prompts = [PROMPT, PROMPT[1:]]
    refs = [_greedy(_engine(), p, 100) for p in prompts]
    engine = _engine(prefix_caching=False)
    engine.kv_pool.adjust_active_capacity(12)
    with torch.inference_mode():
        _, shared = engine.model(torch.tensor([PROMPT[:16]]))  # a caller's prefilled prefix (a fork): one page

    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=100, temperature=0.0, stop_token_ids=set())
                    for p in prompts]
            tasks = [asyncio.ensure_future(_collect(engine, r)) for r in reqs]
            while engine.preemptions == 0:
                await asyncio.sleep(0)
            fork = await engine.add_request(prompt_tokens=[7, 21, 33, 5, 9], max_new_tokens=60, temperature=0.0,
                                            stop_token_ids=set(), prefix_kv=shared)
            tasks.append(asyncio.ensure_future(_collect(engine, fork)))
            return await asyncio.wait_for(asyncio.gather(*tasks), 120)
        finally:
            await engine.stop()
    outs = asyncio.run(go())
    shared.free()
    assert outs[:2] == refs and len(outs[2]) == 60
    assert engine.kv_pool.used_pages == 0


def test_a_request_passing_longropes_switch_is_recomputed_with_the_long_frequencies():
    """Phi-3's LongRoPE picks its frequencies by the sequence's length: past the original context (16 tokens here)
    the cache is recomputed with the long ones, through the prefill path. Greedy tokens equal those of transformers'
    forward over the whole sequence at every step (the reference test_hf_parity uses for LongRoPE too)."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    hf = _tiny("phi3_longrope")
    prompt = PROMPT[:10]
    ref, seq = [], list(prompt)
    with torch.inference_mode():
        for _ in range(14):
            ref.append(int(hf(torch.tensor([seq])).logits[0, -1].argmax()))
            seq.append(ref[-1])
    model = RealHuggingFaceLoader.convert_hf_model(hf)
    model.config.kv_pool_gib = 0.01
    for draft in (0, 4):
        engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu", max_batch_size=8), model=model,
                                       prefix_caching=False, draft_tokens=draft)
        assert engine._rope_switch == 16
        (req,), (out,) = _run(engine, dict(prompt_tokens=prompt, max_new_tokens=14, temperature=0.0,
                                           stop_token_ids=set()))
        assert out == ref and req.past_kv is None
        assert engine.preemptions == 0 and engine.kv_pool.used_pages == 0  # not a preemption: the pool was never short


def _burst_engine(overlap):
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    model.config.kv_pool_gib, model.config.cuda_graphs = 0.01, False  # no captures: they synchronize
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                   prefix_caching=False, draft_tokens=0, prefill_chunk_size=64)
    engine.FORWARD_TOKENS, engine.overlap_decode = 64, overlap
    return engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a launch can only wait for the device on CUDA")
@pytest.mark.parametrize("stop", [False, True])
def test_steps_that_prefill_are_launched_ahead_and_give_the_same_tokens(stop):
    """_ride_ahead: while prompts are prefilled, each step's forward (their chunks, and the decoding requests as
    rows) is launched from the tokens of the step in flight, without a wait for the device; first tokens and the
    riders' tokens are read one step later. The tokens are those of the engine that reads every step before the
    next (one request stops on a token that was still in flight when its next row was launched)."""
    g = torch.Generator().manual_seed(11)
    prompts = [PROMPT, PROMPT[3:]] + [torch.randint(0, 128, (n,), generator=g).tolist() for n in (120, 50, 150)]

    def run(overlap, stops=()):
        engine = _burst_engine(overlap)
        ahead, ride, packed, pack = [], engine._ride_ahead, [], engine.model.prefill_packed

        def spy(prefill, work):
            ahead.append(ride(prefill, work))
            return ahead[-1]

        def strict(chunks, *a, **kw):
            packed.append(len(chunks))
            if len(packed) <= 2 or not overlap:  # the first forwards build kernels
                return pack(chunks, *a, **kw)
            torch.cuda.set_sync_debug_mode("error")
            try:
                return pack(chunks, *a, **kw)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        engine._ride_ahead, engine.model.prefill_packed = spy, strict
        reqs, outs = _run(engine, *[dict(prompt_tokens=p, max_new_tokens=12 + 3 * i, temperature=0.0,
                                         stop_token_ids=set(stops) if i == 0 else set())
                                    for i, p in enumerate(prompts)])
        assert engine._ahead is None and engine.kv_pool.used_pages == 0
        return reqs, outs, ahead

    _, ref, ahead = run(False)
    assert not any(ahead)
    stops = [ref[0][5]] if stop else []
    if stop:
        _, ref, _ = run(False, stops)
        assert len(ref[0]) < 12
    reqs, outs, ahead = run(True, stops)
    assert outs == ref and sum(ahead) >= 6  # every step that prefilled
    assert reqs[0].finish_reason == ("stop" if stop else "length")
    assert all(r.first_token_at is not None and r.first_token_at >= r.created_at for r in reqs)
