"""EAGLE-3 draft head (turing/models/eagle3.py): the cached pass against a full recompute, the greedy draft chain,
both checkpoint layouts, and the refused variants."""

import json
import warnings

import pytest
import torch
from safetensors.torch import save_file

from turing.models.eagle3 import Eagle3Head, default_aux_layers, load_eagle3

H, V, DV = 32, 50, 20


def _head(**kw):
    torch.manual_seed(0)
    head = Eagle3Head(hidden=H, intermediate=48, num_heads=4, num_kv_heads=2, head_dim=8, draft_vocab=DV, **kw).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.2)
        head.d2t.copy_(torch.arange(DV) * 2 - torch.arange(DV))  # draft id i -> target id 2 i
    return head


def _inputs(T, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Embedding(V, H), torch.randn(T, 3 * H, generator=g), torch.randint(0, V, (T,), generator=g)


def test_default_aux_layers_are_low_mid_high():
    assert default_aux_layers(36) == (2, 18, 33) and default_aux_layers(28) == (2, 14, 25)


@torch.no_grad()
@pytest.mark.parametrize("norm_before_residual", [False, True])
def test_extending_in_chunks_equals_one_pass(norm_before_residual):
    head = _head(norm_before_residual=norm_before_residual)
    embed, aux, toks = _inputs(9)
    whole = head.extend(embed, aux, toks, 5, None)
    cache = head.extend(embed, aux[:4], toks[:4], 5, None)
    cache = head.extend(embed, aux[4:], toks[4:], 9, cache)
    for a, b in zip(whole, cache):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@torch.no_grad()
def test_norm_before_residual_changes_the_output():
    embed, aux, toks = _inputs(3)
    a = _head(norm_before_residual=False)._layer(embed(toks), _head().fc(aux), 0, None)[1]
    b = _head(norm_before_residual=True)._layer(embed(toks), _head().fc(aux), 0, None)[1]
    assert not torch.allclose(a, b)


@torch.no_grad()
def test_draft_chain_feeds_back_the_heads_own_output_and_maps_through_d2t():
    head = _head()
    embed, aux, toks = _inputs(6)
    drafts, cache = head.draft(embed, aux, toks, 10, 3, None)
    assert len(drafts) == 3 and all(d % 2 == 0 and 0 <= d < 2 * DV for d in drafts)  # d2t: target id = 2 * draft id
    assert cache[0].shape[2] == 6  # the cache holds the verified positions only, not the drafted ones
    # the same chain without the cache: every step recomputes all positions from scratch
    emb, feat = embed(toks), head.fc(aux)
    want = []
    for i in range(3):
        logits, out, _ = head._layer(emb, feat, 10, None)
        tok = int(logits.argmax()) * 2
        want.append(tok)
        emb = torch.cat([emb, embed(torch.tensor([tok]))])
        feat = torch.cat([feat, out[-1:]])
    assert drafts == want
    again, _ = head.draft(embed, aux[3:], toks[3:], 13, 3, head.extend(embed, aux[:3], toks[:3], 10, None))
    assert again == drafts  # verified positions taken in over two calls


@torch.no_grad()
def test_drafting_for_several_requests_at_once_equals_drafting_for_each():
    """One pass of the head's layers over the rows of three requests (different numbers of verified positions,
    chain lengths, starts, one with a cache from an earlier call) gives each request what a call of its own gives."""
    head = _head()
    embed = _inputs(1)[0]
    work = []
    for seed, (T, start, k, seen) in enumerate([(6, 10, 3, 0), (2, 40, 5, 4), (1, 7, 1, 0)]):
        _, aux, toks = _inputs(T + seen, seed=seed + 3)
        cache = head.extend(embed, aux[:seen], toks[:seen], start - seen, None) if seen else None
        work.append((aux[seen:], toks[seen:], start, k, cache))
    many = head.draft_many(embed, work)
    for item, (drafts, cache) in zip(work, many):
        want, want_cache = head.draft(embed, *item)
        assert drafts == want and len(drafts) == item[3]
        for a, b in zip(cache, want_cache):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the head's paged pool uses the Triton decode kernel")
@torch.no_grad()
def test_the_heads_paged_pool_drafts_what_its_tensor_caches_draft():
    """use_pool: keys and values in a paged pool, attention in the decode kernel. Three requests over three calls
    (verified positions arriving in pieces, chains of different lengths, one request only taking positions in) draft
    what the SDPA path over per-request tensors drafts."""
    torch.manual_seed(0)
    head = Eagle3Head(hidden=64, intermediate=96, num_heads=4, num_kv_heads=2, head_dim=16, draft_vocab=DV).eval()
    for p in head.parameters():
        p.normal_(0.0, 0.2)
    head.d2t.copy_(torch.arange(DV) * 2 - torch.arange(DV))
    head = head.to("cuda", torch.float16)
    g = torch.Generator().manual_seed(5)
    embed = torch.nn.Embedding(V, 64).to("cuda", torch.float16)
    calls = []  # per call: (aux, next_tokens, start, k) for each of three requests
    starts = [10, 40, 7]
    for T, ks in ([(6, 2, 1), (3, 5, 1)], [(2, 4, 3), (4, 1, 0)], [(1, 1, 5), (3, 6, 2)]):
        call = []
        for r, (n, k) in enumerate(zip(T, ks)):
            call.append((torch.randn(n, 192, generator=g).to("cuda", torch.float16),
                         torch.randint(0, V, (n,), generator=g).cuda(), starts[r], k))
            starts[r] += n
        calls.append(call)

    def run():
        caches, out = [None, None, None], []
        for call in calls:
            got = head.draft_many(embed, [(*item, cache) for item, cache in zip(call, caches)])
            caches = [c for _, c in got]
            out.append([d for d, _ in got])
        return out
    want = run()
    assert head.use_pool(4096)
    assert run() == want and head.graphs  # each step a replay of its CUDA graph
    # a chain's steps are queued one behind the other: a call waits for the device once, to read its drafts
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as waits:
            warnings.simplefilter("always")
            assert run() == want
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert len(waits) == len(calls), [str(w.message) for w in waits]
    head.graphs = None
    assert run() == want  # and with the step's kernels launched one by one
    assert all(len(d) == item[3] for call, drafts in zip(calls, want) for item, d in zip(call, drafts))


def _write(tmp_path, head, layout, **extra):
    sd = {k: v.clone() for k, v in head.state_dict().items()}
    prefix = "midlayer." if layout == "original" else "layers.0."
    out = {}
    for k, v in sd.items():
        if k.split(".")[0] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[prefix + "self_attn." + k] = v
        elif k.split(".")[0] in ("gate_proj", "up_proj", "down_proj"):
            out[prefix + "mlp." + k] = v
        elif k.split(".")[0] in ("input_layernorm", "hidden_norm", "post_attention_layernorm"):
            out[prefix + k] = v
        else:
            out[k] = v
    out["t2d"] = torch.zeros(V, dtype=torch.bool)
    layer = dict(hidden_size=H, intermediate_size=48, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                 num_hidden_layers=1, rms_norm_eps=1e-6, rope_theta=10000.0, max_position_embeddings=4096)
    if layout == "original":
        cfg = dict(layer, draft_vocab_size=DV, **extra)
    else:
        out["embed_tokens.weight"] = torch.zeros(V, H)
        cfg = dict(transformer_layer_config=layer, draft_vocab_size=DV, norm_before_residual=True, **extra)
    save_file(out, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(cfg))


@torch.no_grad()
@pytest.mark.parametrize("layout", ["original", "speculators"])
def test_load_eagle3_reads_both_checkpoint_layouts(layout, tmp_path):
    head = _head(norm_before_residual=layout == "speculators")
    _write(tmp_path, head, layout)
    loaded = load_eagle3(str(tmp_path), target_hidden=H, target_layers=36)
    assert loaded.aux_layers == (2, 18, 33) and loaded.norm_before_residual == (layout == "speculators")
    embed, aux, toks = _inputs(5)
    assert loaded.draft(embed, aux, toks, 0, 4, None)[0] == head.draft(embed, aux, toks, 0, 4, None)[0]


def test_load_eagle3_refuses_variants_it_does_not_implement(tmp_path):
    _write(tmp_path, _head(), "original", fc_norm=True)
    with pytest.raises(ValueError, match="fc_norm"):
        load_eagle3(str(tmp_path), target_hidden=H, target_layers=36)
    _write(tmp_path, _head(), "original")
    with pytest.raises(ValueError, match="hidden size"):
        load_eagle3(str(tmp_path), target_hidden=2 * H, target_layers=36)


# ---------------------------------------------------------------------------------------------- in the engine

import asyncio  # noqa: E402

from turing.config import ModelConfig, TuringConfig  # noqa: E402
from turing.models.causal_lm import SubspaceCausalLM  # noqa: E402
from turing.serving.engine import ContinuousBatchEngine  # noqa: E402

PROMPT = [5, 9, 13, 7, 21, 33, 40, 2, 17, 8, 3, 11, 29, 64, 90, 101, 77, 6, 12, 55, 31, 4]


def _target(device="cpu"):
    torch.manual_seed(0)
    cfg = ModelConfig(name="eagle-target", hidden_dim=64, ffn_dim=128, num_layers=4, num_heads=4, num_kv_heads=2,
                      head_dim=16, vocab_size=128)
    cfg.kv_pool_gib = 0.01
    return SubspaceCausalLM(cfg).eval().to(device)


def _engine(model, head=None, device="cpu", **kw):
    return ContinuousBatchEngine(model.config, TuringConfig(device=device, max_batch_size=4), model=model,
                                 prefix_caching=False, draft_head=head, **kw)


def _target_head(model):
    torch.manual_seed(1)
    head = Eagle3Head(hidden=64, intermediate=96, num_heads=4, num_kv_heads=2, head_dim=16, draft_vocab=128,
                      aux_layers=(1, 2, 3)).eval()
    with torch.no_grad():
        for p in head.parameters():
            p.normal_(0.0, 0.1)
    return head


def _generate(engine, n, prompts=(PROMPT,), **kw):
    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set(),
                                             **kw) for p in prompts]
            return await asyncio.gather(*[_collect(engine, r) for r in reqs])
        finally:
            await engine.stop()

    async def _collect(engine, r):
        return [t async for t in engine.stream_request(r)]
    return asyncio.run(go())


def test_a_draft_head_never_changes_greedy_output():
    model = _target()
    ref = _generate(_engine(model), 20)[0]
    for chunk in (512, 8):  # one prefill forward, and three chunks the head takes in one after another
        engine = _engine(model, _target_head(model), draft_tokens=3, prefill_chunk_size=chunk)
        assert _generate(engine, 20)[0] == ref
        assert engine.spec_drafted > 0 and engine.kv_pool.used_pages == 0
    assert model.aux_layers == (1, 2, 3)


def test_correct_drafts_are_all_accepted_and_the_head_sees_every_verified_position():
    model = _target()
    ref = _generate(_engine(model), 20)[0]
    engine = _engine(model, _target_head(model), draft_tokens=3)
    engine.SPEC_PROBE = 10 ** 9  # never judged: the head is never rested
    seen = []

    def oracle(embed, aux, next_tokens, start, k, cache):  # drafts the tokens the target will produce
        seen.append((start, len(next_tokens), aux.shape))
        produced = start + len(next_tokens) - len(PROMPT) + 1  # tokens generated so far, the last of next_tokens too
        return ref[produced:produced + k], cache
    engine.draft_head.draft_many = lambda embed, work: [oracle(embed, *w) for w in work]
    assert _generate(engine, 20)[0] == ref
    assert engine.spec_accepted == engine.spec_drafted > 0
    assert seen[0] == (0, len(PROMPT), (len(PROMPT), 3 * 64))  # the prompt's positions, with the first token sampled
    for (a, n, _), (b, _, _) in zip(seen, seen[1:]):
        assert b == a + n  # contiguous: nothing skipped, nothing seen twice
    assert len(seen) <= 1 + 20 // 4 + 1  # four tokens per target forward: three drafts and the bonus token


def test_the_heads_chain_is_as_long_as_its_drafts_are_likely_to_be_reached():
    model = _target()
    ref = _generate(_engine(model), 60)[0]
    asked = []

    def head(right):  # drafts the target's own tokens where right(position in the chain), wrong ones elsewhere
        engine = _engine(model, _target_head(model), draft_tokens=6)
        engine.SPEC_PROBE = 10 ** 9  # never rested
        asked.clear()

        def draft(embed, aux, next_tokens, start, k, cache):
            asked.append(k)
            produced = start + len(next_tokens) - len(PROMPT) + 1
            good = ref[produced:produced + k]
            return [t if right(i) else (t + 1) % 128 for i, t in enumerate(good)], cache
        engine.draft_head.draft_many = lambda embed, work: [draft(embed, *w) for w in work]
        assert _generate(engine, 60)[0] == ref
        return list(asked)

    assert head(lambda i: False)[:4] == [6, 1, 1, 1]  # nothing accepted: one draft, to notice when that changes
    assert set(head(lambda i: True)[:-2]) == {6}  # everything accepted: the request's limit (the last steps: tokens left)
    # the first draft of every chain accepted, the second refused: half of the checked drafts are accepted, and the
    # chance of reaching a third draft, 0.5 ^ 3, is under HEAD_REACH
    assert set(head(lambda i: i == 0)[2:-2]) == {3}


def test_the_head_drafts_for_every_request_of_a_step_in_one_call():
    model = _target()
    prompts = (PROMPT, PROMPT[5:] + [44, 17, 9])
    refs = _generate(_engine(model), 24, prompts)
    engine = _engine(model, _target_head(model), draft_tokens=3)
    calls, many = [], engine.draft_head.draft_many
    engine.draft_head.draft_many = lambda embed, work: calls.append(len(work)) or many(embed, work)
    assert _generate(engine, 24, prompts) == refs
    assert max(calls) == 2 and engine.spec_drafted > 0  # both requests' verified positions in one pass of the head


def test_draft_head_needs_ascending_layers_inside_the_model():
    model = _target()
    head = _target_head(model)
    head.aux_layers = (2, 2, 1)
    with pytest.raises(ValueError, match="distinct, ascending"):
        _engine(model, head)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="packed prefill and the decode kernels need CUDA")
def test_draft_head_on_cuda_with_concurrent_requests_keeps_greedy_output():
    model = _target("cuda")
    prompts = (PROMPT, PROMPT[3:] + [44, 45])
    ref = _generate(_engine(model, device="cuda"), 16, prompts)
    engine = _engine(model, _target_head(model), device="cuda", draft_tokens=3)
    assert _generate(engine, 16, prompts) == ref
    assert engine.spec_drafted > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a verify step is launched ahead on CUDA only")
def test_the_verify_step_is_launched_behind_the_chain_and_gives_the_same_tokens():
    """With a draft head on CUDA the step that verifies a chain's drafts is queued before the host has read them
    (_verify_ahead), and while several requests draft the whole next cycle is queued behind it from the accept
    counts on the device (_chain_ahead): the same tokens as plain greedy decoding for requests that end one after
    the other (a step in flight when one leaves), and a launch does not wait for the device (but for the ones that
    capture a graph)."""
    model = _target("cuda")
    prompts, lens = (PROMPT, PROMPT[3:] + [44, 45], PROMPT[5:]), (48, 21, 33)

    def run(engine):
        async def collect(r):
            return [t async for t in engine.stream_request(r)]

        async def go():
            await engine.start()
            try:
                reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=n, temperature=0.0,
                                                 stop_token_ids=set()) for p, n in zip(prompts, lens)]
                return await asyncio.gather(*[collect(r) for r in reqs])
            finally:
                await engine.stop()
        return asyncio.run(go())
    ref = run(_engine(model, device="cuda"))
    engine = _engine(model, _target_head(model), device="cuda", draft_tokens=3)
    engine.SPEC_PROBE = 10 ** 9  # never judged: the head is never rested
    launch, waits = engine._verify_ahead, []

    def counted(reqs):
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                launched = launch(reqs)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        if launched:
            waits.append(len(seen))
        return launched
    engine._verify_ahead = counted
    chain, chained = engine._chain_ahead, []

    def chain_counted():
        torch.cuda.set_sync_debug_mode("warn")
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                chain()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        if engine._chained is not None:
            chained.append(len(seen))
    engine._chain_ahead = chain_counted
    assert run(engine) == ref
    _all_pages_back(engine)
    assert engine.spec_drafted > 0
    both = sorted(waits + chained)
    assert len(both) > 20 and both[len(both) // 2] == 0, (waits, chained)
    assert len(chained) > 5 and sorted(chained)[len(chained) // 2] == 0, chained  # cycles launched a cycle ahead


def _serve(engine, prompts, lens, late=0):
    """Each request's tokens; the last `late` prompts are submitted once a verify step is in flight."""
    async def collect(r):
        return [t async for t in engine.stream_request(r)]

    async def add(p, n):
        r = await engine.add_request(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set())
        return asyncio.ensure_future(collect(r))

    async def go():
        await engine.start()
        try:
            first = len(prompts) - late
            tasks = [await add(p, n) for p, n in zip(prompts[:first], lens)]
            while late and engine._verifying is None and not all(t.done() for t in tasks):
                await asyncio.sleep(0)  # the scheduler's turn
            assert not late or engine._verifying is not None, "no verify step was launched ahead"
            tasks += [await add(p, n) for p, n in zip(prompts[first:], lens[first:])]
            return await asyncio.gather(*tasks)
        finally:
            await engine.stop()
    return asyncio.run(go())


def _all_pages_back(engine):
    """No step in flight, and each pool has its pages again (but the scratch page of its padded decode steps)."""
    assert engine._verifying is None and engine._chained is None
    for pool in (engine.kv_pool, engine.draft_head.pool):
        assert pool.used_pages == (getattr(pool, "_scratch", None) is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a verify step is launched ahead on CUDA only")
def test_a_request_cancelled_while_its_verify_step_is_in_flight_costs_the_others_nothing():
    """The step in flight still writes the cancelled request's rows: its pages are free by then, and the other
    requests of that step get the tokens of plain greedy decoding."""
    model = _target("cuda")
    prompts, lens = (PROMPT, PROMPT[3:] + [44, 45], PROMPT[5:]), (48, 40, 33)
    ref = _serve(_engine(model, device="cuda"), prompts, lens)
    engine = _engine(model, _target_head(model), device="cuda", draft_tokens=3)
    engine.SPEC_PROBE = 10 ** 9  # never judged: the head is never rested
    launch, launched = engine._verify_ahead, []

    def cancelling(reqs):
        launched.append(launch(reqs))
        if launched[-1] and sum(launched) == 3:
            engine.cancel(reqs[0])  # seen at the next step, whose first act is to read this one
        return launched[-1]
    engine._verify_ahead = cancelling
    chain, chained = engine._chain_ahead, []
    engine._chain_ahead = lambda: chained.append(chain() or engine._chained is not None)  # cycles queued a cycle ahead
    out = _serve(engine, prompts, lens)
    cut = [i for i in range(3) if out[i] != ref[i]]
    assert len(cut) == 1 and len(out[cut[0]]) < lens[cut[0]] and out[cut[0]] == ref[cut[0]][:len(out[cut[0]])]
    assert sum(launched) + sum(chained) > 10  # the two others went on with steps launched ahead
    _all_pages_back(engine)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="a verify step is launched ahead on CUDA only")
def test_a_request_admitted_while_a_verify_step_is_in_flight_joins_without_changing_a_token():
    model = _target("cuda")
    prompts, lens = (PROMPT, PROMPT[3:] + [44, 45], PROMPT[5:]), (48, 40, 33)
    ref = _serve(_engine(model, device="cuda"), prompts, lens)
    engine = _engine(model, _target_head(model), device="cuda", draft_tokens=3)
    engine.SPEC_PROBE = 10 ** 9
    assert _serve(engine, prompts, lens, late=1) == ref
    assert engine.spec_drafted > 0
    _all_pages_back(engine)


@pytest.mark.parametrize("overlap", [False, True])
def test_drafts_made_before_a_stretch_of_plain_decoding_are_not_verified_after_it(overlap):
    """A head that is always right for the position it drafted at: a rejected draft can only be a stale one."""
    model = _target()
    ref = _generate(_engine(model), 60)[0]
    engine = _engine(model, _target_head(model), draft_tokens=3)
    engine.overlap_decode = overlap  # plain steps run ahead of the host there (the rule holds drafting for a few)

    def oracle(embed, aux, next_tokens, start, k, cache):
        produced = start + len(next_tokens) - len(PROMPT) + 1
        return ref[produced:produced + k], cache
    engine.draft_head.draft_many = lambda embed, work: [oracle(embed, *w) for w in work]
    assert _generate(engine, 60)[0] == ref
    assert engine.spec_accepted == engine.spec_drafted > 0
