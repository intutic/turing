"""Paged prefill attention kernel (speed plan P5.1) against the gather + SDPA path it replaces (CUDA)."""
import pytest
import torch
import torch.nn.functional as F

from turing.core.kv_cache import KVCache, KVPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton prefill kernel")


def _reference(pool, layer, cache, q, start, L, scale, window, softcap):
    """The old path: read (dequantize) the whole cache, repeat KV heads, masked attention in fp32."""
    H, Hkv = q.shape[1], pool.num_kv_heads
    k, v = pool.read(layer, cache, start + L)
    k, v = (x.float().repeat_interleave(H // Hkv, dim=1) for x in (k, v))
    s = q.float() @ k.transpose(2, 3) * scale
    if softcap:
        s = torch.tanh(s / softcap) * softcap
    qpos = start + torch.arange(L, device=q.device)[:, None]
    kpos = torch.arange(start + L, device=q.device)
    see = kpos <= qpos
    if window:
        see &= kpos > qpos - window
    s = s.masked_fill(~see, float("-inf"))
    out = torch.softmax(s, -1) @ v
    return out.transpose(1, 2).reshape(1, L, H * q.shape[-1])


@pytest.mark.parametrize("kv_cache_dtype,native", [("native", False), ("int8", False), ("int4", False), ("int4", True),
                                                   ("fp8", False)])
@pytest.mark.parametrize("H,Hkv,D", [(12, 2, 128), (8, 8, 64)])
@pytest.mark.parametrize("start,L", [(0, 50), (37, 70), (160, 16)])  # off page boundaries; a cached prefix
@pytest.mark.parametrize("window,softcap", [(0, 0.0), (24, 0.0), (0, 30.0)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_paged_prefill_matches_gather_and_sdpa(kv_cache_dtype, native, H, Hkv, D, start, L, window, softcap, dtype):
    from turing.kernels.triton_kv_prefill import triton_pool_prefill
    g = torch.Generator(device="cuda").manual_seed(0)
    pool = KVPool(1, Hkv, D, 64, dtype, "cuda", kv_cache_dtype, native_layers=(0,) if native else ())
    cache = KVCache(pool)
    cache.reserve(start + L)
    if start:
        pool.append(0, cache, torch.randn(1, Hkv, start, D, device="cuda", generator=g).to(dtype),
                    torch.randn(1, Hkv, start, D, device="cuda", generator=g).to(dtype))
        cache.length = start
    pool.append(0, cache, torch.randn(1, Hkv, L, D, device="cuda", generator=g).to(dtype),
                torch.randn(1, Hkv, L, D, device="cuda", generator=g).to(dtype))
    q = torch.randn(1, H, L, D, device="cuda", generator=g).to(dtype)
    scale = D ** -0.5
    out = triton_pool_prefill(pool, 0, q, cache.slot, start, scale, window, softcap)
    ref = _reference(pool, 0, cache, q, start, L, scale, window, softcap)
    assert out.dtype == dtype and out.shape == (1, L, H * D)
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=0 if dtype == torch.float32 else 1e-2)


@pytest.mark.parametrize("arch", ["llama", "qwen2_qkv_bias", "gemma2", "gpt2"])
@torch.no_grad()
def test_packed_prefill_matches_one_request_at_a_time(arch):
    """P4.3: chunks of three requests (one continuing a cached prefix) in one packed forward give each request the
    logits and the KV cache it gets when prefilled alone."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), device="cuda")
    assert model.can_pack_prefill([KVCache(model.kv_pool())])
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(0, 128, (n,), generator=g).tolist() for n in (21, 7, 40)]
    solo, caches = [], []
    for i, p in enumerate(prompts):
        c = KVCache(model.kv_pool())
        if i == 2:  # a cached prefix: the packed chunk continues it
            model(torch.tensor([p[:19]], device="cuda"), past_key_values=c)
        caches.append(c)
        ref = KVCache(model.kv_pool())
        logits, _ = model(torch.tensor([p], device="cuda"), past_key_values=ref, logits_to_keep=1)
        solo.append((logits[0, -1], ref))
    chunks = [prompts[0], prompts[1], prompts[2][19:]]
    out = model.prefill_packed(chunks, caches)
    for i, (c, (ref_logits, ref)) in enumerate(zip(caches, solo)):
        assert c.length == ref.length == len(prompts[i])
        torch.testing.assert_close(out[i].float(), ref_logits.float(), atol=1e-3, rtol=1e-3)
        k, v = model.kv_pool().read(0, c, c.length)
        kr, vr = model.kv_pool().read(0, ref, ref.length)
        torch.testing.assert_close(k, kr, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("storage", ["native", "int8"])
@pytest.mark.parametrize("arch", ["llama", "qwen2_qkv_bias", "gemma2"])
@torch.no_grad()
def test_decode_rows_ride_in_a_packed_prefill(arch, storage):
    """prefill_packed's riders (the engine's decoding requests in a step that prefills): each row gets the logits and
    the cache that a decode step of its own gives it, and the prompt chunks get theirs."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), device="cuda")
    model.config.kv_cache_dtype = storage
    pool = model.kv_pool()
    tol = dict(atol=1e-3, rtol=1e-3) if storage == "native" else dict(atol=2e-2, rtol=2e-2)  # a code may round apart
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 128, (n,), generator=g).tolist() for n in (40, 9)]
    rows = [torch.randint(0, 128, (n,), generator=g).tolist() for n in (30, 17, 23)]  # cached tokens, then the input
    refs, caches = [], []
    for r in rows:
        own, rider = KVCache(pool), KVCache(pool)
        for c in (own, rider):
            model(torch.tensor([r[:-1]], device="cuda"), past_key_values=c)
        logits, _ = model(torch.tensor([r[-1:]], device="cuda"), past_key_values=own)
        refs.append((logits[0, -1], own))
        caches.append(rider)
    for p in prompts:
        own = KVCache(pool)
        logits, _ = model(torch.tensor([p], device="cuda"), past_key_values=own, logits_to_keep=1)
        refs.append((logits[0, -1], own))
    new = [KVCache(pool) for _ in prompts]
    assert model.can_ride_prefill(new + caches)
    from turing.core.rope import _rope_table
    tables = _rope_table.cache_info().currsize
    out = model.prefill_packed(prompts + [r[-1:] for r in rows], new + caches, riders=len(rows))
    # no RoPE table per sequence length: riders reach a new length every step, and the tables are cached
    assert _rope_table.cache_info().currsize == tables
    for i, (c, (ref_logits, own)) in enumerate(zip(caches + new, refs)):
        assert c.length == own.length
        torch.testing.assert_close(out[(i + len(prompts)) % len(out)].float(), ref_logits.float(), **tol)
        for layer in (0, pool.num_layers - 1):
            for x, y in zip(pool.read(layer, c, c.length), pool.read(layer, own, own.length)):
                torch.testing.assert_close(x.float(), y.float(), **tol)


def test_no_rides_with_page_selection():
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    cfg = model.config
    pool = KVPool(cfg.num_layers, cfg.num_kv_heads, cfg.head_dim, 8, torch.float16, "cuda", track_pages=True)
    assert model.can_pack_prefill([KVCache(pool)]) and not model.can_ride_prefill([KVCache(pool)])


@pytest.mark.parametrize("draft", [0, 4])
def test_decoding_requests_ride_in_the_prefill_forwards_of_a_burst(draft):
    """While a long prompt is prefilled chunk by chunk, the requests already decoding take their next tokens from the
    same forward and no decode forward runs; with drafts (right ones for the first request, wrong ones for the
    second) each draft is a row of that forward. Their tokens are those of plain decoding."""
    from tests.test_engine_generation import PROMPT, _run
    from tests.test_hf_parity import _tiny
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    long = torch.randint(0, 128, (120,), generator=torch.Generator().manual_seed(7)).tolist()
    a, b = len(PROMPT), len(PROMPT) - 3  # 38 and 35 tokens: the second prompt does not fit the first step's 64

    def run(ride, k, oracle=None):
        model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
        model.config.kv_pool_gib = 0.01
        engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                       prefix_caching=False, draft_tokens=k, prefill_chunk_size=64)
        engine.ride_prefill = ride
        if oracle is not None:
            engine._lookup = lambda req, n: (oracle[len(req.generated_tokens):][:n] if len(req.prompt_tokens) == a
                                             else [1, 2, 3, 4][:n] if len(req.prompt_tokens) == b else None)
        packed, pack, decodes = [], model.prefill_packed, []
        model.prefill_packed = lambda chunks, *x, **kw: packed.append([len(c) for c in chunks]) or pack(chunks, *x, **kw)
        for name in ("_launch", "_decode_rows"):  # every decode forward of these requests
            def spy(*x, _f=getattr(engine, name)):
                decodes.append(len(packed))
                return _f(*x)
            setattr(engine, name, spy)
        reqs = [dict(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set())
                for p, n in ((PROMPT, 30), (PROMPT[3:], 30), (long, 6))]
        _, outs = _run(engine, *reqs)
        pool = engine.kv_pool  # all pages are back but the one of the decode graphs' padding rows
        assert [len(o) for o in outs] == [30, 30, 6] and pool.used_pages == (getattr(pool, "_scratch", None) is not None)
        return outs, packed, decodes, engine

    ref, packed, decodes, _ = run(False, 0)
    # a prompt that fits a step starts the next one whole; the long one takes what is left of each step's 64
    assert packed == [[b, 64 - b]] and decodes[0] == 1  # a decode forward from step 2 on
    outs, packed, decodes, engine = run(True, draft, ref[0] if draft else None)
    r1, r2 = (1 + draft, 2 * (1 + draft))  # rows of the first request alone, and of both
    # the first prompt alone is a packed forward too: a step launched ahead (_ride_ahead)
    assert packed == [[a], [b, 64 - b] + [1] * r1, [64] + [1] * r2, [120 - (64 - b) - 64] + [1] * r2]
    assert decodes[0] == 4  # none until the prompts are in
    assert outs == ref
    assert engine.spec_accepted >= 3 * draft and engine.spec_drafted > engine.spec_accepted or not draft


@pytest.mark.parametrize("storage", ["native", "fp8", "int8"])
@torch.no_grad()
def test_decode_rows_through_flashinfer(storage, monkeypatch):
    """Where FlashInfer reads a layer's pages in place (model dtype, fp8), the decode rows of a packed prefill go
    through its decode wrapper in a call of their own; an int8 layer's rows keep the decode kernel. Either way the
    logits and caches are those of the decode kernel's path (a cache listed twice: a draft row)."""
    from turing.config import ModelConfig
    from turing.kernels import flashinfer_prefill as fi
    from turing.models import causal_lm
    if not fi.HAS_FLASHINFER:
        pytest.skip("flashinfer is not installed")
    torch.manual_seed(0)
    cfg = ModelConfig(name="riders", vocab_size=128, hidden_dim=256, ffn_dim=512, num_layers=3, num_heads=4,
                      num_kv_heads=2, head_dim=64)  # FlashInfer serves heads of 64 dims and more
    cfg.kv_cache_dtype, cfg.cuda_graphs, cfg.kv_pool_gib = storage, False, 0.05
    model = causal_lm.SubspaceCausalLM(cfg).eval().to("cuda", torch.float16)
    pool = model.kv_pool()
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 128, (n,), generator=g).tolist() for n in (540, 40)]
    rows = [torch.randint(0, 128, (n,), generator=g).tolist() for n in (30, 17, 23)]

    def step(through_flashinfer):
        if not through_flashinfer:  # the rows' own check fails: they take the decode kernel, the chunks as before
            real_ok = causal_lm.flashinfer_prefill_supported
            monkeypatch.setattr(causal_lm, "flashinfer_prefill_supported",
                                lambda *a, **kw: "min_chunk" not in kw and real_ok(*a, **kw))
        calls = {"flashinfer": 0, "decode": 0}
        for name, key in (("flashinfer_prefill", "flashinfer"), ("flashinfer_rows", "flashinfer"),
                          ("triton_pool_decode", "decode")):
            real = getattr(causal_lm, name)
            monkeypatch.setattr(causal_lm, name, lambda *a, _r=real, _k=key, **kw: calls.__setitem__(_k, calls[_k] + 1) or _r(*a, **kw))
        riders = []
        for r in rows:
            c = KVCache(pool)
            model(torch.tensor([r[:-1]], device="cuda"), past_key_values=c)
            riders.append(c)
        riders.append(riders[0])  # a draft after the first row's token
        new = [KVCache(pool) for _ in prompts]
        out = model.prefill_packed(prompts + [r[-1:] for r in rows] + [[9]], new + riders, riders=len(riders))
        kv = [[x.float().clone() for layer in (0, pool.num_layers - 1) for x in pool.read(layer, c, c.length)]
              for c in new + riders[:3]]  # the first row's cache holds its token and the draft
        monkeypatch.undo()
        return out.float(), kv, calls

    out, kv, calls = step(True)
    ref, ref_kv, ref_calls = step(False)
    in_place = [pool.layer_fields(i, "k")[3] in ("native", "fp8") for i in range(pool.num_layers)]
    assert calls == {"flashinfer": pool.num_layers + in_place.count(True), "decode": in_place.count(False)}
    assert ref_calls == {"flashinfer": pool.num_layers, "decode": pool.num_layers}
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    for a, b in zip(kv, ref_kv):
        for x, y in zip(a, b):  # a stored code may round apart where the two paths' states differ in the last bit
            apart = (x - y).abs() > 2e-2 + 2e-2 * y.abs()
            assert apart.float().mean() < 0.01 and ((x - y).abs() <= 0.13 * y.abs().clamp(min=1.0))[apart].all()


@pytest.mark.parametrize("plan,graphs", [((64, False), True), ((64, True), False), ((64, True), True),
                                         ((56, False), False)])
def test_a_forward_keeps_to_the_rows_the_model_measured(plan, graphs):
    """The engine cuts a forward to the row count the model's forward_rows gives (here FORWARD_TOKENS is 64): a
    forward that cuts or continues a prompt takes `rows` in all, decode rows included, where that is under
    FORWARD_TOKENS or rows past it cost more (cliff); the cut tokens go first in the next forward, and the last
    forward of a burst runs as it is. The tokens are those of plain decoding."""
    from tests.test_engine_generation import PROMPT, _run
    from tests.test_hf_parity import _tiny
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    long = torch.randint(0, 128, (120,), generator=torch.Generator().manual_seed(7)).tolist()
    b = len(PROMPT) - 3  # 35 tokens; PROMPT's 38 go first, alone

    def run(ride):
        model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
        model.config.kv_pool_gib, model.config.cuda_graphs = 0.01, graphs
        model.forward_rows = lambda: plan
        engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                       prefix_caching=False, draft_tokens=0, prefill_chunk_size=64)
        engine.ride_prefill, engine.FORWARD_TOKENS = ride, 64
        packed, pack = [], model.prefill_packed
        model.prefill_packed = lambda chunks, *x, **kw: packed.append([len(c) for c in chunks]) or pack(chunks, *x, **kw)
        reqs = [dict(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set())
                for p, n in ((PROMPT, 30), (PROMPT[3:], 30), (long, 6))]
        _, outs = _run(engine, *reqs)
        return outs, packed

    ref, _ = run(False)
    outs, packed = run(True)
    # the second forward cuts the long prompt, the third continues it, the last ends the burst
    assert packed[0] == [len(PROMPT)]  # the first prompt, alone
    assert packed[1:] == {(64, False): [[b, 29, 1], [64, 1, 1], [27, 1, 1]],  # 65 and 66 rows: nothing to gain
                          (64, True): [[b, 28, 1], [62, 1, 1], [30, 1, 1]],
                          (56, False): [[b, 20, 1], [54, 1, 1], [46, 1, 1]]}[plan]
    assert outs == ref


def test_a_long_prompts_forward_of_a_larger_budget_holds_the_decode_rows_within_the_budget():
    """A prompt longer than a forward takes a larger step budget in one forward (here FORWARD_TOKENS is 32 and the
    budget 96). The decode rows riding in it count in the budget's rows, as in a step of vLLM's: the tokens they
    displace go first in the next forward, and the forward that ends the burst runs as it is."""
    from tests.test_engine_generation import _run
    from tests.test_hf_parity import _tiny
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    g = torch.Generator().manual_seed(11)
    short, long = (torch.randint(0, 128, (n,), generator=g).tolist() for n in (20, 250))

    def run(ride):
        model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
        model.config.kv_pool_gib, model.config.cuda_graphs = 0.01, False
        model.forward_rows = lambda: (32, False)
        engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                       prefix_caching=False, draft_tokens=0, prefill_chunk_size=96)
        engine.ride_prefill, engine.FORWARD_TOKENS = ride, 32
        packed, pack = [], model.prefill_packed
        model.prefill_packed = lambda chunks, *x, **kw: packed.append([len(c) for c in chunks]) or pack(chunks, *x, **kw)
        _, outs = _run(engine, dict(prompt_tokens=short, max_new_tokens=30, temperature=0.0, stop_token_ids=set()),
                       dict(prompt_tokens=long, max_new_tokens=6, temperature=0.0, stop_token_ids=set()))
        return outs, packed

    ref, _ = run(False)
    outs, packed = run(True)
    # the short prompt and what is left of its forward; then the long one: the rest of that budget (64), a whole
    # budget, and the end of the burst, each with the short request's decode row
    assert packed == [[20, 12], [63, 1], [95, 1], [80, 1]]
    assert outs == ref


def test_whole_prompts_stay_whole_unless_rows_past_the_forward_cost_more():
    """A forward of whole new prompts with decode rows riding: the rows come on top, or, on a model with a cliff
    past FORWARD_TOKENS rows, take the places of the last prompt's last tokens when another forward follows."""
    from tests.test_engine_generation import _run
    from tests.test_hf_parity import _tiny
    from turing.config import TuringConfig
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.serving.engine import ContinuousBatchEngine
    g = torch.Generator().manual_seed(3)
    prompts = [torch.randint(0, 128, (32,), generator=g).tolist() for _ in range(6)]

    def run(plan):
        model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
        model.config.kv_pool_gib, model.config.cuda_graphs = 0.01, False
        model.forward_rows = lambda: plan
        engine = ContinuousBatchEngine(model.config, TuringConfig(device="cuda", max_batch_size=8), model=model,
                                       prefix_caching=False, draft_tokens=0, prefill_chunk_size=64)
        engine.FORWARD_TOKENS = 64
        packed, pack = [], model.prefill_packed
        model.prefill_packed = lambda chunks, *x, **kw: packed.append([len(c) for c in chunks]) or pack(chunks, *x, **kw)
        _, outs = _run(engine, *[dict(prompt_tokens=p, max_new_tokens=8, temperature=0.0, stop_token_ids=set())
                                 for p in prompts])
        return outs, packed

    ref, packed = run((64, False))
    assert packed == [[32, 32], [32, 32, 1, 1], [32, 32, 1, 1, 1, 1]]
    outs, packed = run((64, True))
    # two rows over: the fourth prompt leaves 2 tokens; then a forward that continues it, cut to 64 rows; the last
    # forward ends the burst and runs whole
    assert packed == [[32, 32], [32, 30, 1, 1], [2, 32, 27, 1, 1, 1], [5, 1, 1, 1, 1, 1]]
    assert outs == ref


@pytest.mark.parametrize("cached", [False, True])
def test_forward_rows_are_measured_once_and_kept(cached, tmp_path, monkeypatch):
    """calibrate_forward_rows times forwards of 2,040, 2,048 and 2,056 tokens, gives the pool its pages back and
    keeps the plan with the KV calibration; a second model of the same weights reads it."""
    import json
    from turing.config import ModelConfig
    from turing.models.causal_lm import SubspaceCausalLM
    monkeypatch.setenv("TURING_CACHE_DIR", str(tmp_path))

    def model():
        torch.manual_seed(0)
        cfg = ModelConfig(name="rows", vocab_size=256, hidden_dim=1024, ffn_dim=1024, num_layers=2, num_heads=16,
                          num_kv_heads=4, head_dim=64)
        cfg.kv_pool_gib = 0.05
        return SubspaceCausalLM(cfg).eval().to("cuda", torch.float16)

    first = model()
    bodies, body = [], first._body
    first._body = lambda ids, *a, **kw: bodies.append(ids.shape[1]) or body(ids, *a, **kw)
    pool = first.kv_pool()
    rows, cliff = first.forward_rows()
    assert rows in (2040, 2048) and isinstance(cliff, bool) and sorted(bodies) == sorted([2040, 2048, 2056] * 9)
    assert pool.used_pages == 0
    (path,) = tmp_path.glob("kv_calibration/*.json")
    (plan,) = json.loads(path.read_text())["forward_rows"].values()
    assert plan[:2] == [rows, cliff] and len(plan) == 5
    if cached:
        plan[:2] = [2040, True]
        rec = json.loads(path.read_text())
        rec["forward_rows"] = {k: plan for k in rec["forward_rows"]}
        path.write_text(json.dumps(rec))
        second = model()
        second._body = lambda *a, **kw: pytest.fail("measured again")
        second.kv_pool()
        assert second.forward_rows() == (2040, True)


@pytest.mark.parametrize("storage", ["int8", "int4", "fp8"])
@pytest.mark.parametrize("Hkv,D", [(2, 128), (8, 64)])
def test_chunk_append_kernel_stores_what_the_reference_stores(storage, Hkv, D):
    """triton_pool_append (a prefill chunk into a low-bit layer) against KVPool.append: the same codes, scales and
    zeros, from a start off a page boundary, with strided inputs."""
    from turing.kernels.triton_kv_decode import pool_append_supported, triton_pool_append
    g = torch.Generator(device="cuda").manual_seed(2)
    pools = [KVPool(2, Hkv, D, 64, torch.float16, "cuda", storage, native_layers=(1,)) for _ in range(2)]
    assert pool_append_supported(pools[0], 0) and not pool_append_supported(pools[0], 1)  # layer 1 is kept native
    assert not pool_append_supported(KVPool(1, Hkv, D, 8, torch.float16, "cuda", storage, track_pages=True), 0)
    wide_k, wide_v = (torch.randn(1, Hkv, 37 + 300, 2 * D, device="cuda", generator=g).half() for _ in range(2))
    k, v = wide_k[..., ::2], wide_v[..., ::2]  # not contiguous in the last dimension
    caches = [KVCache(p) for p in pools]
    for append, pool, cache in ((triton_pool_append, pools[0], caches[0]),
                                (lambda p, layer, c, a, b: p.append(layer, c, a, b), pools[1], caches[1])):
        cache.reserve(337)
        pool.append(0, cache, k[:, :, :37], v[:, :, :37])
        cache.length = 37
        append(pool, 0, cache, k[:, :, 37:], v[:, :, 37:])
        cache.length = 337
    for which in ("k", "v"):
        for a, b in zip(pools[0].layer_fields(0, which)[:3], pools[1].layer_fields(0, which)[:3]):
            if a is None or a.dim() == 1:  # native storage has no scales; fp8's is the layer's, not stored per token
                continue
            pa, pb = (t[p.page_table[c.slot, :22].long()].transpose(1, 2).flatten(0, 1)[:337]  # the written tokens
                      for t, p, c in ((a, pools[0], caches[0]), (b, pools[1], caches[1])))
            if a.dtype == torch.uint8:
                assert torch.equal(pa, pb)
            else:
                torch.testing.assert_close(pa.float(), pb.float(), atol=0, rtol=1e-3)


@pytest.mark.parametrize("arch", ["llama", "qwen2_qkv_bias", "gpt2", "mixtral"])
@torch.no_grad()
def test_prefill_graph_matches_the_eager_forward(arch):
    """PrefillGraphs: a prompt with no cached prefix replayed from a CUDA graph of its length bucket (padded to it)
    gives the logits and the cache of the eager forward, and decoding goes on from it as from the eager cache."""
    from tests.test_hf_parity import ARCHS, _tiny
    from turing.models.causal_lm import PrefillGraphs
    from turing.models.hf_loader import RealHuggingFaceLoader
    if arch not in ARCHS:
        pytest.skip(f"no tiny {arch} model in the parity set")
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), device="cuda")
    model.config.kv_pool_gib = 0.02
    pool, graphs = model.kv_pool(), model._prefill_graphs
    g = torch.Generator().manual_seed(11)
    for n in (40, 33, 64, 97):  # 40 and 33 share the 64-token bucket with 64; 97 pads to 128
        ids = torch.randint(0, 128, (1, n + 3), generator=g).cuda()
        runs = []
        for use in (False, True):
            model.config.cuda_graphs = use
            graphs.seen[(1, PrefillGraphs.bucket(n), 0)] = 1  # as if the bucket had been used once: capture now
            cache = KVCache(pool)
            out = [model(ids[:, :n], past_key_values=cache, logits_to_keep=1)[0][0, -1]]
            assert cache.length == n
            for t in range(n, n + 3):  # decode steps overwrite the padding's keys and values
                out.append(model(ids[:, t:t + 1], past_key_values=cache)[0][0, -1])
            runs.append((torch.stack(out), [x.clone() for x in pool.read(0, cache, cache.length)], cache))
        (eager, kv_e, _), (graph, kv_g, _) = runs
        assert (1, PrefillGraphs.bucket(n), 0) in graphs.graphs
        torch.testing.assert_close(graph.float(), eager.float(), atol=2e-3, rtol=2e-3)
        for a, b in zip(kv_g, kv_e):
            torch.testing.assert_close(a.float(), b.float(), atol=2e-3, rtol=2e-3)
        for _, _, c in runs:
            c.free()
    assert sorted(graphs.graphs) == [(1, 64, 0), (1, 128, 0)]  # one graph per bucket, whatever the prompts' lengths
    # a graph reads its token-position tensors on every replay: had they been freed after the capture, these
    # allocations would now sit in their blocks
    junk = [torch.full((s,), 1 << 20, dtype=t, device="cuda") for s in (64, 128) for t in (torch.int32, torch.int64)
            for _ in range(32)]
    for n in (40, 97):
        ids = torch.randint(0, 128, (1, n), generator=g).cuda()
        outs = []
        for use in (False, True):
            model.config.cuda_graphs = use
            cache = KVCache(pool)
            outs.append(model(ids, past_key_values=cache, logits_to_keep=1)[0][0, -1].float())
            cache.free()
        torch.testing.assert_close(outs[1], outs[0], atol=2e-3, rtol=2e-3)
    del junk


@pytest.mark.parametrize("arch,storage", [("llama", "native"), ("qwen2_qkv_bias", "native"), ("gpt2", "native"),
                                          ("mixtral", "native"), ("llama", "int8"), ("qwen2_qkv_bias", "fp8")])
@torch.no_grad()
def test_a_step_of_several_prompts_and_decode_rows_replays_from_one_graph(arch, storage):
    """PrefillGraphs on a burst step: prompts with no cached prefix (padded to the longest one's bucket) and the
    decode rows of requests already generating (padded to a decode bucket, one of them listed twice as with a draft)
    give each request the logits and the cache of the eager packed forward, and decoding goes on from them. With an
    int8 or fp8 pool the prompts attend over what their pages give back, as the eager forward reads them."""
    from tests.test_hf_parity import ARCHS, _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    if arch not in ARCHS:
        pytest.skip(f"no tiny {arch} model in the parity set")
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), device="cuda")
    model.config.kv_pool_gib, model.config.kv_cache_dtype = 0.05, storage
    tol = dict(atol=2e-3, rtol=2e-3) if storage == "native" else dict(atol=2e-2, rtol=2e-2)  # a code may round apart
    kv_tol = dict(atol=2e-2, rtol=0.13) if storage == "fp8" else tol  # by one E4M3 step: an eighth of the value
    pool, graphs = model.kv_pool(), model._prefill_graphs
    g = torch.Generator().manual_seed(13)
    rand = lambda n: torch.randint(0, 128, (n,), generator=g).tolist()  # noqa: E731

    def step(lens, cached, use):
        """One packed step: prompts of `lens` tokens, then a decode row for each of `cached` (prompt tokens already
        in a cache), the last request with a second row (a draft)."""
        model.config.cuda_graphs = False
        riders = []
        for n in cached:
            c = KVCache(pool)
            model(torch.tensor([rand(n)], device="cuda"), past_key_values=c)
            riders.append(c)
        riders += riders[-1:]
        model.config.cuda_graphs = use
        new = [KVCache(pool) for _ in lens]
        out = [model.prefill_packed([rand(n) for n in lens] + [rand(1) for _ in riders], new + riders, len(riders))]
        for c in riders[-1:]:
            c.length = cached[-1] + 2
        assert [c.length for c in new] == list(lens)
        assert [len(c.pages) for c in new] == [-(-n // 16) for n in lens]  # the padding's pages are given back
        for c in new[:1] + riders[:1]:  # decode steps overwrite the padding's keys and values
            for _ in range(3):
                out.append(model(torch.tensor([rand(1)], device="cuda"), past_key_values=c)[0][0, -1:])
        kv = [x.clone() for c in new + riders[:-1] for x in pool.read(pool.num_layers - 1, c, c.length)]
        for c in new + riders[:-1]:
            c.free()
        return torch.cat(out), kv

    for lens, cached, key in (((40, 50, 61), (30, 17), (3, 64, 4)), ((90, 70), (), (2, 96, 0)), ((33,), (9,), (1, 64, 2))):
        runs = []
        for use in (False, True):
            state = g.get_state()
            graphs.seen[key] = 1  # as if the shape had been used once: capture now
            runs.append(step(lens, cached, use))
            if not use:
                g.set_state(state)  # the same tokens for the graph
        assert key in graphs.graphs
        (eager, kv_e), (graph, kv_g) = runs
        torch.testing.assert_close(graph.float(), eager.float(), **tol)
        for a, b in zip(kv_g, kv_e):
            torch.testing.assert_close(a.float(), b.float(), **kv_tol)
        again, _ = step(lens, cached, True)  # a replay, other tokens
        assert again.shape == eager.shape and torch.isfinite(again).all()
    # prompts of unlike lengths (most of the step would be padding) and a prompt continuing a cache stay eager
    model.config.cuda_graphs = True
    assert graphs.step_key(model, [KVCache(pool), KVCache(pool)], [20, 90], 0) is None
    c = KVCache(pool)
    model(torch.tensor([rand(5)], device="cuda"), past_key_values=c)
    assert graphs.step_key(model, [c, KVCache(pool)], [40, 40], 0) is None


def test_prefill_graph_waits_for_a_bucket_s_second_use_and_skips_what_it_cannot_serve():
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    ids = torch.randint(0, 128, (1, 50), generator=torch.Generator().manual_seed(1)).cuda()
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    model.config.kv_pool_gib = 0.02
    with torch.no_grad():
        for expect in (0, 1, 1):  # eager, capture, replay
            model(ids, past_key_values=KVCache(model.kv_pool()), logits_to_keep=1)
            assert len(model._prefill_graphs.graphs) == expect
        model(ids, past_key_values=KVCache(model.kv_pool()))  # every position's logits: eager
        cache = KVCache(model.kv_pool())
        model(ids[:, :20], past_key_values=cache, logits_to_keep=1)
        model(ids[:, 20:], past_key_values=cache, logits_to_keep=1)  # a cached prefix: eager
        assert sorted(model._prefill_graphs.graphs) == [(1, 64, 0)]
        assert model._prefill_graphs.seen == {(1, 64, 0): 3, (1, 32, 0): 1}
    for arch, kv in (("gemma2", "native"), ("llama", "int4")):  # a sliding window and soft-capping; an int4 pool
        other = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), device="cuda")
        other.config.kv_pool_gib, other.config.kv_cache_dtype = 0.02, kv
        with torch.no_grad():
            for _ in range(3):
                other(ids, past_key_values=KVCache(other.kv_pool()), logits_to_keep=1)
        assert not other._prefill_graphs.graphs


def test_a_prefill_graph_no_faster_than_the_eager_forward_is_dropped_with_every_longer_one():
    """PrefillGraphs times a bucket's eager forward and its first two replays: where the replay is no faster (the
    length is bound by the device on this model and GPU), that bucket and the longer ones go back to the eager path."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    g = torch.Generator().manual_seed(2)
    long, short = (torch.randint(0, 128, (1, n), generator=g).cuda() for n in (70, 40))  # buckets of 96 and 64
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    model.config.kv_pool_gib = 0.02
    graphs, pool = model._prefill_graphs, model.kv_pool()
    first = lambda ids: model(ids, past_key_values=KVCache(pool), logits_to_keep=1)[0][0, -1].float()  # noqa: E731
    with torch.no_grad():
        eager = first(long)  # timed
        torch.testing.assert_close(first(long), eager, atol=2e-3, rtol=2e-3)  # captured; the first timed replay
        key = (1, 96, 0)
        assert key in graphs.graphs and len(graphs.trial[key]) == 2
        graphs.trial[key][0] = 0.0  # as if the eager forward had taken no time: no replay beats that
        torch.testing.assert_close(first(long), eager, atol=2e-3, rtol=2e-3)  # the second replay: judged
        assert key not in graphs.graphs and key not in graphs.trial and graphs.limit == 95
        torch.testing.assert_close(first(long), eager, atol=2e-3, rtol=2e-3)  # eager from now on
        assert key not in graphs.graphs and graphs.seen[key] == 3  # not even counted any more
        for _ in range(3):
            first(short)
        # a shorter bucket is still judged on its own
        assert sorted(graphs.graphs) == [(1, 64, 0)] and (1, 64, 0) not in graphs.trial
        # a step of several prompts of the dropped length stays eager too; a shorter one whose replay is no faster
        # is dropped by itself
        pair = lambda n: model.prefill_packed(  # noqa: E731
            [torch.randint(0, 128, (n,), generator=g).tolist() for _ in range(2)], [KVCache(pool), KVCache(pool)])
        assert graphs.step_key(model, [KVCache(pool), KVCache(pool)], [70, 70], 0) is None
        pair(40), pair(40)
        graphs.trial[(2, 64, 0)][0] = 0.0
        pair(40)
        assert (2, 64, 0) in graphs.dead and sorted(graphs.graphs) == [(1, 64, 0)] and graphs.limit == 95
        assert graphs.step_key(model, [KVCache(pool), KVCache(pool)], [40, 40], 0) is None


def test_a_capture_after_every_graph_was_dropped_gets_a_new_pool():
    """All prefill graphs share one memory pool; once the last of them is dropped PyTorch's allocator refuses a
    capture into that pool, so the next capture starts another."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    g = torch.Generator().manual_seed(2)
    a, b = (torch.randint(0, 128, (1, n), generator=g).cuda() for n in (40, 20))  # buckets of 64 and 32
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cuda")
    model.config.kv_pool_gib = 0.02
    graphs, pool = model._prefill_graphs, model.kv_pool()
    first = lambda ids: model(ids, past_key_values=KVCache(pool), logits_to_keep=1)[0][0, -1].float()  # noqa: E731
    with torch.no_grad():
        first(a), first(a)
        handle = graphs.pool
        graphs.trial[(1, 64, 0)][0] = 0.0
        first(a)  # judged and dropped: no graph is left
        assert not graphs.graphs
        eager = first(b)
        torch.testing.assert_close(first(b), eager, atol=2e-3, rtol=2e-3)  # captured
        assert sorted(graphs.graphs) == [(1, 32, 0)] and graphs.pool != handle


def test_int4_layers_with_a_cached_prefix_take_the_gather_path_and_the_rest_the_kernel():
    from turing.kernels.triton_kv_prefill import pool_prefill_preferred
    q = torch.zeros(1, 4, 8, 64, device="cuda", dtype=torch.float16)
    int4 = KVPool(2, 2, 64, 8, torch.float16, "cuda", "int4", native_layers=(1,))
    assert pool_prefill_preferred(int4, 0, q, 0)  # no prefix: the kernel
    assert not pool_prefill_preferred(int4, 0, q, 16)  # int4 prefix: one gather beats per-block dequantization
    assert pool_prefill_preferred(int4, 1, q, 16)  # a layer kept native in an int4 pool
    assert pool_prefill_preferred(KVPool(2, 2, 64, 8, torch.float16, "cuda", "int8"), 0, q, 16)
