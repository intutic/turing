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


@pytest.mark.parametrize("kv_cache_dtype,native", [("native", False), ("int8", False), ("int4", False), ("int4", True)])
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
    a, b = len(PROMPT), len(PROMPT) - 3  # 38 and 35 tokens: the first step's 64 end inside the second prompt

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
    assert packed == [[a, 64 - a], [b - (64 - a), 128 - a - b]] and decodes[0] == 2  # a decode forward from step 2 on
    outs, packed, decodes, engine = run(True, draft, ref[0] if draft else None)
    r1, r2 = (1 + draft, 2 * (1 + draft))  # rows of the first request alone, and of both
    assert packed == [[a, 64 - a], [b - (64 - a), 128 - a - b] + [1] * r1, [64] + [1] * r2,
                      [120 - (128 - a - b) - 64] + [1] * r2]
    assert decodes[0] == 4  # none until the prompts are in
    assert outs == ref
    assert engine.spec_accepted >= 3 * draft and engine.spec_drafted > engine.spec_accepted or not draft


@pytest.mark.parametrize("storage", ["int8", "int4"])
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
            if a is None:  # native storage has no scales
                continue
            pa, pb = (t[p.page_table[c.slot, :22].long()].transpose(1, 2).flatten(0, 1)[:337]  # the written tokens
                      for t, p, c in ((a, pools[0], caches[0]), (b, pools[1], caches[1])))
            if a.dtype == torch.uint8:
                assert torch.equal(pa, pb)
            else:
                torch.testing.assert_close(pa.float(), pb.float(), atol=0, rtol=1e-3)


def test_int4_layers_with_a_cached_prefix_take_the_gather_path_and_the_rest_the_kernel():
    from turing.kernels.triton_kv_prefill import pool_prefill_preferred
    q = torch.zeros(1, 4, 8, 64, device="cuda", dtype=torch.float16)
    int4 = KVPool(2, 2, 64, 8, torch.float16, "cuda", "int4", native_layers=(1,))
    assert pool_prefill_preferred(int4, 0, q, 0)  # no prefix: the kernel
    assert not pool_prefill_preferred(int4, 0, q, 16)  # int4 prefix: one gather beats per-block dequantization
    assert pool_prefill_preferred(int4, 1, q, 16)  # a layer kept native in an int4 pool
    assert pool_prefill_preferred(KVPool(2, 2, 64, 8, torch.float16, "cuda", "int8"), 0, q, 16)
