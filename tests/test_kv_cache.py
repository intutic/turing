"""Paged KV pool: low-bit error bounds, pages and Quest bounds, copy-on-write forks, and equivalence of the paths."""
import pytest
import torch

from turing.core.kv_cache import PAGE, KVCache, KVPool, KVPoolExhausted, kv_bytes_per_token
from tests.test_hf_parity import _tiny_llama
from turing.models.hf_loader import RealHuggingFaceLoader


def _kv(T, H=2, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, H, T, D, generator=g)
    k[..., 3] *= 20  # an outlier channel, as real keys have
    return k, torch.randn(1, H, T, D, generator=g)


def _pool(kv_cache_dtype="native", pages=128, H=2, D=64, layers=1, track=False):
    return KVPool(layers, H, D, pages, torch.float32, "cpu", kv_cache_dtype, track)


def _fill(pool, k, v, chunk=None):
    """A cache holding k, v in layer 0 (appended in chunks of `chunk` tokens)."""
    c = KVCache(pool)
    T = k.shape[2]
    for s in range(0, T, chunk or T):
        e = min(T, s + (chunk or T))
        c.reserve(e)
        pool.append(0, c, k[:, :, s:e], v[:, :, s:e])
        c.length = e
    return c


@pytest.mark.parametrize("dtype,levels,group", [("int8", 255, 64), ("int4", 15, 32)])
def test_low_bit_error_is_within_half_a_step_of_each_group(dtype, levels, group):
    k, v = _kv(40)
    pool = _pool(dtype)
    kr, vr = pool.read(0, _fill(pool, k, v), 40)
    for x, xr in ((k, kr), (v, vr)):
        g = x.unflatten(-1, (-1, group))
        step = (g.amax(-1, keepdim=True) - g.amin(-1, keepdim=True)) / levels
        err = (x - xr).abs().unflatten(-1, (-1, group))
        assert (err <= step / 2 + 2e-3 * g.abs().amax(-1, keepdim=True)).all()  # + fp16 rounding of scale and zero


def test_uneven_chunks_keep_content_and_page_bounds():
    k, v = _kv(600)
    pool = _pool(track=True)
    c = _fill(pool, k, v, chunk=37)  # chunks cross page boundaries
    kr, vr = pool.read(0, c, 600)
    torch.testing.assert_close(kr, k, rtol=0, atol=0)
    torch.testing.assert_close(vr, v, rtol=0, atol=0)
    pages = -(-600 // PAGE)
    assert len(c.pages) == pages  # no capacity margin: only the last page is partly used
    kp = torch.nn.functional.pad(k, (0, 0, 0, pages * PAGE - 600), value=float("nan")).unflatten(2, (pages, PAGE))
    phys = torch.tensor(c.pages)
    torch.testing.assert_close(pool.kmin[0][phys].float().transpose(0, 1),
                               kp.nan_to_num(float("inf")).amin(3).half().float()[0])
    torch.testing.assert_close(pool.kmax[0][phys].float().transpose(0, 1),
                               kp.nan_to_num(float("-inf")).amax(3).half().float()[0])


def test_quest_bound_is_an_upper_bound_and_forces_first_and_last_page():
    k, v = _kv(200)
    pool = _pool(track=True)
    c = _fill(pool, k, v)
    q = torch.randn(1, 4, 1, 64)  # 2 query heads per KV head
    sel = pool.select_pages(0, [c], [200], q, budget=0.25, min_tokens=0)[0]   # [Hkv, K]
    n_pages = -(-200 // PAGE)
    assert sel.shape[-1] == max(2, -(-n_pages // 4)) and (sel >= 0).all()
    assert all({0, n_pages - 1} <= set(row.tolist()) for row in sel)
    qg = q.view(1, 2, 2, 64)
    exact = (qg @ k.transpose(-1, -2)).amax(2)[0]                                       # [Hkv, T]
    per_page = torch.nn.functional.pad(exact, (0, n_pages * PAGE - 200), value=float("-inf"))
    per_page = per_page.unflatten(-1, (n_pages, PAGE)).amax(-1)
    phys = torch.tensor(c.pages)
    kmin, kmax = pool.kmin[0][phys].float().transpose(0, 1), pool.kmax[0][phys].float().transpose(0, 1)
    bound = (qg[0].clamp(min=0) @ kmax.transpose(-1, -2) + qg[0].clamp(max=0) @ kmin.transpose(-1, -2)).amax(1)
    assert (per_page <= bound + 1e-2).all()


def test_batched_selection_equals_per_row_selection():
    pool = _pool(track=True, pages=256)
    caches, lengths = [], (30, 300, 1000)
    for i, T in enumerate(lengths):
        k, v = _kv(T, seed=i)
        caches.append(_fill(pool, k, v))
    q = torch.randn(3, 4, 1, 64)
    batched = pool.select_pages(0, caches, list(lengths), q, 0.25, min_tokens=64)
    for b, (c, T) in enumerate(zip(caches, lengths)):
        alone = pool.select_pages(0, [c], [T], q[b:b + 1], 0.25, min_tokens=64)[0]
        assert [set(r.tolist()) - {-1} for r in batched[b]] == [set(r.tolist()) - {-1} for r in alone]
    assert (batched[0] >= 0).sum(-1).tolist() == [2, 2]  # 30 tokens < min_tokens: both of its pages kept


def test_fork_shares_full_pages_and_isolates_writes():
    k, v = _kv(40)
    pool = _pool("int8")
    parent = _fill(pool, k, v)
    child = parent.clone()
    assert child.pages[:2] == parent.pages[:2] and child.pages[2] != parent.pages[2]  # 2 full pages shared
    free_before = pool.num_free_pages
    for c, seed in ((parent, 1), (child, 2)):
        nk, nv = _kv(9, seed=seed)
        c.reserve(c.length + 9)
        pool.append(0, c, nk, nv)
        c.length += 9
    kp, _ = pool.read(0, parent, 49)
    kc, _ = pool.read(0, child, 49)
    torch.testing.assert_close(kp[:, :, :40], kc[:, :, :40], rtol=0, atol=0)     # the shared prefix
    assert not torch.equal(kp[:, :, 40:], kc[:, :, 40:])                          # each fork's own tokens
    assert pool.num_free_pages == free_before - 2  # one new page each for token 48
    child.free()
    parent.free()
    assert pool.num_free_pages == pool.num_pages


def test_independent_copy_and_exhaustion():
    k, v = _kv(32)
    pool = _pool(pages=4)
    a = _fill(pool, k, v)
    b = a.clone(share=False)
    assert set(a.pages).isdisjoint(b.pages)
    torch.testing.assert_close(pool.read(0, a, 32)[0], pool.read(0, b, 32)[0], rtol=0, atol=0)
    with pytest.raises(KVPoolExhausted):
        a.clone(share=False)
    del b  # garbage collection returns the pages
    assert pool.num_free_pages == 2


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
def test_chunked_prefill_equals_step_decode_and_full_budget_equals_no_selection(kv_cache_dtype):
    hf = _tiny_llama()
    ids = torch.randint(0, 128, (1, 40), generator=torch.Generator().manual_seed(2))
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    model.config.kv_cache_dtype, model.config.kv_pool_gib = kv_cache_dtype, 0.01

    chunked, cache = model(ids[:, :24])
    tail, _ = model(ids[:, 24:], past_key_values=cache)
    chunked = torch.cat([chunked, tail], 1)

    _, cache = model(ids[:, :24])
    steps = [model(ids[:, t:t + 1], past_key_values=cache)[0] for t in range(24, 40)]
    torch.testing.assert_close(torch.cat(steps, 1), chunked[:, 24:], atol=1e-4, rtol=0)

    model.config.kv_page_budget, model.config.kv_select_min_tokens = 1.0 - 1e-9, 0  # selection on, keeps every page
    _, cache = model(ids[:, :24])
    selected = [model(ids[:, t:t + 1], past_key_values=cache)[0] for t in range(24, 40)]
    torch.testing.assert_close(torch.cat(selected, 1), chunked[:, 24:], atol=1e-4, rtol=0)


def test_tuple_caches_are_rejected():
    model = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    with pytest.raises(TypeError, match="tuples are not supported"):
        model(torch.tensor([[1]]), past_key_values=[(None, None)] * len(model.layers))


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8"])
def test_batched_decode_across_requests_matches_each_request_alone(kv_cache_dtype):
    hf = _tiny_llama()
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    model.config.kv_cache_dtype, model.config.kv_pool_gib = kv_cache_dtype, 0.01
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(0, 128, (1, n), generator=g) for n in (5, 17, 33)]  # different lengths and positions
    nxt = torch.randint(0, 128, (3, 3), generator=g)

    alone = []
    for p, toks in zip(prompts, nxt):
        _, c = model(p)
        alone.append(torch.cat([model(toks[t].view(1, 1), past_key_values=c)[0] for t in range(3)], 1))

    caches = [model(p)[1] for p in prompts]
    batched = torch.cat([model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(3)], 1)
    torch.testing.assert_close(batched, torch.cat(alone), atol=1e-4, rtol=0)
    assert [c.length for c in caches] == [8, 20, 36]


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
def test_kv_bytes_per_token_matches_what_a_page_holds(kv_cache_dtype):
    pool = _pool(kv_cache_dtype, H=3, layers=2)
    k, v = _kv(48, H=3)
    c = _fill(pool, k, v)
    assert c.nbytes() == 48 * kv_bytes_per_token(2, 3, 64, kv_cache_dtype, native_bytes=4)


def test_engine_reserves_pages_at_admission_frees_them_at_finish_and_rejects_what_cannot_fit():
    import asyncio
    import dataclasses
    import time
    from turing.config import TuringConfig
    from turing.models.registry import get_model_config
    from turing.serving.engine import ContinuousBatchEngine
    cfg = dataclasses.replace(get_model_config("test-tiny"), kv_pool_gib=0.002)  # the registry's object is shared
    engine = ContinuousBatchEngine(cfg, TuringConfig(device="cpu", max_batch_size=4))
    pool = engine.kv_pool
    req = asyncio.run(engine.add_request(prompt_tokens=list(range(20)), max_new_tokens=12))
    assert pool.used_pages == 2 and req.past_kv.length == 0  # 32 tokens -> 2 pages, held before prefill
    engine._finish_request(req, time.time())
    assert pool.used_pages == 0 and req.past_kv is None
    with pytest.raises(RuntimeError, match="rejected"):
        asyncio.run(engine.add_request(prompt_tokens=[1], max_new_tokens=pool.num_pages * PAGE + 1))
    assert pool.used_pages == 0


def test_elastic_manager_drives_the_serving_pool():
    from turing.core.elastic_memory import ElasticMemoryBudgetManager
    from turing.core.expert_cache import GPULRUExpertCache
    experts = GPULRUExpertCache(num_slots=32, hidden_dim=256, active_subspace_dim=128, device=torch.device("cpu"),
                                dtype=torch.float32)
    pool = _pool(pages=256, layers=4)
    manager = ElasticMemoryBudgetManager(expert_cache=experts, kv_pool=pool, min_kv_pages=32, max_kv_pages=256)
    assert manager.bytes_per_page == max(1024, pool.page_bytes())
    res = manager.evaluate_and_rebalance(current_active_tokens=256, force=True)
    assert pool.page_limit == res["active_kv_pages"] == pool.get_stats()["total_pages"]
    with pytest.raises(KVPoolExhausted):
        KVCache(pool).reserve((pool.page_limit + 1) * PAGE)


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
def test_batched_read_and_append_equal_per_request(kv_cache_dtype):
    pool = _pool(kv_cache_dtype, pages=64)
    g = torch.Generator().manual_seed(0)
    caches = []
    for T in (3, 17, 40):
        c = KVCache(pool)
        c.reserve(T + 1)
        H, D = pool.num_kv_heads, pool.head_dim
        pool.append(0, c, torch.randn(1, H, T, D, generator=g), torch.randn(1, H, T, D, generator=g))
        c.length = T
        caches.append(c)
    ref = [c.clone(share=False) for c in caches]
    k = torch.randn(3, pool.num_kv_heads, 1, pool.head_dim, generator=g)
    v = torch.randn(3, pool.num_kv_heads, 1, pool.head_dim, generator=g)
    slots = torch.tensor([c.slot for c in caches], dtype=torch.int32)
    starts = torch.tensor([c.length for c in caches], dtype=torch.int32)
    pool.append_batch(0, slots, starts, k, v)
    for b, r in enumerate(ref):
        pool.append(0, r, k[b:b + 1], v[b:b + 1])
    kb, vb, valid = pool.read_batch(0, slots, [c.length + 1 for c in caches])
    for b, r in enumerate(ref):
        n = r.length + 1
        kr, vr = pool.read(0, r, n)
        torch.testing.assert_close(kb[b:b + 1, :, :n], kr, rtol=0, atol=0)
        torch.testing.assert_close(vb[b:b + 1, :, :n], vr, rtol=0, atol=0)
        assert valid[b].sum() == n


def test_pool_created_in_inference_mode_is_writable_outside_it():
    with torch.inference_mode():
        pool = _pool(pages=16)
        c = KVCache(pool)
    c.reserve(40)  # admission runs outside inference mode
    for _ in range(9):  # grows the page table
        KVCache(pool)


@torch.no_grad()
def test_key_smoothing_is_exact_and_tames_outlier_channels_in_a_4bit_cache():
    """Qwen2-style key-bias outliers: smoothing leaves full-precision logits unchanged and makes the int4 cache's
    logits much closer to them."""
    import copy
    import transformers as tf
    from tests.test_hf_parity import _COMMON
    from turing.models.hf_loader import RealHuggingFaceLoader
    torch.manual_seed(0)
    hf = tf.Qwen2ForCausalLM(tf.Qwen2Config(**{**_COMMON, "num_attention_heads": 2, "num_key_value_heads": 1})).eval()
    for layer in hf.model.layers:
        layer.self_attn.k_proj.bias[::7] = 40.0  # a few channels 40x the rest
    ids = torch.randint(0, 128, (1, 48), generator=torch.Generator().manual_seed(0))
    base = RealHuggingFaceLoader.convert_hf_model(hf)
    ref = base(ids)[0]
    errs = {}
    for smooth in (False, True):
        m = copy.deepcopy(base)
        m.config.kv_cache_dtype, m.config.kv_smoothing, m.config.kv_pool_gib = "int4", smooth, 0.01
        logits, cache = m(ids[:, :40])
        outs = [logits] + [m(ids[:, t:t + 1], past_key_values=cache)[0] for t in range(40, 48)]
        errs[smooth] = (torch.cat(outs, 1) - ref).abs().max().item()
    assert errs[True] < 0.7 * errs[False]  # Qwen2.5-0.5B real weights: int4 +0.74 -> +0.08 nats
    exact = copy.deepcopy(base)
    exact.calibrate_kv_smoothing()
    torch.testing.assert_close(exact(ids)[0], ref, atol=1e-4, rtol=0)


@pytest.mark.parametrize("dtype", ["int8", "int4"])
def test_native_layers_store_exactly_and_others_quantize(dtype):
    k, v = _kv(40)
    pool = KVPool(2, 2, 64, 16, torch.float32, "cpu", dtype, native_layers=(0,))
    c = KVCache(pool)
    c.reserve(40)
    for layer in (0, 1):
        pool.append(layer, c, k, v)
    c.length = 40
    k0, v0 = pool.read(0, c, 40)
    torch.testing.assert_close(k0, k, rtol=0, atol=0)
    torch.testing.assert_close(v0, v, rtol=0, atol=0)
    k1, _ = pool.read(1, c, 40)
    assert not torch.equal(k1, k) and (k1 - k).abs().max() < 2.0  # quantized, within its step
    slots = torch.tensor([c.slot])
    kb, vb, _ = pool.read_batch(0, slots, [40])
    torch.testing.assert_close(kb[:, :, :40], k, rtol=0, atol=0)
    plain = KVPool(2, 2, 64, 16, torch.float32, "cpu", dtype)
    assert pool.page_bytes() - plain.page_bytes() == 2 * 2 * PAGE * 64 * 4  # one layer's K and V in fp32
    dst = pool.alloc_pages(1)[0]
    pool.copy_page(c.pages[0], dst)
    torch.testing.assert_close(pool.k_hi[0, dst], pool.k_hi[0, c.pages[0]], rtol=0, atol=0)
    assert KVPool(2, 2, 64, 16, torch.float32, "cpu", "native", native_layers=(0,)).k_hi is None


@torch.no_grad()
def test_full_precision_layer_selection():
    from tests.test_hf_parity import _tiny
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cpu")
    model.config.kv_cache_dtype = "int4"
    ids = torch.randint(0, 128, (1, 12))
    before = model._body(ids, 0, None, None)  # stateless: no pool, no KV storage
    L = model.config.num_layers
    assert model.select_kv_native_layers(threshold=1e9) == ()
    assert len(model.select_kv_native_layers(threshold=0.0)) == max(1, L // 4)  # the cap, worst layers first
    kls = model._kv_layer_kl
    assert len(kls) == L and all(k >= 0 for k in kls)
    if L < 8:
        assert model.select_kv_native_layers(threshold=0.0) == (max(range(L), key=lambda i: kls[i]),)
    torch.testing.assert_close(model._body(ids, 0, None, None), before, rtol=0, atol=0)  # round trip off again

    model.config.kv_native_layers = ()  # explicit: kept
    assert model.kv_pool().native_layers == ()
    model.config.kv_native_layers = None  # automatic: chosen when the pool is made, again for another storage
    model.kv_pool()
    assert model.config.kv_native_layers is not None and model._kv_native_auto == "int4"
    model.config.kv_cache_dtype = "int8"
    model.kv_pool()
    assert model._kv_native_auto == "int8"
