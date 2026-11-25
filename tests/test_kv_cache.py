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


def test_fp8_stores_pytorchs_e4m3_cast_and_a_layers_scale_sets_its_range():
    k, v = _kv(40)
    pool = _pool("fp8")
    kr, vr = pool.read(0, _fill(pool, k, v), 40)
    for x, xr in ((k, kr), (v, vr)):
        torch.testing.assert_close(xr, x.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)  # round to nearest even
        assert ((x - xr).abs() <= x.abs() / 16 + 2.0 ** -10).all()  # 3 mantissa bits: half a step
    # values past 448 and values below 2^-6 need the layer's scales: K / 4 reaches 1,792, V / 2^-6 resolves 2^-15
    big, small = k * 10, v * 0.01
    assert big.abs().max() > 448 and small.abs().median() < 2.0 ** -6
    for scales, exact in (([(4.0, 2.0 ** -6)], True), (None, False)):
        pool = KVPool(1, 2, 64, 128, torch.float32, "cpu", "fp8", fp8_scales=scales)
        kr, vr = pool.read(0, _fill(pool, big, small), 40)
        ok = all(((x - xr).abs() <= x.abs() / 16 + s * 2.0 ** -10).all()
                 for x, xr, s in ((big, kr, 4.0), (small, vr, 2.0 ** -6)))
        assert ok == exact
    assert pool.fp8_scales == ((1.0, 1.0),) and kr.abs().max() == 448  # without scales: saturated


@torch.no_grad()
def test_an_fp8_pool_takes_the_layer_guard_and_no_smoothing():
    from tests.test_hf_parity import _tiny
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="cpu")
    model.config.kv_cache_dtype, model.config.kv_pool_gib = "int4", 0.01
    model.kv_pool()
    assert model._kv_smoothed  # int4's key smoothing
    model.config.kv_cache_dtype = "fp8"
    pool = model.kv_pool()
    assert pool.kv_cache_dtype == "fp8" and pool.fp8_scales == ((1.0, 1.0),) * model.config.num_layers
    assert model.config.kv_native_layers is not None and model._kv_native_auto == "fp8"  # chosen again, for fp8
    assert not model._kv_smoothed and all(layer.self_attn.k_smooth is None for layer in model.layers)


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


def test_page_table_takes_more_requests_than_its_first_rows_and_keeps_their_pages():
    pool = KVPool(1, 1, 16, 4, torch.float32, "cpu", max_tokens_per_request=64)
    first = KVCache(pool)
    first.reserve(20)  # two pages
    rows = pool.page_table.shape[0]
    others = [KVCache(pool) for _ in range(rows)]  # one more request than the table had rows
    assert pool.page_table.shape[0] == 2 * rows and len({c.slot for c in others + [first]}) == rows + 1
    assert pool.page_table[first.slot, :2].tolist() == first.pages
    for c in others + [first]:
        c.free()
    assert pool.used_pages == 0


def test_a_cache_gives_back_pages_and_a_full_pool_keeps_its_rows():
    """trim() returns a cache's last pages (a prefill graph's padding) and reserve() takes pages again after it; the
    scratch row of the graphs' padding rows costs a full pool nothing when it cannot be had."""
    pool = KVPool(1, 1, 16, 4, torch.float32, "cpu")
    c = KVCache(pool)
    c.reserve(3 * PAGE)
    kept = c.pages[:1]
    c.trim(1)
    assert c.pages == kept and pool.num_free_pages == 3
    c.trim(2)  # holds fewer already: nothing to give back
    assert c.pages == kept
    c.reserve(4 * PAGE)
    assert len(c.pages) == 4 and len(set(c.pages)) == 4 and pool.num_free_pages == 0
    assert pool.page_table[c.slot, :4].tolist() == c.pages
    rows = len(pool._free_slots)
    with pytest.raises(KVPoolExhausted):
        pool.scratch_slot()
    assert len(pool._free_slots) == rows
    c.free()
    assert pool.page_table[pool.scratch_slot(), 0].item() in range(4) and pool.num_free_pages == 3


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
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4", "fp8"])
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


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4", "fp8"])
def test_kv_bytes_per_token_matches_what_a_page_holds(kv_cache_dtype):
    pool = _pool(kv_cache_dtype, H=3, layers=2)
    k, v = _kv(48, H=3)
    c = _fill(pool, k, v)
    assert c.nbytes() == 48 * kv_bytes_per_token(2, 3, 64, kv_cache_dtype, native_bytes=4)


def test_engine_reserves_the_prompts_pages_at_admission_frees_them_at_finish_and_rejects_what_cannot_fit():
    import asyncio
    import dataclasses
    import time
    from turing.config import TuringConfig
    from turing.models.registry import get_model_config
    from turing.serving.engine import ContinuousBatchEngine
    cfg = dataclasses.replace(get_model_config("test-tiny"), kv_pool_gib=0.002)  # the registry's object is shared
    engine = ContinuousBatchEngine(cfg, TuringConfig(device="cpu", max_batch_size=4))
    pool = engine.kv_pool
    req = asyncio.run(engine.add_request(prompt_tokens=list(range(20)), max_new_tokens=100))
    assert pool.used_pages == 2 and req.past_kv.length == 0  # the prompt's 20 tokens -> 2 pages, held before prefill
    engine._finish_request(req, time.time())
    assert pool.used_pages == 0 and req.past_kv is None
    with pytest.raises(RuntimeError, match="rejected"):
        asyncio.run(engine.add_request(prompt_tokens=[1], max_new_tokens=pool.num_pages * PAGE + 1))
    assert pool.used_pages == 0


def test_elastic_manager_drives_the_serving_pool():
    from turing.experimental.elastic_memory import ElasticMemoryBudgetManager
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


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4", "fp8"])
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
    # power-of-two scales: q * s and k / s are exact, so in bf16 the smoothed logits are bit-identical (rounding them
    # cost int8 KV 17 GSM8K points on Qwen2.5-1.5B-Instruct)
    b16 = RealHuggingFaceLoader.convert_hf_model(hf, dtype=torch.bfloat16)
    ref16 = b16._body(ids, 0, None, None)
    b16.calibrate_kv_smoothing()
    scales = torch.cat([layer.self_attn.k_smooth.float().flatten() for layer in b16.layers])
    assert scales.max() > 1 and torch.equal(torch.exp2(torch.log2(scales).round()), scales)
    assert ((scales == 1) | (scales >= 16)).all()  # only real outliers are scaled
    i8 = copy.deepcopy(base)
    i8.config.kv_cache_dtype, i8.config.kv_pool_gib = "int8", 0.01
    i8.kv_pool()
    assert not getattr(i8, "_kv_smoothed", False)  # int8 is stored without smoothing
    assert torch.equal(b16._body(ids, 0, None, None), ref16)


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
    # the low-bit tensors hold layer 1 only: one low-bit layer and one layer's K and V in fp32
    assert pool.k.shape[0] == 1 and pool.page_bytes() == plain.page_bytes() // 2 + 2 * 2 * PAGE * 64 * 4
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


def test_pool_off_cuda_is_sized_to_the_model():
    # a tiny model must not take 4 GiB off CUDA (Windows commits it up front; MPS shares host memory)
    from turing.models.registry import get_model_config
    cfg = get_model_config("test-tiny")
    pool = KVPool.for_model(cfg, torch.float16, "cpu")
    assert pool.num_pages == 64 * -(-cfg.max_position_embeddings // PAGE)
    assert pool.k.numel() * pool.k.element_size() < 2 ** 30


def test_cuda_budget_leaves_a_reserve_or_takes_the_share_asked_for():
    G = 2 ** 30
    assert KVPool.cuda_budget(18 * G, 22.5 * G) == pytest.approx(15.75 * G)  # a tenth of the device
    assert KVPool.cuda_budget(6 * G, 8 * G) == pytest.approx(4 * G)  # at least 2 GiB
    assert KVPool.cuda_budget(2 * G, 22.5 * G) == pytest.approx(1 * G)  # never more than half of what is free
    assert KVPool.cuda_budget(18 * G, 22.5 * G, 0.6) == pytest.approx(10.8 * G)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device memory is counted on CUDA")
def test_a_replaced_pool_is_freed_before_the_new_one_is_sized():
    model = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), device="cuda")
    model.config.kv_pool_gib = 0.25
    model.kv_pool()
    held = torch.cuda.memory_allocated()
    seen = []

    def choose(*a, **kw):
        seen.append(torch.cuda.memory_allocated())
        model.config.kv_native_layers = ()
    model.select_kv_native_layers = choose
    model._load_kv_calibration = lambda *a: False  # calibrate here, not from a file of an earlier run
    model._save_kv_calibration = lambda *a: None
    model.config.kv_cache_dtype = "int8"
    model.kv_pool()
    assert seen and seen[0] < held - 0.2 * 2 ** 30  # the fp16 pool was gone while the int8 pool's layers were chosen
