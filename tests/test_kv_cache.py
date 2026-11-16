"""KVCache: low-bit storage error bounds, growth, Quest page bounds, and equivalence of the decode paths."""
import pytest
import torch

from turing.core.kv_cache import PAGE, KVCache, LayerKVCache
from tests.test_hf_parity import _tiny_llama
from turing.models.hf_loader import RealHuggingFaceLoader


def _kv(T, H=2, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(1, H, T, D, generator=g)
    k[..., 3] *= 20  # an outlier channel, as real keys have
    return k, torch.randn(1, H, T, D, generator=g)


@pytest.mark.parametrize("dtype,levels,group", [("int8", 255, 64), ("int4", 15, 32)])
def test_low_bit_error_is_within_half_a_step_of_each_group(dtype, levels, group):
    k, v = _kv(40)
    c = LayerKVCache(dtype)
    c.append(k, v)
    kr, vr = c.read()
    for x, xr in ((k, kr), (v, vr)):
        g = x.unflatten(-1, (-1, group))
        step = (g.amax(-1, keepdim=True) - g.amin(-1, keepdim=True)) / levels
        err = (x - xr).abs().unflatten(-1, (-1, group))
        assert (err <= step / 2 + 2e-3 * g.abs().amax(-1, keepdim=True)).all()  # + fp16 rounding of scale and zero


def test_growth_keeps_content_and_page_bounds():
    k, v = _kv(600)
    c = LayerKVCache("native", track_pages=True)
    for s in range(0, 600, 37):  # uneven chunks cross pages and the 256 / 512 capacity steps
        c.append(k[:, :, s:s + 37], v[:, :, s:s + 37])
    kr, vr = c.read()
    torch.testing.assert_close(kr, k, rtol=0, atol=0)
    torch.testing.assert_close(vr, v, rtol=0, atol=0)
    pages = -(-600 // PAGE)
    kp = torch.nn.functional.pad(k, (0, 0, 0, pages * PAGE - 600), value=float("nan")).unflatten(2, (pages, PAGE))
    torch.testing.assert_close(c.store["kmin"][:, :, :pages].float(), kp.nan_to_num(float("inf")).amin(3).half().float())
    torch.testing.assert_close(c.store["kmax"][:, :, :pages].float(), kp.nan_to_num(float("-inf")).amax(3).half().float())


def test_quest_bound_is_an_upper_bound_and_forces_first_and_last_page():
    k, v = _kv(200, H=2)
    c = LayerKVCache("native", track_pages=True)
    c.append(k, v)
    q = torch.randn(1, 4, 1, 64)  # 2 query heads per KV head
    pos = c.select_pages(q, budget=0.25)
    pages = (pos[..., ::PAGE] // PAGE)
    n_pages = -(-200 // PAGE)
    assert pages.shape[-1] == max(2, -(-n_pages // 4))
    assert (pages[..., 0] == 0).all() and (pages[..., -1] == n_pages - 1).all()
    qg = q.view(1, 2, 2, 64)
    exact = (qg @ k.transpose(-1, -2)).amax(2)  # [1, Hkv, T]
    kmin, kmax = c.store["kmin"][:, :, :n_pages].float(), c.store["kmax"][:, :, :n_pages].float()
    bound = (qg.clamp(min=0) @ kmax.transpose(-1, -2) + qg.clamp(max=0) @ kmin.transpose(-1, -2)).amax(2)
    per_page = torch.nn.functional.pad(exact, (0, n_pages * PAGE - 200), value=float("-inf")).unflatten(-1, (n_pages, PAGE))
    assert (per_page.amax(-1) <= bound + 1e-2).all()


def test_clone_is_independent():
    k, v = _kv(20)
    a = KVCache(1, "int8")
    a[0].append(k, v)
    b = a.clone()
    b[0].append(k[:, :, :3], v[:, :, :3])
    assert a.length == 20 and b.length == 23
    torch.testing.assert_close(a[0].read()[0], b[0].read()[0][:, :, :20])


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
def test_chunked_prefill_equals_step_decode_and_full_budget_equals_no_selection(kv_cache_dtype):
    hf = _tiny_llama()
    ids = torch.randint(0, 128, (1, 40), generator=torch.Generator().manual_seed(2))
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    model.config.kv_cache_dtype = kv_cache_dtype

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


@torch.no_grad()
def test_legacy_tuple_cache_still_matches():
    hf = _tiny_llama()
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    empty = [(None, None)] * len(model.layers)
    logits, past = model(ids[:, :8], past_key_values=empty)
    assert isinstance(past, list) and past[0][0].shape[2] == 8
    step, _ = model(ids[:, 8:9], past_key_values=past)
    torch.testing.assert_close(step, hf(ids[:, :9]).logits[:, 8:9], atol=1e-3, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8"])
def test_batched_decode_across_requests_matches_each_request_alone(kv_cache_dtype):
    hf = _tiny_llama()
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    model.config.kv_cache_dtype = kv_cache_dtype
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
def test_kv_bytes_per_token_matches_what_the_cache_holds(kv_cache_dtype):
    from turing.core.kv_cache import kv_bytes_per_token
    k, v = _kv(48, H=3, D=64)
    cache = KVCache(2, kv_cache_dtype)
    for layer in cache.layers:
        layer.append(k, v)
    assert cache.nbytes() == 48 * kv_bytes_per_token(2, 3, 64, kv_cache_dtype, native_bytes=4)
