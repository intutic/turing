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


def test_int4_layers_with_a_cached_prefix_take_the_gather_path_and_the_rest_the_kernel():
    from turing.kernels.triton_kv_prefill import pool_prefill_preferred
    q = torch.zeros(1, 4, 8, 64, device="cuda", dtype=torch.float16)
    int4 = KVPool(2, 2, 64, 8, torch.float16, "cuda", "int4", native_layers=(1,))
    assert pool_prefill_preferred(int4, 0, q, 0)  # no prefix: the kernel
    assert not pool_prefill_preferred(int4, 0, q, 16)  # int4 prefix: one gather beats per-block dequantization
    assert pool_prefill_preferred(int4, 1, q, 16)  # a layer kept native in an int4 pool
    assert pool_prefill_preferred(KVPool(2, 2, 64, 8, torch.float16, "cuda", "int8"), 0, q, 16)
