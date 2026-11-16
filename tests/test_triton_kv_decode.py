"""Fused Triton decode attention over the KV cache against the reference read + SDPA path (CUDA only)."""
import pytest
import torch
import torch.nn.functional as F

from turing.core.kv_cache import LayerKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton decode kernel")


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
@pytest.mark.parametrize("T,H,Hkv,D", [(1, 4, 4, 64), (300, 9, 3, 64), (4100, 32, 8, 128)])
@pytest.mark.parametrize("budget", [None, 0.25])
def test_kernel_matches_reference(kv_cache_dtype, T, H, Hkv, D, budget):
    from turing.kernels.triton_kv_decode import triton_kv_decode
    g = torch.Generator(device="cuda").manual_seed(0)
    k = torch.randn(1, Hkv, T, D, device="cuda", generator=g).half()
    v = torch.randn(1, Hkv, T, D, device="cuda", generator=g).half()
    q = torch.randn(1, H, 1, D, device="cuda", generator=g).half()
    cache = LayerKVCache(kv_cache_dtype, track_pages=budget is not None)
    cache.append(k, v)
    if budget is None:
        kr, vr = cache.read()
        ref = F.scaled_dot_product_attention(q.float(), kr.float(), vr.float(), enable_gqa=True)
        out = triton_kv_decode(q, cache)
    else:
        pos = cache.select_pages(q, budget)
        kr, vr, valid = cache.read_positions(pos)
        mask = valid.repeat_interleave(H // Hkv, dim=1)
        ref = F.scaled_dot_product_attention(q.float(), kr.float(), vr.float(), attn_mask=mask, enable_gqa=True)
        out = triton_kv_decode(q, cache, pos)
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=0)


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4"])
@pytest.mark.parametrize("H,Hkv,D", [(9, 3, 64), (32, 8, 128)])
def test_batched_append_and_decode_match_reference(kv_cache_dtype, H, Hkv, D):
    from turing.core.kv_cache import KVCache
    from turing.kernels.triton_kv_decode import build_decode_tables, triton_kv_decode_batch
    g = torch.Generator(device="cuda").manual_seed(1)
    lengths = (1, 255, 1000)  # 255 + 1 crosses the first capacity step, so ensure() grows it
    batched, reference = [], []
    for T in lengths:
        c = KVCache(1, kv_cache_dtype)
        c[0].append(torch.randn(1, Hkv, T, D, device="cuda", generator=g).half(),
                    torch.randn(1, Hkv, T, D, device="cuda", generator=g).half())
        batched.append(c)
        reference.append(c.clone())
    q = torch.randn(3, H, 1, D, device="cuda", generator=g).half()
    k = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()
    v = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()

    for c in batched:
        c[0].ensure(c.length + 1)
    table = build_decode_tables(batched, torch.device("cuda"))[0]
    out = triton_kv_decode_batch(q.float(), k, v, [c[0] for c in batched], table)  # fp32 query -> fp32 output

    for b, (c, r) in enumerate(zip(batched, reference)):
        r[0].append(k[b:b + 1], v[b:b + 1])
        assert c.length == r.length == lengths[b] + 1
        for name, t in r[0].store.items():  # the fused append stores exactly what the reference append stores
            torch.testing.assert_close(c[0].store[name][:, :, :r.length], t[:, :, :r.length], rtol=0, atol=0)
        r[0].dtype = torch.float32  # dequantize in fp32 as the kernel does (the model path rounds to fp16 first)
        kr, vr = r[0].read()
        ref = F.scaled_dot_product_attention(q[b:b + 1].float(), kr.float(), vr.float(), enable_gqa=True)
        torch.testing.assert_close(out[b:b + 1], ref, atol=1e-3, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8"])
def test_model_batched_decode_on_cuda_matches_cpu(kv_cache_dtype):
    import copy
    from tests.test_hf_parity import _tiny_llama
    from turing.models.hf_loader import RealHuggingFaceLoader
    cpu = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    cpu.config.kv_cache_dtype = kv_cache_dtype
    gpu = copy.deepcopy(cpu).cuda()  # fp32 on CUDA: the fused kernels against the reference path on CPU
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 128, (1, n), generator=g) for n in (3, 20, 41)]
    nxt = torch.randint(0, 128, (3, 4), generator=g)
    outs = []
    for model, dev in ((cpu, "cpu"), (gpu, "cuda")):
        caches = [model(p.to(dev))[1] for p in prompts]
        outs.append(torch.cat([model(nxt[:, t:t + 1].to(dev), past_key_values=caches)[0].cpu() for t in range(4)], 1))
    torch.testing.assert_close(outs[1], outs[0], atol=1e-3, rtol=0)
