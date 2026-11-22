"""Fused Triton decode over the paged KV pool against the reference path (CUDA only)."""
import pytest
import torch
import torch.nn.functional as F

from turing.core.kv_cache import PAGE, KVCache, KVPool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton decode kernels")


def _filled(pool, T, g):
    c = KVCache(pool)
    c.reserve(T)
    H, D = pool.num_kv_heads, pool.head_dim
    pool.append(0, c, torch.randn(1, H, T, D, device="cuda", generator=g).half(),
                torch.randn(1, H, T, D, device="cuda", generator=g).half())
    c.length = T
    return c


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4", "fp8"])
@pytest.mark.parametrize("H,Hkv,D", [(9, 3, 64), (32, 8, 128)])
@pytest.mark.parametrize("budget", [None, 0.25])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("qdtype", [torch.float32, torch.float16])  # fp16: tensor-core products in fp16
def test_fused_decode_step_matches_reference(kv_cache_dtype, H, Hkv, D, budget, native, qdtype):
    from turing.kernels.triton_kv_decode import triton_pool_decode
    if native and kv_cache_dtype == "native":
        pytest.skip("a native pool has no full-precision layers of its own")
    g = torch.Generator(device="cuda").manual_seed(1)
    pool = KVPool(1, Hkv, D, 512, torch.float16, "cuda", kv_cache_dtype, track_pages=budget is not None,
                  native_layers=(0,) if native else ())
    lengths = (1, 255, 1000)  # 255 + 1 fills a page exactly; 1000 + 1 is mid-page
    fused = [_filled(pool, T, g) for T in lengths]
    ref = [c.clone(share=False) for c in fused]
    q = torch.randn(3, H, 1, D, device="cuda", generator=g).half()
    k = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()
    v = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()
    for c in fused + ref:
        c.reserve(c.length + 1)
    totals = [T + 1 for T in lengths]

    sel = None
    if budget is not None:
        for b, c in enumerate(ref):  # Quest bounds include this step's key on both paths
            pool.append(0, c, k[b:b + 1], v[b:b + 1])
        sel = pool.select_pages(0, ref, totals, q, budget, min_tokens=0)
    rows = torch.tensor([[c.slot for c in fused], list(lengths)], dtype=torch.int32, device="cuda")
    out = triton_pool_decode(pool, 0, q.to(qdtype), k, v, rows[0], rows[1], sel)  # output in the query's dtype
    assert out.dtype == qdtype
    out = out.float()

    pool.dtype = torch.float32  # the reference dequantizes in fp32, as the kernel does
    for b, (c, r, T) in enumerate(zip(fused, ref, totals)):
        if budget is None:
            pool.append(0, r, k[b:b + 1], v[b:b + 1])
        page_f, page_r, slot = c.pages[(T - 1) // PAGE], r.pages[(T - 1) // PAGE], (T - 1) % PAGE
        # the fused append stores exactly what the reference append stores, in the fields layer 0 uses (a layer the
        # pool keeps in the model dtype never writes its low-bit slots)
        for t in (x for w in ("k", "v") for x in pool.layer_fields(0, w)[:3] if x is not None and x.dim() > 1):
            torch.testing.assert_close(t[page_f, :, slot], t[page_r, :, slot], rtol=0, atol=0)
        if budget is None:
            kr, vr = pool.read(0, r, T)
            ref_out = F.scaled_dot_product_attention(q[b:b + 1].float(), kr.float(), vr.float(), enable_gqa=True)
        else:
            kr, vr, valid = pool.read_selected(0, r, sel[b], T)
            ref_out = F.scaled_dot_product_attention(q[b:b + 1].float(), kr.float(), vr.float(),
                                                     attn_mask=valid.repeat_interleave(H // Hkv, dim=1), enable_gqa=True)
        torch.testing.assert_close(out[b:b + 1], ref_out, atol=1e-3, rtol=0 if qdtype == torch.float32 else 1e-2)
    if budget is not None:
        for c, r in zip(fused, ref):
            torch.testing.assert_close(pool.kmin[0, c.pages[-1]], pool.kmin[0, r.pages[-1]], rtol=0, atol=0)


# Families whose decode differs from Llama's in the kernel or the graph: sliding windows, soft-capping with a custom
# scale (Gemma 2), two RoPE bases (Gemma 3), learned positions (GPT-2), partial rotary (Phi-2), MoE (OLMoE, no graphs).
ARCH_CASES = [("llama", "native"), ("llama", "int8"), ("mistral_sliding", "native"), ("gemma2", "native"),
              ("gemma3", "int8"), ("gpt2", "native"), ("phi2", "native"), ("olmoe", "native")]


@torch.no_grad()
@pytest.mark.parametrize("arch,kv_cache_dtype", ARCH_CASES)
def test_model_batched_decode_on_cuda_matches_cpu(arch, kv_cache_dtype):
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    cpu = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    cpu.config.kv_cache_dtype, cpu.config.kv_pool_gib = kv_cache_dtype, 0.05
    gpu = copy.deepcopy(cpu).cuda()  # fp32 on CUDA: the fused kernels against the reference path on CPU
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 128, (1, n), generator=g) for n in (3, 20, 41)]
    nxt = torch.randint(0, 128, (3, 4), generator=g)  # prompts and steps cross the 6-token windows
    outs = []
    for model, dev in ((cpu, "cpu"), (gpu, "cuda")):
        caches = [model(p.to(dev))[1] for p in prompts]
        outs.append(torch.cat([model(nxt[:, t:t + 1].to(dev), past_key_values=caches)[0].cpu() for t in range(4)], 1))
    # int8: the two devices' fp32 activations differ in the last bits, so a few codes round to the neighbouring level
    torch.testing.assert_close(outs[1], outs[0], atol=1e-3 if kv_cache_dtype == "native" else 1e-2, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("arch,kv_cache_dtype", ARCH_CASES + [("mixtral", "native"), ("qwen2_moe", "int8")])
def test_cuda_graph_decode_matches_eager(arch, kv_cache_dtype):
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    base = RealHuggingFaceLoader.convert_hf_model(_tiny(arch), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    base.config.kv_cache_dtype, base.config.kv_pool_gib = kv_cache_dtype, 0.05
    g = torch.Generator().manual_seed(6)
    prompts = [torch.randint(0, 128, (1, n), generator=g).cuda() for n in (3, 14, 31)]  # steps cross page boundaries
    nxt = torch.randint(0, 128, (3, 6), generator=g).cuda()
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base).cuda()
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        steps = [model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(4)]  # 3 rows in the 4-row graph
        steps += [model(nxt[:1, t:t + 1], past_key_values=caches[:1])[0] for t in range(4, 6)]  # the 1-row graph
        outs.append((steps, [c.length for c in caches]))
        assert (len(model._graphs.graphs) == 2) == graphs
    for eager, graphed in zip(outs[0][0], outs[1][0]):
        torch.testing.assert_close(graphed, eager, atol=1e-4, rtol=0)
    assert outs[0][1] == outs[1][1] == [9, 18, 35]


@torch.no_grad()
def test_ten_requests_decode_in_the_twelve_row_graph():
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    base = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    base.config.kv_pool_gib = 0.05
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 128, (1, 3 + 2 * i), generator=g).cuda() for i in range(10)]
    nxt = torch.randint(0, 128, (10, 3), generator=g).cuda()
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base).cuda()
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        outs.append([model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(3)])
        assert list(model._graphs.graphs) == ([(12, None)] if graphs else [])
    for eager, graphed in zip(*outs):
        torch.testing.assert_close(graphed, eager, atol=1e-4, rtol=0)


@pytest.mark.parametrize("window,softcap,scale", [(6, 0.0, None), (0, 5.0, 24 ** -0.5), (40, 5.0, 0.1)])
def test_fused_decode_window_softcap_and_scale_match_reference(window, softcap, scale):
    from turing.kernels.triton_kv_decode import triton_pool_decode
    g = torch.Generator(device="cuda").manual_seed(3)
    H, Hkv, D = 8, 2, 64
    pool = KVPool(1, Hkv, D, 256, torch.float16, "cuda", "native")
    lengths = (5, 100, 300)
    caches = [_filled(pool, T, g) for T in lengths]
    q = torch.randn(3, H, 1, D, device="cuda", generator=g)
    k = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()
    v = torch.randn(3, Hkv, 1, D, device="cuda", generator=g).half()
    for c in caches:
        c.reserve(c.length + 1)
    rows = torch.tensor([[c.slot for c in caches], list(lengths)], dtype=torch.int32, device="cuda")
    out = triton_pool_decode(pool, 0, q, k, v, rows[0], rows[1], None, scale=scale, window=window, softcap=softcap)
    for b, (c, T) in enumerate(zip(caches, lengths)):
        kr, vr = pool.read(0, c, T + 1)  # the kernel appended this step's key
        kr = kr.float().repeat_interleave(H // Hkv, dim=1)
        vr = vr.float().repeat_interleave(H // Hkv, dim=1)
        s = q[b:b + 1] @ kr.transpose(2, 3) * (scale if scale is not None else D ** -0.5)
        if softcap:
            s = softcap * torch.tanh(s / softcap)
        if window:
            s[..., : max(0, T + 1 - window)] = float("-inf")
        ref = torch.softmax(s, -1) @ vr
        torch.testing.assert_close(out[b:b + 1], ref, atol=1e-3, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8"])
def test_quest_cuda_graph_decode_matches_eager_quest(kv_cache_dtype):
    """Quest inside a CUDA graph (static page-table width, selection on the device) picks the same pages as eager
    Quest, so the logits agree."""
    import copy
    from tests.test_hf_parity import _tiny_llama
    from turing.models.hf_loader import RealHuggingFaceLoader
    base = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    base.config.kv_cache_dtype, base.config.kv_pool_gib = kv_cache_dtype, 0.05
    base.config.kv_page_budget, base.config.kv_select_min_tokens = 0.25, 48
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 128, (1, n), generator=g).cuda() for n in (20, 70, 120)]  # one row below the threshold
    nxt = torch.randint(0, 128, (3, 5), generator=g).cuda()
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base).cuda()
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        outs.append([model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(5)])
        assert any(w is not None for _, w in model._graphs.graphs) == graphs
    for eager, graphed in zip(*outs):
        torch.testing.assert_close(graphed, eager, atol=1e-3, rtol=0)


@torch.no_grad()
def test_cuda_graph_decode_steps_queued_behind_other_gpu_work_match_eager():
    """A caller may queue step after step before the GPU has run the first one (here it is kept busy first). Each step
    must still decode the rows it was called with: they travel through two pinned buffers that are reused."""
    import copy
    from tests.test_hf_parity import _tiny_llama
    from turing.models.hf_loader import RealHuggingFaceLoader
    base = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    base.config.kv_cache_dtype, base.config.kv_pool_gib = "native", 0.05
    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, 128, (1, n), generator=g).cuda() for n in (20, 70, 120)]
    nxt = torch.randint(0, 128, (3, 9), generator=g).cuda()
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base).cuda()
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        first = model(nxt[:, :1], past_key_values=caches)[0]  # captures the graph
        busy = torch.eye(4096, device="cuda")
        for _ in range(30):  # about half a second of work queued ahead of the steps below
            busy = busy @ busy
        outs.append([first] + [model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(1, 9)])
        assert bool(model._graphs.graphs) == graphs
    for eager, graphed in zip(*outs):
        torch.testing.assert_close(graphed, eager, atol=1e-3, rtol=0)


@torch.no_grad()
def test_sdpa_decode_attention_matches_the_kernel():
    import copy
    from tests.test_hf_parity import _tiny_llama
    from turing.models.hf_loader import RealHuggingFaceLoader
    base = RealHuggingFaceLoader.convert_hf_model(_tiny_llama(), device="cpu")
    base.config.kv_pool_gib = 0.05
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(0, 128, (1, n), generator=g).cuda() for n in (5, 33)]
    nxt = torch.randint(0, 128, (2, 3), generator=g).cuda()
    outs = []
    for mode in ("kernel", "sdpa"):
        model = copy.deepcopy(base).cuda()
        model.config.decode_attention, model.config.cuda_graphs = mode, False
        caches = [model(p)[1] for p in prompts]
        outs.append(torch.cat([model(nxt[:, t:t + 1], past_key_values=caches)[0] for t in range(3)], 1))
    torch.testing.assert_close(outs[1], outs[0], atol=1e-3, rtol=0)


@pytest.mark.parametrize("kv_cache_dtype", ["native", "int8", "int4", "fp8"])
def test_one_split_writes_the_output_directly_and_matches_the_split_path(kv_cache_dtype):
    """n_splits = 1 (a large batch) stores attention without the reduce launch; same result as the split path. K and V
    are passed as non-contiguous views of one fused projection, as the model's fused q/k/v gives them."""
    from turing.kernels.triton_kv_decode import triton_pool_decode
    g = torch.Generator(device="cuda").manual_seed(2)
    H, Hkv, D = 12, 2, 128
    pool = KVPool(1, Hkv, D, 256, torch.float16, "cuda", kv_cache_dtype)
    caches = [_filled(pool, T, g) for T in (5, 300, 700)]
    q = torch.randn(3, H, 1, D, device="cuda", generator=g).half()
    kv = torch.randn(3, 1, 2 * Hkv * D, device="cuda", generator=g).half()
    k = kv[..., :Hkv * D].view(3, 1, Hkv, D).transpose(1, 2)
    v = kv[..., Hkv * D:].view(3, 1, Hkv, D).transpose(1, 2)
    for c in caches:
        c.reserve(c.length + 1)
    twins = [c.clone(share=False) for c in caches]
    rows = torch.tensor([[c.slot for c in caches], [c.length for c in caches]], dtype=torch.int32, device="cuda")
    rows2 = torch.tensor([[c.slot for c in twins], [c.length for c in twins]], dtype=torch.int32, device="cuda")
    split = triton_pool_decode(pool, 0, q, k, v, rows[0], rows[1])
    one = triton_pool_decode(pool, 0, q, k, v, rows2[0], rows2[1], n_splits=1)
    torch.testing.assert_close(one.float(), split.float(), atol=1e-3, rtol=0)
