"""4-bit weights: packing, RTN and GPTQ against their definitions, QuantLinear against dequantize + matmul, and the
Triton W4A16 kernel against the same reference (CUDA)."""
import pytest
import torch

from turing.models.quant import (GROUP, QuantLinear, dequantize, gptq_quantize, pack, rtn_quantize, unpack,
                                 quantize_model)


def test_pack_round_trip_and_rtn_definition():
    torch.manual_seed(0)
    W = torch.randn(32, 128)
    codes, scales, zeros = rtn_quantize(W)
    assert torch.equal(unpack(pack(codes)), codes)
    g = W.view(32, -1, GROUP)
    s = ((g.amax(-1) - g.amin(-1)) / 15).clamp_min(1e-8).half().float()
    torch.testing.assert_close(scales.float(), s)
    W_hat = dequantize(pack(codes), scales, zeros, torch.float32)
    assert (W_hat - W).abs().max() <= s.max() / 2 + 1e-6  # within half a step of the grid


def test_gptq_beats_rtn_on_correlated_inputs():
    torch.manual_seed(0)
    K = 256
    X = torch.randn(4096, 16) @ torch.randn(16, K) + 0.1 * torch.randn(4096, K)  # strongly correlated features
    W = torch.randn(64, K)
    H = X.double().T @ X.double()

    def out_err(c, s, z):
        return ((X @ dequantize(pack(c), s, z, torch.float32).T - X @ W.T) ** 2).mean()
    assert out_err(*gptq_quantize(W, H)) < 0.7 * out_err(*rtn_quantize(W))


def test_quant_linear_matches_dequantized_matmul_and_quantize_model_runs():
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("qwen2_qkv_bias"))
    ids = torch.randint(0, 128, (2, 16))
    ref = model(ids)[0]
    for method, calib in (("rtn", None), ("gptq", torch.randint(0, 128, (4, 32)))):
        import copy
        m = quantize_model(copy.deepcopy(model), method, calib)
        lin = m.layers[0].self_attn.q_proj
        assert isinstance(lin, QuantLinear) and lin.bias is not None
        x = torch.randn(3, lin.in_features)
        torch.testing.assert_close(lin(x), x @ lin.dequantized(torch.float32).T + lin.bias)
        out = m(ids)[0]
        assert out.shape == ref.shape and (out - ref).abs().mean() < 0.5  # close to the dense model, not equal
        logits, past = m(ids[:1, :8])
        assert m(ids[:1, 8:9], past_key_values=past)[0].shape == (1, 1, 128)  # decode through the cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton W4A16 kernel")
@pytest.mark.parametrize("M", [1, 3, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N,K", [(384, 512), (2048, 8192)])  # the second splits a long K across many programs
def test_triton_w4a16_matches_reference(M, dtype, N, K):
    from turing.kernels.triton_w4a16 import w4a16_gemm
    torch.manual_seed(0)
    codes, scales, zeros = rtn_quantize(torch.randn(N, K))
    q, s, z = pack(codes).cuda(), scales.cuda(), zeros.cuda()
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    ref = x.float() @ dequantize(q, s, z, torch.float32).T
    torch.testing.assert_close(w4a16_gemm(x, q, s, z).float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)


def test_tgate4_round_trip(tmp_path):
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import load_quantized, save_quantized, weight_bytes
    dense = RealHuggingFaceLoader.convert_hf_model(_tiny("gemma2"))
    model = quantize_model(copy.deepcopy(dense), "rtn")
    save_quantized(model, str(tmp_path))
    loaded, tok = load_quantized(str(tmp_path))
    assert tok is None and loaded.config.quantization == "w4a16-rtn" and loaded.config.arch == "gemma2"
    ids = torch.randint(0, 128, (1, 12))
    torch.testing.assert_close(loaded(ids)[0], model(ids)[0])
    assert isinstance(loaded.layers[1].mlp.down_proj, QuantLinear)
    assert weight_bytes(loaded) < 0.6 * weight_bytes(dense)  # embeddings and head stay fp32 in this tiny model


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS only")
@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_quantize_model_on_mps(method):
    """MPS has no float64: the quantizers run on the host and the packed layers land on MPS."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"), device="mps")
    quantize_model(model, method, torch.randint(0, 128, (2, 32)) if method == "gptq" else None)
    lin = model.layers[0].self_attn.q_proj
    assert isinstance(lin, QuantLinear) and lin.qweight.device.type == "mps"
    assert model(torch.randint(0, 128, (1, 8), device="mps"))[0].isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton dequantize kernel")
def test_triton_dequantize_matches_reference():
    from turing.kernels.triton_w4a16 import dequantize_cuda
    codes, scales, zeros = rtn_quantize(torch.randn(384, 512))
    q, s, z = pack(codes).cuda(), scales.cuda(), zeros.cuda()
    out = torch.empty(384, 512, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(dequantize_cuda(q, s, z, out), dequantize(q, s, z, torch.float16), rtol=0, atol=1e-3)


def test_dequantize_scratch_is_usable_inside_and_outside_inference_mode():
    from turing.models.quant import _scratch
    with torch.inference_mode():
        a = _scratch(1 << 20, torch.device("cpu"), torch.float16)  # first use inside inference mode
    a.copy_(torch.ones_like(a))  # then written outside it (GPTQ calibration runs under no_grad)
    with torch.inference_mode():
        _scratch(1 << 20, torch.device("cpu"), torch.float16).zero_()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton W4A16 kernel")
@pytest.mark.parametrize("split", [1, 4, None])
@pytest.mark.parametrize("M", [1, 5, 16])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N,K", [(256, 1536), (2048, 1536), (1536, 8960)])  # Qwen2.5-1.5B's k/v, fused qkv, down
def test_triton_w4a16_direct_and_split_with_bias_match_reference(split, M, dtype, N, K):
    """The direct kernel (one launch: bias and the store in x's dtype inside) and the split-K path agree with the
    dequantized reference, with a bias added."""
    from turing.kernels.triton_w4a16 import w4a16_gemm
    torch.manual_seed(0)
    codes, scales, zeros = rtn_quantize(torch.randn(N, K))
    q, s, z = pack(codes).cuda(), scales.cuda(), zeros.cuda()
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)
    ref = x.float() @ dequantize(q, s, z, torch.float32).T + bias.float()
    y = w4a16_gemm(x, q, s, z, bias=bias, split=split)
    assert y.dtype == dtype and y.shape == (M, N)
    torch.testing.assert_close(y.float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)


@pytest.mark.parametrize("arch", ["qwen2_qkv_bias", "llama"])
@pytest.mark.parametrize("quantized", [False, True])
def test_fused_projections_match_the_separate_layers(arch, quantized, tmp_path):
    """fuse_projections: q/k/v and gate/up as one linear each give the same logits, keep the state dict's keys and
    memory (the parts are views), save to .tgate4, and step aside for a forward hook or a replaced part."""
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import save_quantized
    torch.manual_seed(0)
    model = RealHuggingFaceLoader.convert_hf_model(_tiny(arch))
    if quantized:
        model = quantize_model(model, "rtn")
    ids = torch.randint(0, 128, (1, 12))
    ref = model(ids)[0]
    keys = list(model.state_dict())
    nbytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    fused = copy.deepcopy(model)
    assert fused.fuse_projections() == 2 * len(fused.layers)
    attn = fused.layers[0].self_attn
    from turing.models.causal_lm import fused_outputs
    x = torch.randn(1, 3, attn.hidden_dim)
    parts = (attn.q_proj, attn.k_proj, attn.v_proj)
    torch.testing.assert_close(torch.cat(fused_outputs(attn, "_qkv", parts, x), -1),
                               torch.cat([p(x) for p in parts], -1), atol=1e-5, rtol=1e-5)
    assert list(fused.state_dict()) == keys
    w = (lambda m: m.qweight) if quantized else (lambda m: m.weight)
    assert w(attn.k_proj).untyped_storage().data_ptr() == w(attn.q_proj).untyped_storage().data_ptr()
    assert sum(t.numel() * t.element_size() for t in fused.state_dict().values()) == nbytes
    torch.testing.assert_close(fused(ids)[0], ref, atol=1e-5, rtol=1e-5)
    seen = []
    h = attn.q_proj.register_forward_hook(lambda m, i, o: seen.append(o.shape))
    torch.testing.assert_close(fused(ids)[0], ref, atol=1e-5, rtol=1e-5)
    h.remove()
    assert seen, "a hooked part must be called on its own (GPTQ and TEAL calibrate through hooks)"
    attn.q_proj = copy.deepcopy(attn.q_proj)  # replaced (an adapter would do this): the fused layer is not used
    torch.testing.assert_close(fused(ids)[0], ref, atol=1e-5, rtol=1e-5)
    if quantized:
        save_quantized(fused, str(tmp_path))


@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_quantized_lm_head_round_trips_and_stays_close(method, tmp_path):
    """quantize_model(lm_head=True): the head becomes QuantLinear (a tied head gets its own packed copy and the
    embedding keeps its weight), logits stay close to the 4-bit body's, and .tgate4 saves and loads it."""
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import load_quantized, save_quantized
    torch.manual_seed(0)
    dense = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"))
    calib = torch.randint(0, 128, (4, 32)) if method == "gptq" else None
    body = quantize_model(copy.deepcopy(dense), method, calib)
    both = quantize_model(copy.deepcopy(dense), method, calib, lm_head=True)
    assert isinstance(both.lm_head, QuantLinear) and not isinstance(body.lm_head, QuantLinear)
    assert both.embed_tokens.weight.dtype == dense.embed_tokens.weight.dtype
    ids = torch.randint(0, 128, (1, 12))
    a, b = body(ids)[0], both(ids)[0]
    assert (a - b).abs().mean() < 0.1 * a.abs().mean()
    save_quantized(both, str(tmp_path))
    loaded, _ = load_quantized(str(tmp_path))
    assert isinstance(loaded.lm_head, QuantLinear)
    torch.testing.assert_close(loaded(ids)[0], b)


def test_kv_calibration_is_kept_between_processes(tmp_path, monkeypatch):
    """P7.2: a new model object with the same weights and KV settings takes the stored native layers and smoothing
    scales instead of calibrating again; other weights miss the cache."""
    import copy
    from tests.test_hf_parity import _tiny
    from turing.models.causal_lm import SubspaceCausalLM
    from turing.models.hf_loader import RealHuggingFaceLoader
    monkeypatch.setenv("TURING_CACHE_DIR", str(tmp_path))
    base = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"))  # head_dim 32: one int4 group per head
    base.config.kv_cache_dtype = "int4"
    first = copy.deepcopy(base)
    first.kv_pool()
    assert list((tmp_path / "kv_calibration").iterdir())
    second = copy.deepcopy(base)
    second.config.kv_native_layers = None
    calls = []
    monkeypatch.setattr(SubspaceCausalLM, "select_kv_native_layers", lambda self, *a, **k: calls.append(1))
    monkeypatch.setattr(SubspaceCausalLM, "calibrate_kv_smoothing", lambda self, *a, **k: calls.append(1))
    second.kv_pool()
    assert not calls
    assert second.config.kv_native_layers == first.config.kv_native_layers
    for a, b in zip(first.layers, second.layers):
        torch.testing.assert_close(a.self_attn.k_smooth, b.self_attn.k_smooth)
    other = copy.deepcopy(base)
    other.config.kv_native_layers = None
    with torch.no_grad():
        other.layers[0].self_attn.k_proj.weight.add_(1.0)
    other.kv_pool()
    assert calls  # different weights: calibrated again
