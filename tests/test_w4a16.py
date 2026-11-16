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
def test_triton_w4a16_matches_reference(M, dtype):
    from turing.kernels.triton_w4a16 import w4a16_gemm
    torch.manual_seed(0)
    N, K = 384, 512
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
