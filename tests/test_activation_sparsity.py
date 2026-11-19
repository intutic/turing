"""TEAL-style activation sparsity: thresholds reach the requested sparsity, and the CUDA sparse GEMV equals the dense
product of the thresholded input."""
import pytest
import torch

from turing.models.activation_sparsity import SparseInputLinear, calibrate_and_apply


def test_calibration_hits_the_target_sparsity_and_model_still_runs():
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama"))
    calib = torch.randint(0, 128, (8, 64))
    calibrate_and_apply(model, calib, 0.4)
    lin = model.layers[1].mlp.down_proj
    assert isinstance(lin, SparseInputLinear)
    seen = []
    h = lin.register_forward_hook(lambda m, i, o: seen.append(float((i[0].abs() < m.threshold).float().mean())))
    model(torch.randint(0, 128, (4, 64)))
    h.remove()
    assert 0.3 < seen[0] < 0.5  # about 40% of inputs fall under the threshold on fresh data
    logits, past = model(torch.randint(0, 128, (1, 10)))
    assert model(torch.randint(0, 128, (1, 1)), past_key_values=past)[0].isfinite().all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton sparse GEMV")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sparse_gemv_matches_dense(dtype):
    from turing.kernels.triton_sparse_gemv import sparse_gemv
    torch.manual_seed(0)
    K, N = 2048, 5632
    x = torch.randn(K, device="cuda", dtype=dtype)
    x[torch.rand(K, device="cuda") < 0.5] = 0
    wt = torch.randn(K, N, device="cuda", dtype=dtype)
    ref = x.float() @ wt.float()
    torch.testing.assert_close(sparse_gemv(x, wt).float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton sparse GEMV")
def test_sparse_int4_gemv_matches_its_dequantized_weight():
    from turing.kernels.triton_sparse_gemv import quantize_transposed_int4, sparse_gemv_int4
    from turing.models.quant import dequantize
    torch.manual_seed(0)
    N, K = 1024, 768
    w = torch.randn(N, K, device="cuda")
    q, s, z = quantize_transposed_int4(w)
    wt = dequantize(q, s, z, torch.float32)  # [K, N]
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    x[torch.rand(K, device="cuda") < 0.5] = 0
    ref = x.float() @ wt
    torch.testing.assert_close(sparse_gemv_int4(x, q, s, z).float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton sparse GEMV")
@pytest.mark.parametrize("K,N", [(1536, 256), (256, 65536)])  # split K; one split (the direct store)
def test_sparse_gemvs_add_the_bias_in_the_kernel(K, N):
    from turing.kernels.triton_sparse_gemv import quantize_transposed_int4, sparse_gemv, sparse_gemv_int4
    from turing.models.quant import dequantize
    torch.manual_seed(0)
    x = torch.randn(K, device="cuda", dtype=torch.float16)
    x[torch.rand(K, device="cuda") < 0.5] = 0
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    wt = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = x.float() @ wt.float() + bias.float()
    y = sparse_gemv(x, wt, bias)
    assert y.dtype == torch.float16
    torch.testing.assert_close(y.float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)
    q, s, z = quantize_transposed_int4(wt.float().t())
    ref4 = x.float() @ dequantize(q, s, z, torch.float32) + bias.float()
    torch.testing.assert_close(sparse_gemv_int4(x, q, s, z, bias).float(), ref4, atol=1e-2 * ref4.abs().max().item(),
                               rtol=1e-2)
