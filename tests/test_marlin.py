"""Marlin W4A16 (turing.kernels.marlin, vendored from vLLM v0.31.0): the layout conversion against a transcription of
vLLM's and AutoAWQ's reference loops (CPU), and on CUDA with the extension built, marlin_gemm and QuantLinear's Marlin
path against the dequantized reference and the Triton kernel."""
import numpy as np
import pytest
import torch

from turing.kernels.marlin import HAS_MARLIN, marlin_layout
from turing.models.quant import dequantize, pack, rtn_quantize

needs_marlin = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_MARLIN),
                                  reason="CUDA and the Marlin extension (TURING_BUILD_MARLIN=1) required")


def _vllm_scale_perms():
    """vllm marlin_utils.get_scale_perms, as written there."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _vllm_pack_cols(q, size_k, size_n):
    """vllm quant_utils.pack_cols for 4 bits."""
    q = q.astype(np.uint32)
    res = np.zeros((size_k, size_n // 8), dtype=np.uint32)
    for i in range(8):
        res |= q[:, i::8] << 4 * i
    return res.astype(np.int32)


def _reference_layout(codes, scales, zeros, bias, dtype):
    N, K = codes.shape
    G = scales.shape[1]
    # AutoAWQ's pack (the layout awq_marlin_repack reads): order_map [0, 2, 4, 6, 1, 3, 5, 7]
    w = codes.t().numpy().astype(np.uint32)
    awq = np.zeros((K, N // 8), dtype=np.uint32)
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    for col in range(N // 8):
        for i in range(8):
            awq[:, col] |= w[:, col * 8 + order_map[i]] << (i * 4)
    scale_perm, scale_perm_single = _vllm_scale_perms()
    # marlin_permute_scales (group < size_k: the grouped permutation)
    s = scales.t().to(dtype).reshape(-1, len(scale_perm))[:, scale_perm].reshape(-1, N).contiguous()
    # marlin_zero_points(zp, size_k=G, size_n=N, num_bits=4)
    zp = zeros.t().numpy().astype(np.int64).reshape(-1, len(scale_perm))[:, scale_perm]
    zp = zp.reshape(-1, 8)[:, np.array([0, 2, 4, 6, 1, 3, 5, 7])].ravel().reshape(-1, N)
    z = _vllm_pack_cols(zp, G, N)
    b = bias.to(dtype).reshape(-1, len(scale_perm_single))[:, scale_perm_single].reshape(N)  # marlin_permute_bias
    return torch.from_numpy(awq.astype(np.int32)), s, torch.from_numpy(z), b


@pytest.mark.parametrize("N,K,group", [(256, 512, 64), (192, 1024, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_marlin_layout_matches_vllm_reference(N, K, group, dtype):
    torch.manual_seed(0)
    codes, scales, zeros = rtn_quantize(torch.randn(N, K), group=group)
    bias = torch.randn(N)
    awq, s, z, b = marlin_layout(pack(codes), scales, zeros, dtype, bias)
    ref = _reference_layout(codes, scales, zeros, bias, dtype)
    for got, want in zip((awq, s, z, b), ref):
        assert got.dtype == want.dtype and torch.equal(got, want)
    # Read back the way awq_marlin_repack_kernel does: column n is nibble undo_pack[n % 8] of int32 [k, n / 8]
    undo_pack = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])
    n = torch.arange(N)
    nib = (awq[:, n // 8].to(torch.int64) >> (4 * undo_pack[n % 8])) & 0xF
    assert torch.equal(nib, codes.t().to(torch.int64))


def _case(N, K, group, dtype, seed=0):
    torch.manual_seed(seed)
    codes, scales, zeros = rtn_quantize(torch.randn(N, K), group=group)
    return pack(codes).cuda(), scales.cuda(), zeros.cuda()


@needs_marlin
@pytest.mark.parametrize("M", [1, 2, 8, 16, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("N,K", [(2048, 1536), (1536, 1536), (17920, 1536), (1536, 8960)])
@pytest.mark.parametrize("group", [64, 128])
def test_marlin_gemm_matches_reference(M, dtype, N, K, group):
    from turing.kernels.marlin import marlin_gemm, marlin_pack, marlin_supported
    q, s, z = _case(N, K, group, dtype)
    assert marlin_supported(N, K, group, dtype, q.device)
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    bias = torch.randn(N, device="cuda", dtype=dtype)
    ref = x.float() @ dequantize(q, s, z, torch.float32).T
    for b, r in ((None, ref), (bias, ref + bias.float())):
        y = marlin_gemm(x, marlin_pack(q, s, z, dtype, b))
        assert y.dtype == dtype and y.shape == (M, N)
        torch.testing.assert_close(y.float(), r, atol=1e-2 * r.abs().max().item(), rtol=1e-2)


@needs_marlin
def test_marlin_supported_shapes():
    from turing.kernels.marlin import marlin_supported
    assert marlin_supported(2048, 1536, 64, torch.float16, "cuda")
    assert not marlin_supported(1000, 1536, 64, torch.float16, "cuda")  # N % 64
    assert not marlin_supported(2048, 1600, 64, torch.float16, "cuda")  # K % 128
    assert not marlin_supported(2048, 128, 128, torch.float16, "cuda")  # one group per row
    assert not marlin_supported(2048, 1536, 32, torch.float16, "cuda")  # group not instantiated
    assert not marlin_supported(2048, 1536, 64, torch.float32, "cuda")


@needs_marlin
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("M", [1, 2, 8, 64, 2048])
def test_quant_linear_marlin_path_matches_triton(dtype, M):
    """QuantLinear routes MARLIN_MIN_ROWS..MARLIN_MAX_ROWS rows to Marlin; it agrees with the Triton int4 GEMM, and a
    fused q/k/v QuantLinear (fuse_linears) gets its own Marlin copy."""
    import torch.nn as nn
    from turing.kernels.triton_w4a16 import w4a16_gemm
    from turing.models.causal_lm import fuse_linears, fused_outputs
    from turing.models.quant import MARLIN_MAX_ROWS, MARLIN_MIN_ROWS, QuantLinear
    assert MARLIN_MIN_ROWS <= M <= MARLIN_MAX_ROWS
    torch.manual_seed(0)
    K = 1536
    parts = []
    for N in (1536, 256, 256):
        codes, s, z = rtn_quantize(torch.randn(N, K))
        parts.append(QuantLinear.from_codes(codes.cuda(), s.cuda(), z.cuda(), torch.randn(N, device="cuda", dtype=dtype)))
    lin = parts[0]
    x = torch.randn(M, K, device="cuda", dtype=dtype)
    y = lin(x)
    assert lin._marlin[dtype] is not None  # the Marlin path ran
    ref = w4a16_gemm(x, lin.qweight, lin.scales, lin.zeros, lin.group, lin.bias).float()
    torch.testing.assert_close(y.float(), ref, atol=1e-2 * ref.abs().max().item(), rtol=1e-2)
    owner = nn.Module()
    assert fuse_linears(owner, "_qkv", tuple(parts))
    outs = fused_outputs(owner, "_qkv", tuple(parts), x)
    fused = owner.__dict__["_qkv"][0]
    assert fused._marlin[dtype] is not None
    for p, o in zip(parts, outs):
        r = w4a16_gemm(x, p.qweight, p.scales, p.zeros, p.group, p.bias).float()
        torch.testing.assert_close(o.float(), r, atol=1e-2 * r.abs().max().item(), rtol=1e-2)


@needs_marlin
def test_marlin_copies_are_built_before_the_kv_pool_is_sized():
    """Every 4-bit layer a CUDA forward uses (fused q/k/v and gate/up, the unfused rest) has its Marlin copy, or its
    'unsupported' mark, once the KV pool exists: none is allocated later, out of memory the pool has taken."""
    from tests.test_hf_parity import _tiny
    from turing.models.hf_loader import RealHuggingFaceLoader
    from turing.models.quant import QuantLinear, quantize_model
    model = RealHuggingFaceLoader.convert_hf_model(_tiny("llama")).half().cuda()
    quantize_model(model, "rtn")
    model.config.kv_pool_gib = 0.01
    model.kv_pool()
    fused = [o.__dict__[k][0] for o in model.modules() for k in ("_qkv", "_gate_up") if k in o.__dict__]
    parts = {id(p) for o in model.modules() for k in ("_qkv", "_gate_up") if k in o.__dict__ for p in o.__dict__[k][1]}
    used = fused + [m for m in model.modules() if isinstance(m, QuantLinear) and id(m) not in parts]
    assert fused and used and all(torch.float16 in m._marlin for m in used)
