"""Fused norm, SiLU x up and RoPE kernels (speed plan P2.1) against the PyTorch code they replace (CUDA)."""
import pytest
import torch
import torch.nn.functional as F

from turing.core.rope import apply_rotary_pos_emb

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton fused ops")
DTYPES = [torch.float16, torch.bfloat16]


def _close(a, b):
    torch.testing.assert_close(a.float(), b.float(), atol=1e-3 * max(1.0, b.abs().max().item()), rtol=1e-2)


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("offset", [0.0, 1.0])
@pytest.mark.parametrize("D", [960, 1536, 3072])
def test_rms_norm_with_and_without_residual_matches_the_module(dtype, offset, D):
    from turing.kernels.triton_fused_ops import rms_norm
    from turing.models.causal_lm import RMSNorm
    torch.manual_seed(0)
    norm = RMSNorm(D, 1e-6, offset).cuda().to(dtype)
    norm.weight.data.normal_(0, 0.5).add_(0 if offset else 1)
    x = (torch.randn(3, 5, D, device="cuda") * 20).to(dtype)  # large values: the fp32 square must not overflow
    r = torch.randn(3, 5, D, device="cuda").to(dtype)
    with torch.no_grad():
        ref_s = x + r
        xf = ref_s.float()
        n = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (n * (offset + norm.weight.float())).to(dtype) if offset else n.to(dtype) * norm.weight
        y, s = rms_norm(x, norm.weight, 1e-6, offset, residual=r)
        assert torch.equal(s, ref_s)
        _close(y, ref)
        _close(rms_norm(ref_s, norm.weight, 1e-6, offset), ref)
        _close(norm(ref_s), ref)  # the module's own CUDA path takes the kernel


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
def test_silu_mul_on_views_of_one_fused_projection(dtype):
    from turing.kernels.triton_fused_ops import silu_mul
    torch.manual_seed(0)
    gu = torch.randn(4, 1, 2 * 8960, device="cuda", dtype=dtype) * 4
    g, u = gu.split(8960, dim=-1)
    _close(silu_mul(g, u), F.silu(g) * u)


@cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rot", [128, 64, 32])  # full, and partial rotary (Phi-2, GPT-NeoX)
@pytest.mark.parametrize("decode", [False, True])
def test_rope_matches_apply_rotary_pos_emb(dtype, rot, decode):
    from turing.kernels.triton_fused_ops import rope
    torch.manual_seed(0)
    B, H, Hkv, D = 3, 12, 2, 128
    L = 1 if decode else 37  # tiles of 16 positions at D = 128: two whole ones and a part
    fused = torch.randn(B, L, (H + 2 * Hkv) * D, device="cuda", dtype=dtype)  # q/k as views of a fused qkv
    q = fused[..., :H * D].view(B, L, H, D).transpose(1, 2)
    k = fused[..., H * D:(H + Hkv) * D].view(B, L, Hkv, D).transpose(1, 2)
    if decode:  # one position per row, as DecodeStep.freqs gives: [B, R] viewed [B, 1, 1, R]
        ang = torch.rand(B, rot, device="cuda") * 100
        cos, sin = ang.cos().view(B, 1, 1, rot), ang.sin().view(B, 1, 1, rot)
    else:
        ang = torch.rand(L, rot, device="cuda") * 100
        cos, sin = ang.cos(), ang.sin()
    rq, rk = apply_rotary_pos_emb(q, k, cos.to(dtype), sin.to(dtype))  # the kernel rounds as this does, in the dtype
    fq, fk = rope(q, k, cos, sin)
    for f, r in ((fq, rq), (fk, rk)):  # within one rounding step of the dtype at the output's magnitude
        torch.testing.assert_close(f.float(), r.float(), rtol=0,
                                   atol=torch.finfo(dtype).eps * max(1.0, r.abs().max().item()))
    tq, tk = rope(q, k, cos, sin, tokens_first=True)  # the same values, q's memory token by token
    assert torch.equal(tq, fq) and torch.equal(tk, fk) and tq.transpose(1, 2).is_contiguous()
