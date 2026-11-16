"""
W4A16 GEMM in Triton: y = x W^T with W stored as packed unsigned 4-bit codes (two per byte, low nibble = even
column), an fp16 scale and a uint8 zero point per group of input columns: w = (q - zero) * scale. Each packed byte is
loaded once and split into its two nibbles, which multiply the even and the odd columns of x. Tiles are autotuned.
Used by turing.models.quant.QuantLinear for small row counts (decode); see tests/test_w4a16.py for parity.
"""

from typing import Optional
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    triton = None
    tl = None

if HAS_TRITON:
    _CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
                for bm in (16, 32) for bn in (64, 128) for bk in (64,) for w in (4,) for s in (2, 3)]

    @triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _w4a16_kernel(X, W, S, Z, Y, M, N, K,
                      s_xm, s_xk, s_wn, s_wk, s_sn, s_sg, s_ym, s_yn,
                      GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_b = tl.arange(0, BLOCK_K // 2)  # packed bytes of one K tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        mask_m, mask_n = offs_m < M, offs_n < N
        for k0 in range(0, K, BLOCK_K):
            kb = k0 // 2 + offs_b
            mask_k = kb < K // 2
            wb = tl.load(W + offs_n[:, None] * s_wn + kb[None, :] * s_wk, mask=mask_n[:, None] & mask_k[None, :],
                         other=0)
            g = k0 // GROUP  # BLOCK_K divides GROUP: one scale and zero per row of this tile
            sc = tl.load(S + offs_n * s_sn + g * s_sg, mask=mask_n, other=0.0).to(tl.float32)
            zp = tl.load(Z + offs_n * s_sn + g * s_sg, mask=mask_n, other=0).to(tl.float32)
            lo = ((wb & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
            hi = (((wb >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * sc[:, None]
            x_even = tl.load(X + offs_m[:, None] * s_xm + (2 * kb)[None, :] * s_xk,
                             mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            x_odd = tl.load(X + offs_m[:, None] * s_xm + (2 * kb + 1)[None, :] * s_xk,
                            mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(x_even, tl.trans(lo.to(x_even.dtype)))
            acc += tl.dot(x_odd, tl.trans(hi.to(x_odd.dtype)))
        tl.store(Y + offs_m[:, None] * s_ym + offs_n[None, :] * s_yn, acc.to(Y.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


def w4a16_gemm(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
               group: int = 64) -> torch.Tensor:
    """x [M, K] fp16/bf16, qweight [N, K / 2] uint8, scales [N, K / group] fp16, zeros [N, K / group] uint8 ->
    y [M, N] in x's dtype."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, K = x.shape
    N = qweight.shape[0]
    if group % 64:
        raise ValueError("the int4 kernel needs a group size that is a multiple of 64")
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _w4a16_kernel[grid](x, qweight, scales, zeros, y, M, N, K,
                        x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
                        scales.stride(0), scales.stride(1), y.stride(0), y.stride(1), GROUP=group)
    return y


if HAS_TRITON:
    @triton.jit
    def _dequant_kernel(W, S, Z, OUT, N, K, s_wn, s_sn, s_on, GROUP: tl.constexpr, BLOCK_B: tl.constexpr):
        """One program = one output row block of packed bytes: OUT[n, 2b], OUT[n, 2b+1] = (nibbles - zero) * scale."""
        n = tl.program_id(0)
        offs_b = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        mask = offs_b < K // 2
        b = tl.load(W + n * s_wn + offs_b, mask=mask, other=0)
        g = (2 * offs_b) // GROUP
        sc = tl.load(S + n * s_sn + g, mask=mask, other=0.0).to(tl.float32)
        zp = tl.load(Z + n * s_sn + g, mask=mask, other=0).to(tl.float32)
        lo = ((b & 0x0F).to(tl.float32) - zp) * sc
        hi = (((b >> 4) & 0x0F).to(tl.float32) - zp) * sc
        tl.store(OUT + n * s_on + 2 * offs_b, lo.to(OUT.dtype.element_ty), mask=mask)
        tl.store(OUT + n * s_on + 2 * offs_b + 1, hi.to(OUT.dtype.element_ty), mask=mask)


def dequantize_cuda(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, out: torch.Tensor,
                    group: int = 64) -> torch.Tensor:
    """Packed int4 [N, K / 2] -> out [N, K] (fp16/bf16) in one memory-bound pass (QuantLinear's large-M path)."""
    N, Kh = qweight.shape
    BLOCK_B = 512
    _dequant_kernel[(N, triton.cdiv(Kh, BLOCK_B))](qweight, scales, zeros, out, N, Kh * 2, qweight.stride(0),
                                                   scales.stride(0), out.stride(0), GROUP=group, BLOCK_B=BLOCK_B)
    return out
