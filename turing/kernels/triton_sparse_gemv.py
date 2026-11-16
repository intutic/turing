"""
Input-sparse GEMV for TEAL-style activation sparsity (Liu et al., ICLR 2025): y = W x with most entries of x zero.
W is kept transposed, WT [K, N], so input channel k is one contiguous row; rows whose x_k is zero are never loaded
(masked loads issue no memory traffic), which is where batch-1 decode time goes. Parity: tests/test_activation_sparsity.py.
"""

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.autotune(configs=[triton.Config({"BLOCK_K": bk, "BLOCK_N": bn}, num_warps=w)
                              for bk in (32, 64) for bn in (128, 256) for w in (4, 8)], key=["K", "N"])
    @triton.jit
    def _sparse_gemv_kernel(X, WT, Y, K, N, s_wk, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_k, mask=offs_k < K, other=0.0)
            live = (x != 0) & (offs_k < K)
            w = tl.load(WT + offs_k[:, None] * s_wk + offs_n[None, :], mask=live[:, None] & (offs_n[None, :] < N),
                        other=0.0)
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0)
        tl.store(Y + offs_n, acc.to(Y.dtype.element_ty), mask=offs_n < N)


def sparse_gemv(x: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
    """x [K] (mostly zeros), wt [K, N] contiguous -> y [N] in x's dtype."""
    K, N = wt.shape
    y = torch.empty(N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _sparse_gemv_kernel[grid](x, wt, y, K, N, wt.stride(0))
    return y


if HAS_TRITON:
    @triton.autotune(configs=[triton.Config({"BLOCK_K": bk}, num_warps=w) for bk in (32, 64, 128) for w in (2, 4)],
                     key=["K", "N"])
    @triton.jit
    def _sparse_gemv_int4_kernel(X, WT, S, Z, Y, K, N, s_wk, s_sk, BLOCK_K: tl.constexpr):
        """One program = 64 outputs (one quantization group of every input row): bytes [K, 32] of packed codes."""
        pid = tl.program_id(0)
        offs_b = pid * 32 + tl.arange(0, 32)  # packed byte columns of this group
        acc_lo = tl.zeros((32,), dtype=tl.float32)
        acc_hi = tl.zeros((32,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_k, mask=offs_k < K, other=0.0).to(tl.float32)
            live = (x != 0) & (offs_k < K)
            b = tl.load(WT + offs_k[:, None] * s_wk + offs_b[None, :], mask=live[:, None], other=0)
            sc = tl.load(S + offs_k * s_sk + pid, mask=live, other=0.0).to(tl.float32)
            zp = tl.load(Z + offs_k * s_sk + pid, mask=live, other=0).to(tl.float32)
            xs = (x * sc)[:, None]
            acc_lo += tl.sum(((b & 0x0F).to(tl.float32) - zp[:, None]) * xs, axis=0)
            acc_hi += tl.sum((((b >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * xs, axis=0)
        n_even = pid * 64 + 2 * tl.arange(0, 32)
        tl.store(Y + n_even, acc_lo.to(Y.dtype.element_ty), mask=n_even < N)
        tl.store(Y + n_even + 1, acc_hi.to(Y.dtype.element_ty), mask=n_even + 1 < N)


def quantize_transposed_int4(w: torch.Tensor):
    """W [N, K] -> codes of W^T packed along N: [K, N / 2] uint8, scales/zeros [K, N / 64] (groups of 64 outputs per
    input row, so a zero input skips one contiguous row of bytes). Round to nearest, fp16 scales, uint8 zeros."""
    from ..models.quant import pack, rtn_quantize
    codes, scales, zeros = rtn_quantize(w.t().float().cpu(), 64)
    return pack(codes).to(w.device), scales.to(w.device), zeros.to(w.device)


def sparse_gemv_int4(x: torch.Tensor, wt_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """x [K] (mostly zeros) times the transposed int4 weight -> y [N] in x's dtype."""
    K, N = wt_packed.shape[0], wt_packed.shape[1] * 2
    y = torch.empty(N, device=x.device, dtype=x.dtype)
    _sparse_gemv_int4_kernel[(triton.cdiv(N, 64),)](x, wt_packed, scales, zeros, y, K, N, wt_packed.stride(0),
                                                    scales.stride(0))
    return y
