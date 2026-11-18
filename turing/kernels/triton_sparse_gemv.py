"""
Input-sparse GEMV for TEAL-style activation sparsity (Liu et al., ICLR 2025): y = W x with most entries of x zero.
W is kept transposed, WT [K, N], so input channel k is one contiguous row; rows whose x_k is zero are never loaded
(masked loads issue no memory traffic), which is where batch-1 decode time goes. Parity: tests/test_activation_sparsity.py.
"""

import functools

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
    import inspect
    # Triton >= 3.2 can keep autotuning results on disk (its cache dir): a new process skips the tuning runs
    # (speed plan P7.2, cold start)
    _TUNE_CACHE = {"cache_results": True} if "cache_results" in inspect.signature(triton.autotune).parameters else {}
except ImportError:
    HAS_TRITON = False

@functools.lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _splits(n_blocks: int, k_blocks: int, device) -> int:
    """K splits so the grid has about 4 programs per SM: a batch-1 GEMV over a narrow output (2048 columns = 8-16
    column blocks) otherwise leaves most of the GPU idle while each program walks all of K."""
    sms = _sm_count(torch.device(device).index or 0)
    return max(1, min(k_blocks, -(-4 * sms // max(1, n_blocks))))


if HAS_TRITON:
    @triton.autotune(configs=[triton.Config({"BLOCK_K": bk, "BLOCK_N": bn}, num_warps=w)
                              for bk in (32, 64) for bn in (128, 256) for w in (4, 8)], key=["K", "N", "SPLIT"],
                     reset_to_zero=["Y"], **_TUNE_CACHE)
    @triton.jit
    def _sparse_gemv_kernel(X, WT, B, Y, K, N, s_wk, SPLIT, HAS_BIAS: tl.constexpr, DIRECT: tl.constexpr,
                            BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
        """Program (n block, k split): its share of K, atomically added into the fp32 output Y (the first split adds
        the bias); with one split (DIRECT) stored in Y's dtype instead."""
        pid, pid_k = tl.program_id(0), tl.program_id(1)
        k_per = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
        k_lo = pid_k * k_per
        k_hi = tl.minimum(k_lo + k_per, K)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_k, mask=offs_k < k_hi, other=0.0)
            live = (x != 0) & (offs_k < k_hi)
            w = tl.load(WT + offs_k[:, None] * s_wk + offs_n[None, :], mask=live[:, None] & (offs_n[None, :] < N),
                        other=0.0)
            acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[:, None], axis=0)
        if HAS_BIAS:
            acc += tl.load(B + offs_n, mask=(offs_n < N) & (pid_k == 0), other=0.0).to(tl.float32)
        if DIRECT:
            tl.store(Y + offs_n, acc.to(Y.dtype.element_ty), mask=offs_n < N)
        else:
            tl.atomic_add(Y + offs_n, acc, mask=offs_n < N)


def sparse_gemv(x: torch.Tensor, wt: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x [K] (mostly zeros), wt [K, N] contiguous, bias [N] or None -> y [N] in x's dtype."""
    K, N = wt.shape
    split = _splits(triton.cdiv(N, 128), triton.cdiv(K, 64), x.device)
    y = torch.empty(N, device=x.device, dtype=x.dtype) if split == 1 else \
        torch.zeros(N, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), split)
    _sparse_gemv_kernel[grid](x, wt, bias if bias is not None else y, y, K, N, wt.stride(0), split,
                              HAS_BIAS=bias is not None, DIRECT=split == 1)
    return y if split == 1 else y.to(x.dtype)


if HAS_TRITON:
    @triton.autotune(configs=[triton.Config({"BLOCK_K": bk}, num_warps=w) for bk in (32, 64, 128) for w in (2, 4)],
                     key=["K", "N", "SPLIT"], reset_to_zero=["Y"], **_TUNE_CACHE)
    @triton.jit
    def _sparse_gemv_int4_kernel(X, WT, S, Z, B, Y, K, N, s_wk, s_sk, SPLIT, HAS_BIAS: tl.constexpr,
                                 DIRECT: tl.constexpr, BLOCK_K: tl.constexpr):
        """Program (64 outputs = one quantization group of every input row, k split): bytes [k share, 32] of packed
        codes, atomically added into the fp32 output Y."""
        pid, pid_k = tl.program_id(0), tl.program_id(1)
        k_per = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
        k_lo = pid_k * k_per
        k_hi = tl.minimum(k_lo + k_per, K)
        offs_b = pid * 32 + tl.arange(0, 32)  # packed byte columns of this group
        acc_lo = tl.zeros((32,), dtype=tl.float32)
        acc_hi = tl.zeros((32,), dtype=tl.float32)
        for k0 in range(k_lo, k_hi, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_k, mask=offs_k < k_hi, other=0.0).to(tl.float32)
            live = (x != 0) & (offs_k < k_hi)
            b = tl.load(WT + offs_k[:, None] * s_wk + offs_b[None, :], mask=live[:, None], other=0)
            sc = tl.load(S + offs_k * s_sk + pid, mask=live, other=0.0).to(tl.float32)
            zp = tl.load(Z + offs_k * s_sk + pid, mask=live, other=0).to(tl.float32)
            xs = (x * sc)[:, None]
            acc_lo += tl.sum(((b & 0x0F).to(tl.float32) - zp[:, None]) * xs, axis=0)
            acc_hi += tl.sum((((b >> 4) & 0x0F).to(tl.float32) - zp[:, None]) * xs, axis=0)
        n_even = pid * 64 + 2 * tl.arange(0, 32)
        if HAS_BIAS:  # the first split adds the bias
            acc_lo += tl.load(B + n_even, mask=(n_even < N) & (pid_k == 0), other=0.0).to(tl.float32)
            acc_hi += tl.load(B + n_even + 1, mask=(n_even + 1 < N) & (pid_k == 0), other=0.0).to(tl.float32)
        if DIRECT:
            tl.store(Y + n_even, acc_lo.to(Y.dtype.element_ty), mask=n_even < N)
            tl.store(Y + n_even + 1, acc_hi.to(Y.dtype.element_ty), mask=n_even + 1 < N)
        else:
            tl.atomic_add(Y + n_even, acc_lo, mask=n_even < N)
            tl.atomic_add(Y + n_even + 1, acc_hi, mask=n_even + 1 < N)


def quantize_transposed_int4(w: torch.Tensor):
    """W [N, K] -> codes of W^T packed along N: [K, N / 2] uint8, scales/zeros [K, N / 64] (groups of 64 outputs per
    input row, so a zero input skips one contiguous row of bytes). Round to nearest, fp16 scales, uint8 zeros."""
    from ..models.quant import pack, rtn_quantize
    codes, scales, zeros = rtn_quantize(w.t().float().cpu(), 64)
    return pack(codes).to(w.device), scales.to(w.device), zeros.to(w.device)


def sparse_gemv_int4(x: torch.Tensor, wt_packed: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x [K] (mostly zeros) times the transposed int4 weight, plus bias [N] or None -> y [N] in x's dtype."""
    K, N = wt_packed.shape[0], wt_packed.shape[1] * 2
    split = _splits(triton.cdiv(N, 64), triton.cdiv(K, 64), x.device)
    y = torch.empty(N, device=x.device, dtype=x.dtype) if split == 1 else \
        torch.zeros(N, device=x.device, dtype=torch.float32)
    _sparse_gemv_int4_kernel[(triton.cdiv(N, 64), split)](x, wt_packed, scales, zeros, bias if bias is not None else y,
                                                          y, K, N, wt_packed.stride(0), scales.stride(0), split,
                                                          HAS_BIAS=bias is not None, DIRECT=split == 1)
    return y if split == 1 else y.to(x.dtype)
