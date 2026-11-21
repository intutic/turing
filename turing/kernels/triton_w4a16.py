"""
W4A16 GEMM in Triton: y = x W^T with W stored as packed unsigned 4-bit codes (two per byte, low nibble = even
column), an fp16 scale and a uint8 zero point per group of input columns: w = (q - zero) * scale. Each packed byte is
loaded once and split into its two nibbles, which multiply the even and the odd columns of x. Tiles are autotuned.
Used by turing.models.quant.QuantLinear for small row counts (decode); see tests/test_w4a16.py for parity.

Two launch shapes. Direct (the default when the output has enough column blocks to fill the GPU): each program walks
all of K for its output tile and stores it in x's dtype with the bias added, so a layer is one launch. Split-K (narrow
outputs): programs share K and add into an fp32 buffer, then a cast and the bias add follow (three more launches).
"""

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
    triton = None
    tl = None

if HAS_TRITON:
    _CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s)
                for bm in (16, 32) for bn in (64, 128) for bk in (64,) for w in (4,) for s in (2, 3)]

    @triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "SPLIT"], reset_to_zero=["Y"], **_TUNE_CACHE)
    @triton.jit
    def _w4a16_kernel(X, W, S, Z, Y, M, N, K,
                      s_xm, s_xk, s_wn, s_wk, s_sn, s_sg, s_ym, s_yn, SPLIT,
                      GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """Program (m block, n block, k split): its share of K, atomically added into the fp32 output Y."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_k = tl.program_id(2)
        k_per = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
        k_lo = pid_k * k_per
        k_hi = tl.minimum(k_lo + k_per, K)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_b = tl.arange(0, BLOCK_K // 2)  # packed bytes of one K tile
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        mask_m, mask_n = offs_m < M, offs_n < N
        for k0 in range(k_lo, k_hi, BLOCK_K):
            kb = k0 // 2 + offs_b
            mask_k = kb < k_hi // 2
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
        tl.atomic_add(Y + offs_m[:, None] * s_ym + offs_n[None, :] * s_yn, acc,
                      mask=mask_m[:, None] & mask_n[None, :])


if HAS_TRITON:
    # 16-row tiles for decode at batch 1-16; 32 and 64 rows let batches of up to 64 read each weight tile once
    # (speed plan P6.1); autotuned per (M, N, K), results kept on disk where Triton allows (_TUNE_CACHE)
    _DIRECT_CONFIGS = [triton.Config({"BLOCK_M": 16, "BLOCK_N": bn, "BLOCK_K": 64}, num_warps=w, num_stages=s)
                       for bn in (16, 32, 64) for w in (2, 4) for s in (2, 3, 4)] + \
                      [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 64}, num_warps=4, num_stages=s)
                       for bm in (32, 64) for bn in (32, 64, 128) for s in (2, 3)]

    def _rows_prune(configs, named_args, **kwargs):
        """Tiles no taller than M rounded up to a power of two (at least 16): decode tries only 16-row tiles."""
        cap = max(16, triton.next_power_of_2(named_args["M"]))
        return [c for c in configs if c.kwargs["BLOCK_M"] <= cap] or configs

    @triton.autotune(configs=_DIRECT_CONFIGS, key=["M", "N", "K"], prune_configs_by={"early_config_prune": _rows_prune},
                     **_TUNE_CACHE)
    @triton.jit
    def _w4a16_direct_kernel(X, W, S, Z, B, Y, M, N, K, s_xm, s_xk, s_wn, s_wk, s_sn, s_sg, s_ym, s_yn,
                             HAS_BIAS: tl.constexpr, GROUP: tl.constexpr, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        """Program (m block, n block): all of K, then bias and a store in Y's dtype (no zero-fill, atomics or cast)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_b = tl.arange(0, BLOCK_K // 2)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        mask_m, mask_n = offs_m < M, offs_n < N
        for k0 in range(0, K, BLOCK_K):
            kb = k0 // 2 + offs_b
            mask_k = kb < K // 2
            wb = tl.load(W + offs_n[:, None] * s_wn + kb[None, :] * s_wk, mask=mask_n[:, None] & mask_k[None, :],
                         other=0)
            g = k0 // GROUP
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
        if HAS_BIAS:
            acc += tl.load(B + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        tl.store(Y + offs_m[:, None] * s_ym + offs_n[None, :] * s_yn, acc.to(Y.dtype.element_ty),
                 mask=mask_m[:, None] & mask_n[None, :])


if HAS_TRITON:
    _GEMV_CONFIGS = [triton.Config({"BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w)
                     for bn in (2, 4, 8, 16) for bk in (256, 512, 1024) for w in (2, 4, 8) if bn * bk <= 8192]

    @triton.autotune(configs=_GEMV_CONFIGS, key=["M", "N", "K"], **_TUNE_CACHE)
    @triton.jit
    def _w4a16_gemv_kernel(X, W, S, Z, B, Y, M, N, K, s_xm, s_xk, s_wn, s_wk, s_sn, s_sg, s_ym,
                           HAS_BIAS: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
        """Program (n block, row m): BLOCK_N output columns of one row of x, reduced in registers over all of K. No
        tensor-core tile of which one row would be used, no K split, no fp32 buffer or cast launch. A step covers
        BLOCK_K / GROUP whole quantization groups ([rows, groups, bytes of a group]): each weight row gives
        BLOCK_K / 2 contiguous bytes and one scale and zero per group."""
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_g = tl.arange(0, BLOCK_K // GROUP)
        offs_b = tl.arange(0, GROUP // 2)
        acc = tl.zeros((BLOCK_N, BLOCK_K // GROUP, GROUP // 2), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            g = k0 // GROUP + offs_g
            kb = g[:, None] * (GROUP // 2) + offs_b[None, :]
            mask_k = kb < K // 2
            wb = tl.load(W + offs_n[:, None, None] * s_wn + kb[None, :, :] * s_wk,
                         mask=mask_n[:, None, None] & mask_k[None, :, :], other=0)
            mask_g = mask_n[:, None] & (g < K // GROUP)[None, :]
            sc = tl.load(S + offs_n[:, None] * s_sn + g[None, :] * s_sg, mask=mask_g, other=0.0).to(tl.float32)
            zp = tl.load(Z + offs_n[:, None] * s_sn + g[None, :] * s_sg, mask=mask_g, other=0).to(tl.float32)
            xe = tl.load(X + m * s_xm + (2 * kb) * s_xk, mask=mask_k, other=0.0).to(tl.float32)
            xo = tl.load(X + m * s_xm + (2 * kb + 1) * s_xk, mask=mask_k, other=0.0).to(tl.float32)
            lo = (wb & 0x0F).to(tl.float32) - zp[:, :, None]
            hi = ((wb >> 4) & 0x0F).to(tl.float32) - zp[:, :, None]
            acc += (lo * xe[None, :, :] + hi * xo[None, :, :]) * sc[:, :, None]
        y = tl.sum(tl.sum(acc, 2), 1)
        if HAS_BIAS:
            y += tl.load(B + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(Y + m * s_ym + offs_n, y.to(Y.dtype.element_ty), mask=mask_n)


GEMV_MAX_ROWS = 1  # each row re-reads the weights: at 2-4 rows the tiled kernel was 3-5x faster (results_w4a16_crossover_ckpt4_cuda.json)


def w4a16_gemv(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group: int = 64,
               bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """As w4a16_gemm, one program per (block of output columns, row): for a few rows (decode), where a matrix-multiply
    tile wastes all but one of its rows. At M = 1 the tiled kernel read Qwen2.5-1.5B's wide layers at about 97 GB/s of
    the L4's 262 (results_decode_profile_ckpt2_cuda.json)."""
    M, K = x.shape
    N = qweight.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
    _w4a16_gemv_kernel[grid](x, qweight, scales, zeros, bias if bias is not None else y, y, M, N, K,
                             x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
                             scales.stride(0), scales.stride(1), y.stride(0), HAS_BIAS=bias is not None, GROUP=group)
    return y


def w4a16_gemm(x: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
               group: int = 64, bias: Optional[torch.Tensor] = None, split: Optional[int] = None) -> torch.Tensor:
    """x [M, K] fp16/bf16, qweight [N, K / 2] uint8, scales [N, K / group] fp16, zeros [N, K / group] uint8, bias [N]
    or None -> y [M, N] in x's dtype. split: K splits (1 = the direct kernel); None picks: the GEMV kernel up to
    GEMV_MAX_ROWS rows, else by the output's width."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available")
    M, K = x.shape
    N = qweight.shape[0]
    if group % 64:
        raise ValueError("the int4 kernel needs a group size that is a multiple of 64")
    from .triton_sparse_gemv import _sm_count, _splits
    if split is None and M <= GEMV_MAX_ROWS:
        return w4a16_gemv(x, qweight, scales, zeros, group, bias)
    if split is None:  # direct when 32-column tiles give every SM about 4 programs; else share K across programs
        # (L4, results_w4a16_crossover_ckpt2_cuda.json: at 1 program per SM the direct kernel took 27 us on a 2048x2048
        # layer at M = 1 against split-K's 11)
        split = 1 if triton.cdiv(M, 16) * triton.cdiv(N, 32) >= 4 * _sm_count(x.device.index or 0) else \
            _splits(triton.cdiv(M, 16) * triton.cdiv(N, 64), triton.cdiv(K, 64), x.device)
    if split == 1:
        y = torch.empty(M, N, device=x.device, dtype=x.dtype)
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
        _w4a16_direct_kernel[grid](x, qweight, scales, zeros, bias if bias is not None else y, y, M, N, K,
                                   x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
                                   scales.stride(0), scales.stride(1), y.stride(0), y.stride(1),
                                   HAS_BIAS=bias is not None, GROUP=group)
        return y
    y = torch.zeros(M, N, device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]), split)
    _w4a16_kernel[grid](x, qweight, scales, zeros, y, M, N, K,
                        x.stride(0), x.stride(1), qweight.stride(0), qweight.stride(1),
                        scales.stride(0), scales.stride(1), y.stride(0), y.stride(1), split, GROUP=group)
    y = y.to(x.dtype)
    return y if bias is None else y + bias


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
