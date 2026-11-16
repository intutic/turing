"""
Fused decode over the paged KV pool (turing/core/kv_cache.py): one decode step for every running request is three
launches per layer.

1. append: each (row, KV head, K or V) program writes this step's token into the row's page (page table lookup),
   quantized as the pool stores it, and for K updates the page's Quest min/max.
2. decode: each (row, KV head, split) program serves the G query heads that share the KV head, walks the row's pages
   (all of them, or a Quest list of logical pages), dequantizes in registers and runs an online softmax. Every row
   splits its own length on the device, so the launch shape depends only on the batch, never on lengths.
3. reduce: merges the splits with a log-sum-exp.
Dot products are fp32; quantization uses IEEE division and round-half-to-even, matching the reference path exactly.
"""

from typing import Optional
import math
import torch

from ..core.kv_cache import INT4_GROUP, PAGE, KVPool

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = ["pool_decode_supported", "triton_pool_decode"]

_MODES = {"native": 0, "int8": 1, "int4": 2}
_ELEM = {}  # torch dtype of a native pool -> Triton element type


if HAS_TRITON:
    _ELEM.update({torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32})

    @triton.jit
    def _store_token(DST, DS, DZ, row, x, offs, D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr):
        """Write x [D] (fp32) as pool row `row` (= (page * Hkv + head) * 16 + slot) in the pool's format."""
        if MODE == 0:
            tl.store(DST + row * D + offs, x.to(DST.dtype.element_ty))
        elif MODE == 1:
            lo = tl.min(x, 0)
            scale = tl.maximum(libdevice.div_rn(tl.max(x, 0) - lo, 255.0), 1e-8)
            codes = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(x - lo, scale)), 0.0), 255.0)
            tl.store(DST + row * D + offs, codes.to(tl.uint8))
            tl.store(DS + row, scale.to(tl.float16))
            tl.store(DZ + row, lo.to(tl.float16))
        else:
            NG: tl.constexpr = D // GROUP
            xg = tl.reshape(x, (NG, GROUP))
            lo = tl.min(xg, 1)
            scale = tl.maximum(libdevice.div_rn(tl.max(xg, 1) - lo, 15.0), 1e-8)
            codes = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(xg - lo[:, None], scale[:, None])), 0.0), 15.0)
            even, odd = tl.split(tl.reshape(codes.to(tl.uint8), (D // 2, 2)))
            tl.store(DST + row * (D // 2) + tl.arange(0, D // 2), even | (odd << 4))
            g = tl.arange(0, NG)
            tl.store(DS + row * NG + g, scale.to(tl.float16))
            tl.store(DZ + row * NG + g, lo.to(tl.float16))

    @triton.jit
    def _pool_append_kernel(KV, PT, SLOTS, STARTS, K, V, KS, KZ, VS, VZ, KMIN, KMAX,
                            s_which, s_b, s_h, pt_stride,
                            HKV: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr,
                            TRACK: tl.constexpr):
        """KV [2, B, Hkv, D]: this step's K (after RoPE) and V of each row, written at position STARTS[b]."""
        b = tl.program_id(0)
        h = tl.program_id(1)
        which = tl.program_id(2)
        pos = tl.load(STARTS + b)
        page = tl.load(PT + tl.load(SLOTS + b).to(tl.int64) * pt_stride + pos // 16).to(tl.int64)
        row = (page * HKV + h) * 16 + pos % 16
        offs = tl.arange(0, D)
        x = tl.load(KV + which * s_which + b * s_b + h * s_h + offs).to(tl.float32)
        if which == 0:
            _store_token(K, KS, KZ, row, x, offs, D, MODE, GROUP)
            if TRACK:
                bo = (page * HKV + h) * D + offs
                xh = x.to(tl.float16).to(tl.float32)
                tl.store(KMIN + bo, tl.minimum(tl.load(KMIN + bo).to(tl.float32), xh).to(tl.float16))
                tl.store(KMAX + bo, tl.maximum(tl.load(KMAX + bo).to(tl.float32), xh).to(tl.float16))
        else:
            _store_token(V, VS, VZ, row, x, offs, D, MODE, GROUP)

    @triton.jit
    def _load_rows(X, XS, XZ, row, valid, offs_d, D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr):
        """Pool rows `row` [BLOCK_N] as fp32 [BLOCK_N, D]; invalid rows read as 0."""
        if MODE == 0:
            x = tl.load(X + row[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        elif MODE == 1:
            c = tl.load(X + row[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0).to(tl.float32)
            sc = tl.load(XS + row, mask=valid, other=0.0).to(tl.float32)
            z = tl.load(XZ + row, mask=valid, other=0.0).to(tl.float32)
            x = c * sc[:, None] + z[:, None]
        else:
            NG: tl.constexpr = D // GROUP
            byte = tl.load(X + row[:, None] * (D // 2) + (offs_d // 2)[None, :], mask=valid[:, None], other=0)
            c = ((byte >> ((offs_d % 2) * 4)[None, :]) & 0xF).to(tl.float32)
            g = row[:, None] * NG + (offs_d // GROUP)[None, :]
            sc = tl.load(XS + g, mask=valid[:, None], other=0.0).to(tl.float32)
            z = tl.load(XZ + g, mask=valid[:, None], other=0.0).to(tl.float32)
            x = c * sc + z
        return x

    @triton.jit
    def _pool_decode_kernel(Q, PT, SLOTS, STARTS, SEL, K, V, KS, KZ, VS, VZ, ACC, M, L,
                            s_qb, s_qh, pt_stride, s_sel_b, s_sel_h, n_sel, sm_scale,
                            G: tl.constexpr, BLOCK_G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                            MODE: tl.constexpr, GROUP: tl.constexpr, HAS_SEL: tl.constexpr,
                            N_SPLITS: tl.constexpr, HKV: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)
        length = tl.load(STARTS + b) + 1  # STARTS holds the length before this step's append
        pt_row = PT + tl.load(SLOTS + b).to(tl.int64) * pt_stride
        n_tok = n_sel * 16 if HAS_SEL else length
        per_split = tl.cdiv(tl.cdiv(n_tok, N_SPLITS), BLOCK_N) * BLOCK_N
        start = split * per_split
        end = tl.minimum(start + per_split, n_tok)
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        q = tl.load(Q + b * s_qb + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
        q = q.to(tl.float32)
        m_i = tl.full([BLOCK_G], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
        acc = tl.zeros([BLOCK_G, D], dtype=tl.float32)
        for i0 in range(start, end, BLOCK_N):
            i = i0 + tl.arange(0, BLOCK_N)
            inside = i < end
            if HAS_SEL:
                lp = tl.load(SEL + b * s_sel_b + h * s_sel_h + i // 16, mask=inside, other=-1)
                valid = inside & (lp >= 0) & (lp * 16 + i % 16 < length)
            else:
                lp = i // 16
                valid = inside
            page = tl.load(pt_row + lp, mask=valid, other=0).to(tl.int64)
            row = (page * HKV + h) * 16 + i % 16
            k = _load_rows(K, KS, KZ, row, valid, offs_d, D, MODE, GROUP)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_rows(V, VS, VZ, row, valid, offs_d, D, MODE, GROUP)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            m_i = m_new
        out_row = ((b * HKV + h) * N_SPLITS + split) * G + offs_g
        tl.store(M + out_row, m_i, mask=gmask)
        tl.store(L + out_row, l_i, mask=gmask)
        tl.store(ACC + out_row[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])

    @triton.jit
    def _kv_reduce_kernel(ACC, M, L, OUT, G: tl.constexpr, BLOCK_G: tl.constexpr, D: tl.constexpr,
                          N_SPLITS: tl.constexpr):
        """Merge the splits of one (row, KV head): OUT [B * Hkv * G, D] in the output dtype."""
        bh = tl.program_id(0)
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        m_max = tl.full([BLOCK_G], -1.0e30, dtype=tl.float32)
        for s in range(N_SPLITS):
            m_max = tl.maximum(m_max, tl.load(M + (bh * N_SPLITS + s) * G + offs_g, mask=gmask, other=-1.0e30))
        num = tl.zeros([BLOCK_G, D], dtype=tl.float32)
        den = tl.zeros([BLOCK_G], dtype=tl.float32)
        for s in range(N_SPLITS):
            row = (bh * N_SPLITS + s) * G + offs_g
            w = tl.exp(tl.load(M + row, mask=gmask, other=-1.0e30) - m_max)
            den += tl.load(L + row, mask=gmask, other=0.0) * w
            num += tl.load(ACC + row[:, None] * D + offs_d[None, :], mask=gmask[:, None], other=0.0) * w[:, None]
        tl.store(OUT + (bh * G + offs_g)[:, None] * D + offs_d[None, :], (num / den[:, None]).to(OUT.dtype.element_ty),
                 mask=gmask[:, None])


def pool_decode_supported(pool: KVPool, q: torch.Tensor) -> bool:
    """CUDA, a power-of-2 head_dim of at least 16, and a native pool dtype Triton can store."""
    D = pool.head_dim
    return (HAS_TRITON and q.is_cuda and D >= 16 and D & (D - 1) == 0
            and (pool.kv_cache_dtype != "native" or pool.dtype in _ELEM))


def triton_pool_decode(pool: KVPool, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       slots: torch.Tensor, starts: torch.Tensor, sel: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One decode step of B requests in one layer: q [B, H, 1, D], k, v [B, Hkv, 1, D] (K after RoPE), slots and
    starts [B] int32 on the device (page-table rows, lengths before this step), sel [B, Hkv, K] logical pages or
    None. Appends k, v to each row's pages (already reserved) and returns attention [B, H, 1, D]."""
    B, H, _, D = q.shape
    Hkv = pool.num_kv_heads
    G = H // Hkv
    mode = _MODES[pool.kv_cache_dtype]
    fields = [t[layer] for t in (pool.k, pool.v)]
    scales = [t[layer] for t in (pool.k_scale, pool.k_zero, pool.v_scale, pool.v_zero)] if mode else fields * 2
    bounds = [pool.kmin[layer], pool.kmax[layer]] if pool.track_pages else fields
    kv = torch.stack((k, v)).reshape(2, B, Hkv, D)
    pt = pool.page_table
    _pool_append_kernel[(B, Hkv, 2)](kv, pt, slots, starts, *fields, *scales, *bounds,
                                     kv.stride(0), kv.stride(1), kv.stride(2), pt.stride(0),
                                     HKV=Hkv, D=D, MODE=mode, GROUP=INT4_GROUP, TRACK=pool.track_pages)
    n_splits = max(1, min(32, 256 // (B * Hkv)))
    BLOCK_N = 64 if D <= 64 else 32  # fp32 K and V tiles must fit the L4's 99 KB of shared memory
    BLOCK_G = max(16, triton.next_power_of_2(G))
    acc = torch.empty(B * Hkv * n_splits * G, D, device=q.device, dtype=torch.float32)
    m = torch.empty(B * Hkv * n_splits * G, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    qc = q.reshape(B, H, D)
    sel_t = sel if sel is not None else slots
    _pool_decode_kernel[(B, Hkv, n_splits)](
        qc, pt, slots, starts, sel_t, *fields, *scales, acc, m, l,
        qc.stride(0), qc.stride(1), pt.stride(0),
        sel.stride(0) if sel is not None else 0, sel.stride(1) if sel is not None else 0,
        sel.shape[-1] if sel is not None else 0, 1.0 / math.sqrt(D),
        G=G, BLOCK_G=BLOCK_G, D=D, BLOCK_N=BLOCK_N, MODE=mode, GROUP=INT4_GROUP, HAS_SEL=sel is not None,
        N_SPLITS=n_splits, HKV=Hkv, num_stages=2,
    )
    out = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    _kv_reduce_kernel[(B * Hkv,)](acc, m, l, out, G=G, BLOCK_G=BLOCK_G, D=D, N_SPLITS=n_splits)
    return out.view(B, H, 1, D)
