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

__all__ = ["pool_decode_supported", "triton_pool_decode", "pool_dequant_supported", "triton_pool_dequant"]

_MODES = {"native": 0, "int8": 1, "int4": 2, "fp8": 3}
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
        elif MODE == 3:  # E4M3 bytes of x / the layer's scale (DS[0]), as kv_cache._fp8_encode
            y = tl.minimum(tl.maximum(libdevice.div_rn(x, tl.load(DS)), -448.0), 448.0)
            bits = y.to(tl.int32, bitcast=True)
            mag = bits & 0x7FFFFFFF
            normal = ((mag + 0x7FFFF + ((mag >> 20) & 1)) >> 20) - 960
            sub = libdevice.rint(tl.abs(y) * 512.0).to(tl.int32)
            code = tl.where(mag < 0x3C800000, sub, normal) | ((bits >> 24) & 0x80)
            tl.store(DST + row * D + offs, code.to(tl.uint8))
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
    def _pool_append_kernel(K_IN, V_IN, PT, SLOTS, STARTS, K, V, KS, KZ, VS, VZ, KMIN, KMAX,
                            s_kb, s_kh, s_vb, s_vh, pt_stride,
                            HKV: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr,
                            TRACK: tl.constexpr):
        """K_IN, V_IN [B, Hkv, (1,) D] (any batch and head strides): this step's K (after RoPE) and V of each row,
        written at position STARTS[b]."""
        b = tl.program_id(0)
        h = tl.program_id(1)
        which = tl.program_id(2)
        pos = tl.load(STARTS + b)
        page = tl.load(PT + tl.load(SLOTS + b).to(tl.int64) * pt_stride + pos // 16).to(tl.int64)
        row = (page * HKV + h) * 16 + pos % 16
        offs = tl.arange(0, D)
        if which == 0:
            x = tl.load(K_IN + b * s_kb + h * s_kh + offs).to(tl.float32)
            _store_token(K, KS, KZ, row, x, offs, D, MODE, GROUP)
            if TRACK:
                bo = (page * HKV + h) * D + offs
                xh = x.to(tl.float16).to(tl.float32)
                tl.store(KMIN + bo, tl.minimum(tl.load(KMIN + bo).to(tl.float32), xh).to(tl.float16))
                tl.store(KMAX + bo, tl.maximum(tl.load(KMAX + bo).to(tl.float32), xh).to(tl.float16))
        else:
            x = tl.load(V_IN + b * s_vb + h * s_vh + offs).to(tl.float32)
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
        elif MODE == 3:  # E4M3 bytes: below 8 multiples of 2^-9, else the fp32 with these 3 mantissa bits; times XS[0]
            b = tl.load(X + row[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0).to(tl.int32)
            mag = b & 0x7F
            x = tl.where(mag < 8, mag.to(tl.float32) * 0.001953125, ((mag + 960) << 20).to(tl.float32, bitcast=True))
            x = tl.where(b > 127, -x, x) * tl.load(XS)
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
    def _pool_decode_kernel(Q, PT, SLOTS, STARTS, SEL, K, V, KS, KZ, VS, VZ, ACC, M, L, OUT,
                            s_qb, s_qh, pt_stride, s_sel_b, s_sel_h, n_sel, sm_scale, softcap, window,
                            G: tl.constexpr, BLOCK_G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                            MODE: tl.constexpr, GROUP: tl.constexpr, HAS_SEL: tl.constexpr,
                            N_SPLITS: tl.constexpr, HKV: tl.constexpr, WINDOW: tl.constexpr,
                            SOFTCAP: tl.constexpr, IEEE: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)
        length = tl.load(STARTS + b) + 1  # STARTS holds the length before this step's append
        pt_row = PT + tl.load(SLOTS + b).to(tl.int64) * pt_stride
        n_tok = n_sel * 16 if HAS_SEL else length
        first = 0  # a sliding window sees keys j > (length - 1) - window only
        if WINDOW:
            first = tl.maximum(length - window, 0)
        per_split = tl.cdiv(tl.cdiv(n_tok - first, N_SPLITS), BLOCK_N) * BLOCK_N
        start = first + split * per_split
        end = tl.minimum(start + per_split, n_tok)
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        q = tl.load(Q + b * s_qb + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
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
            if IEEE:  # fp32 queries: IEEE fp32 products; fp16/bf16: tensor cores in that dtype, fp32 accumulation
                s = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee") * sm_scale
            else:
                s = tl.dot(q, tl.trans(k.to(q.dtype))) * sm_scale
            if SOFTCAP:  # Gemma 2: cap * tanh(s / cap)
                s = softcap * libdevice.tanh(s / softcap)
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_rows(V, VS, VZ, row, valid, offs_d, D, MODE, GROUP)
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
            m_i = m_new
        if N_SPLITS == 1:  # nothing to merge: the output directly, and no reduce launch
            o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
            tl.store(OUT + ((b * HKV + h) * G + offs_g)[:, None] * D + offs_d[None, :], o.to(OUT.dtype.element_ty),
                     mask=gmask[:, None])
        else:
            out_row = ((b * HKV + h) * N_SPLITS + split) * G + offs_g
            tl.store(M + out_row, m_i, mask=gmask)
            tl.store(L + out_row, l_i, mask=gmask)
            tl.store(ACC + out_row[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])

    @triton.jit
    def _kv_reduce_kernel(ACC, M, L, OUT, G: tl.constexpr, D: tl.constexpr, N_SPLITS: tl.constexpr,
                          BLOCK_S: tl.constexpr):
        """Merge the splits of one (row, KV head, query head of its group): OUT [B * Hkv * G, D] in the output dtype.
        Every split in one load: a loop over the splits took 7.4 us a launch at batch 1 on an L4 where this takes
        1.9 (32 splits; 4.2 against 2.2 at batch 8), a tenth of a millisecond of a Qwen2.5-1.5B step."""
        bh = tl.program_id(0)
        g = tl.program_id(1)
        s = tl.arange(0, BLOCK_S)
        live = s < N_SPLITS
        offs_d = tl.arange(0, D)
        row = (bh * N_SPLITS + s) * G + g
        m = tl.load(M + row, mask=live, other=-1.0e30)
        w = tl.exp(m - tl.max(m, 0))
        den = tl.sum(tl.load(L + row, mask=live, other=0.0) * w, 0)
        acc = tl.load(ACC + row[:, None] * D + offs_d[None, :], mask=live[:, None], other=0.0)
        tl.store(OUT + (bh * G + g) * D + offs_d, (tl.sum(acc * w[:, None], 0) / den).to(OUT.dtype.element_ty))

    @triton.jit
    def _pool_dequant_kernel(PHYS, X, XS, XZ, OUT, HKV: tl.constexpr, D: tl.constexpr, MODE: tl.constexpr,
                             GROUP: tl.constexpr):
        """Page PHYS[p], KV head h of a low-bit field -> OUT[p, h] ([16, D] in OUT's type)."""
        p = tl.program_id(0)
        h = tl.program_id(1)
        slots = tl.arange(0, 16)
        offs = tl.arange(0, D)
        row = (tl.load(PHYS + p).to(tl.int64) * HKV + h) * 16 + slots
        x = _load_rows(X, XS, XZ, row, slots >= 0, offs, D, MODE, GROUP)
        tl.store(OUT + ((p * HKV + h) * 16 + slots)[:, None] * D + offs[None, :], x.to(OUT.dtype.element_ty))

def pool_decode_supported(pool: KVPool, q: torch.Tensor) -> bool:
    """CUDA, a power-of-2 head_dim of at least 16, and a native pool dtype Triton can store."""
    D = pool.head_dim
    return (HAS_TRITON and q.is_cuda and D >= 16 and D & (D - 1) == 0
            and ((pool.kv_cache_dtype != "native" and not pool.native_layers) or pool.dtype in _ELEM))


def pool_dequant_supported(pool: KVPool) -> bool:
    D = pool.head_dim
    return HAS_TRITON and pool.device.type == "cuda" and D >= 16 and D & (D - 1) == 0 and pool.dtype in _ELEM


def triton_pool_dequant(pool: KVPool, layer: int, which: str, phys: torch.Tensor) -> torch.Tensor:
    """Pages phys [P] (physical page ids, int32 on the device) of a layer's low-bit K or V ("k" / "v") in the model
    dtype: [P, Hkv, 16, D], one launch (the reference, KVPool._decode_field, is a chain of elementwise passes over
    fp32 temporaries: about 1 ms a layer for a 14K-token prefix on an L4, a tenth of a 2,048-token prefill chunk)."""
    codes, scale, zero, storage = pool.layer_fields(layer, which)
    P, Hkv, D = phys.shape[0], pool.num_kv_heads, pool.head_dim
    out = torch.empty(P, Hkv, 16, D, device=phys.device, dtype=pool.dtype)
    _pool_dequant_kernel[(P, Hkv)](phys, codes, scale, zero, out, HKV=Hkv, D=D, MODE=_MODES[storage],
                                   GROUP=INT4_GROUP)
    return out


# (added below the kernels: Triton's compile cache keys a kernel by its line number too)
__all__ += ["pool_append_supported", "triton_pool_append"]


def pool_append_supported(pool: KVPool, layer: int) -> bool:
    """triton_pool_append serves this layer: low-bit storage on CUDA, no Quest bounds (the kernel updates a page's
    min/max from one program per token, which only a decode step's one token per page makes safe). A layer in the
    model dtype keeps KVPool.append: the kernel was no faster for it (one 512-token prompt on an L4)."""
    return pool_dequant_supported(pool) and not pool.track_pages and pool.layer_fields(layer, "k")[3] != "native"


def triton_pool_append(pool: KVPool, layer: int, cache, k: torch.Tensor, v: torch.Tensor,
                       slots: Optional[torch.Tensor] = None, starts: Optional[torch.Tensor] = None) -> None:
    """KVPool.append for a prefill chunk through the decode step's append kernel, one program per token, head and
    field: k, v [1, Hkv, L, D] (K after RoPE) at positions cache.length .. + L - 1 (pages reserved). A low-bit layer
    is quantized there instead of in the reference's passes over fp32 temporaries. With slots and starts (int32 [L]
    on the device: each token's page-table row and position) no cache is read, so the launch can be captured in a
    CUDA graph (PrefillGraphs)."""
    Hkv, D, L = pool.num_kv_heads, pool.head_dim, k.shape[2]
    kc, ks, kz, storage = pool.layer_fields(layer, "k")
    vc, vs, vz, _ = pool.layer_fields(layer, "v")
    kt, vt = k[0].transpose(0, 1), v[0].transpose(0, 1)  # [L, Hkv, D] views
    if kt.stride(-1) != 1 or vt.stride(-1) != 1:
        kt, vt = kt.contiguous(), vt.contiguous()
    if slots is None:
        slots = torch.full((L,), cache.slot, dtype=torch.int32, device=k.device)
        starts = torch.arange(cache.length, cache.length + L, dtype=torch.int32, device=k.device)
    pt, mode = pool.page_table, _MODES[storage]
    scales = [ks, kz, vs, vz] if mode else [kc, vc] * 2  # unused pointers for a layer in the model dtype
    _pool_append_kernel[(L, Hkv, 2)](kt, vt, pt, slots, starts, kc, vc, *scales, kc, vc,
                                     kt.stride(0), kt.stride(1), vt.stride(0), vt.stride(1), pt.stride(0),
                                     HKV=Hkv, D=D, MODE=mode, GROUP=INT4_GROUP, TRACK=False)


def triton_pool_decode(pool: KVPool, layer: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       slots: torch.Tensor, starts: torch.Tensor, sel: Optional[torch.Tensor] = None,
                       scale: Optional[float] = None, window: int = 0, softcap: float = 0.0,
                       n_splits: Optional[int] = None) -> torch.Tensor:
    """One decode step of B requests in one layer: q [B, H, 1, D], k, v [B, Hkv, 1, D] (K after RoPE), slots and
    starts [B] int32 on the device (page-table rows, lengths before this step), sel [B, Hkv, K] logical pages or
    None. Appends k, v to each row's pages (already reserved) and returns attention [B, H, 1, D]. scale defaults to
    1 / sqrt(D); window > 0 attends to the last `window` keys only; softcap > 0 caps scores as cap * tanh(s / cap)."""
    B, H, _, D = q.shape
    Hkv = pool.num_kv_heads
    G = H // Hkv
    kc, ks, kz, storage = pool.layer_fields(layer, "k")
    vc, vs, vz, _ = pool.layer_fields(layer, "v")
    mode = _MODES[storage]  # a layer the pool keeps in the model dtype runs as native
    fields = [kc, vc]
    scales = [ks, kz, vs, vz] if mode else fields * 2
    bounds = [pool.kmin[layer], pool.kmax[layer]] if pool.track_pages else fields
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        k, v = k.contiguous(), v.contiguous()
    pt = pool.page_table
    _pool_append_kernel[(B, Hkv, 2)](k, v, pt, slots, starts, *fields, *scales, *bounds,
                                     k.stride(0), k.stride(1), v.stride(0), v.stride(1), pt.stride(0),
                                     HKV=Hkv, D=D, MODE=mode, GROUP=INT4_GROUP, TRACK=pool.track_pages)
    if n_splits is None:  # one split (no reduce launch) once the batch alone fills the GPU
        n_splits = max(1, min(32, 256 // (B * Hkv)))
    BLOCK_N = 64 if D <= 64 else 32  # fp32 K and V tiles must fit the L4's 99 KB of shared memory
    BLOCK_G = max(16, triton.next_power_of_2(G))
    out = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    acc = torch.empty(B * Hkv * n_splits * G if n_splits > 1 else 1, D, device=q.device, dtype=torch.float32)
    m = torch.empty(acc.shape[0], device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    qc = q.reshape(B, H, D)
    sel_t = sel if sel is not None else slots
    _pool_decode_kernel[(B, Hkv, n_splits)](
        qc, pt, slots, starts, sel_t, *fields, *scales, acc, m, l, out,
        qc.stride(0), qc.stride(1), pt.stride(0),
        sel.stride(0) if sel is not None else 0, sel.stride(1) if sel is not None else 0,
        sel.shape[-1] if sel is not None else 0, scale if scale is not None else 1.0 / math.sqrt(D),
        float(softcap), int(window),
        G=G, BLOCK_G=BLOCK_G, D=D, BLOCK_N=BLOCK_N, MODE=mode, GROUP=INT4_GROUP, HAS_SEL=sel is not None,
        N_SPLITS=n_splits, HKV=Hkv, WINDOW=window > 0, SOFTCAP=softcap > 0, IEEE=q.dtype == torch.float32,
        num_stages=2,
    )
    if n_splits > 1:
        _kv_reduce_kernel[(B * Hkv, G)](acc, m, l, out, G=G, D=D, N_SPLITS=n_splits,
                                        BLOCK_S=triton.next_power_of_2(n_splits))
    return out.view(B, H, 1, D)
