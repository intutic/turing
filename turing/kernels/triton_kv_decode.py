"""
Fused split-K decode attention over a LayerKVCache (turing/core/kv_cache.py) for one request.

One program per (KV head, split): it serves the G query heads that share the KV head, reads that head's K and V
once, dequantizes them in registers (native, int8 per token, or int4 per group of 32), and runs an online softmax.
An optional list of token positions per KV head (Quest page selection) replaces the contiguous range. The partial
results of the splits are merged with a log-sum-exp in PyTorch. Dot products are fp32 (decode is bandwidth bound).
"""

from typing import Optional
import math
import torch

from ..core.kv_cache import INT4_GROUP, LayerKVCache

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = ["kv_decode_supported", "triton_kv_decode", "batched_decode_supported", "build_decode_tables",
           "triton_kv_decode_batch"]

_MODES = {"native": 0, "int8": 1, "int4": 2}
_ELEM_DTYPES = {}  # torch dtype of a native cache -> Triton element type (filled when Triton is present)


if HAS_TRITON:
    _ELEM_DTYPES.update({torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32})

    @triton.jit
    def _load_rows(X, XS, XZ, base, sbase, tok, valid, s_t, ss_t, offs_d,
                   MODE: tl.constexpr, GROUP: tl.constexpr):
        """Rows `tok` of one KV head as fp32 [BLOCK_N, D]; invalid rows are 0."""
        if MODE == 0:
            x = tl.load(X + base + tok[:, None] * s_t + offs_d[None, :], mask=valid[:, None], other=0.0)
            x = x.to(tl.float32)
        elif MODE == 1:
            c = tl.load(X + base + tok[:, None] * s_t + offs_d[None, :], mask=valid[:, None], other=0).to(tl.float32)
            sc = tl.load(XS + sbase + tok * ss_t, mask=valid, other=0.0).to(tl.float32)
            z = tl.load(XZ + sbase + tok * ss_t, mask=valid, other=0.0).to(tl.float32)
            x = c * sc[:, None] + z[:, None]
        else:
            byte = tl.load(X + base + tok[:, None] * s_t + (offs_d // 2)[None, :], mask=valid[:, None], other=0)
            c = ((byte >> ((offs_d % 2) * 4)[None, :]) & 0xF).to(tl.float32)
            g = (offs_d // GROUP)[None, :]
            sc = tl.load(XS + sbase + tok[:, None] * ss_t + g, mask=valid[:, None], other=0.0).to(tl.float32)
            z = tl.load(XZ + sbase + tok[:, None] * ss_t + g, mask=valid[:, None], other=0.0).to(tl.float32)
            x = c * sc + z
        return x

    @triton.jit
    def _kv_decode_kernel(
        Q, K, V, KS, KZ, VS, VZ, POS, ACC, M, L,
        length, n_tokens, tokens_per_split, sm_scale,
        s_qh, s_h, s_t, ss_h, ss_t, s_pos_h,
        G: tl.constexpr, BLOCK_G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
        MODE: tl.constexpr, GROUP: tl.constexpr, HAS_POS: tl.constexpr, N_SPLITS: tl.constexpr,
    ):
        h = tl.program_id(0)
        split = tl.program_id(1)
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        q = tl.load(Q + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
        q = q.to(tl.float32)

        m_i = tl.full([BLOCK_G], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
        acc = tl.zeros([BLOCK_G, D], dtype=tl.float32)
        base = h * s_h
        sbase = h * ss_h
        start = split * tokens_per_split
        end = tl.minimum(start + tokens_per_split, n_tokens)
        for n0 in range(start, end, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            nmask = offs_n < end
            if HAS_POS:
                tok = tl.load(POS + h * s_pos_h + offs_n, mask=nmask, other=0)
                valid = nmask & (tok < length)
            else:
                tok = offs_n
                valid = nmask
            k = _load_rows(K, KS, KZ, base, sbase, tok, valid, s_t, ss_t, offs_d, MODE, GROUP)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_rows(V, VS, VZ, base, sbase, tok, valid, s_t, ss_t, offs_d, MODE, GROUP)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            m_i = m_new

        row = (h * N_SPLITS + split) * G + offs_g
        tl.store(M + row, m_i, mask=gmask)
        tl.store(L + row, l_i, mask=gmask)
        tl.store(ACC + row[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])


def kv_decode_supported(q: torch.Tensor, cache: LayerKVCache) -> bool:
    D = q.shape[-1]
    return (HAS_TRITON and q.is_cuda and q.shape[0] == 1 and q.shape[2] == 1 and cache.length > 0
            and D >= 16 and D & (D - 1) == 0)


def triton_kv_decode(q: torch.Tensor, cache: LayerKVCache, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
    """q [1, H, 1, D] (after RoPE) over the cache, or over token positions pos [1, Hkv, N] -> [1, H, 1, D]."""
    _, H, _, D = q.shape
    s = cache.store
    k, v = s["k"], s["v"]
    Hkv = k.shape[1]
    G = H // Hkv
    mode = _MODES[cache.kv_cache_dtype]
    ks, kz, vs, vz = (s["k_scale"], s["k_zero"], s["v_scale"], s["v_zero"]) if mode else (k, k, v, v)
    n_tokens = cache.length if pos is None else pos.shape[-1]
    BLOCK_N = 64 if D <= 64 else 32  # fp32 K and V tiles must fit the L4's 99 KB of shared memory
    n_splits = max(1, min(math.ceil(n_tokens / 256), max(1, 128 // Hkv)))
    tokens_per_split = math.ceil(math.ceil(n_tokens / n_splits) / BLOCK_N) * BLOCK_N
    n_splits = math.ceil(n_tokens / tokens_per_split)

    qc = q.reshape(H, D).contiguous()
    acc = torch.empty(Hkv * n_splits * G, D, device=q.device, dtype=torch.float32)
    m = torch.empty(Hkv * n_splits * G, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    pos_t = pos.reshape(Hkv, -1).to(torch.int32).contiguous() if pos is not None else qc
    _kv_decode_kernel[(Hkv, n_splits)](
        qc, k, v, ks, kz, vs, vz, pos_t, acc, m, l,
        cache.length, n_tokens, tokens_per_split, 1.0 / math.sqrt(D),
        qc.stride(0), k.stride(1), k.stride(2), ks.stride(1), ks.stride(2), pos_t.stride(0),
        G=G, BLOCK_G=max(16, triton.next_power_of_2(G)), D=D, BLOCK_N=BLOCK_N,
        MODE=mode, GROUP=INT4_GROUP, HAS_POS=pos is not None, N_SPLITS=n_splits, num_stages=2,
    )
    acc = acc.view(Hkv, n_splits, G, D)
    m = m.view(Hkv, n_splits, G)
    l = l.view(Hkv, n_splits, G)
    w = torch.exp(m - m.amax(1, keepdim=True))
    out = (acc * w.unsqueeze(-1)).sum(1) / (l * w).sum(1).unsqueeze(-1)
    return out.reshape(1, H, 1, D).to(q.dtype)


# ---------------------------------------------------------------------------------------------------------------
# Batched decode across requests: one append, one decode and one reduce launch per layer for the whole batch.
# Each row's cache stays its own tensors; a device table [B, 8] carries their data pointers, length and capacity.
# ---------------------------------------------------------------------------------------------------------------

if HAS_TRITON:
    from triton.language.extra import libdevice

    @triton.jit
    def _row_ptr(TABLE, b, field, ELEM: tl.constexpr):
        return tl.load(TABLE + b * 8 + field).to(tl.pointer_type(ELEM))

    @triton.jit
    def _kv_append_batch_kernel(KV, TABLE, s_which, s_b, s_h,
                                D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr, ELEM: tl.constexpr):
        """KV [2, B, Hkv, D] (K after RoPE, V) -> each row's cache at its position, quantized as stored."""
        b = tl.program_id(0)
        h = tl.program_id(1)
        which = tl.program_id(2)  # 0 = K, 1 = V
        pos = tl.load(TABLE + b * 8 + 6)
        cap = tl.load(TABLE + b * 8 + 7)
        offs = tl.arange(0, D)
        x = tl.load(KV + which * s_which + b * s_b + h * s_h + offs).to(tl.float32)
        if MODE == 0:
            dst = _row_ptr(TABLE, b, which, ELEM)
            tl.store(dst + (h * cap + pos) * D + offs, x.to(ELEM))
        elif MODE == 1:
            lo = tl.min(x, 0)
            scale = tl.maximum(libdevice.div_rn(tl.max(x, 0) - lo, 255.0), 1e-8)  # IEEE division, as torch
            codes = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(x - lo, scale)), 0.0), 255.0).to(tl.uint8)
            tl.store(_row_ptr(TABLE, b, which, tl.uint8) + (h * cap + pos) * D + offs, codes)
            tl.store(_row_ptr(TABLE, b, 2 + 2 * which, tl.float16) + h * cap + pos, scale.to(tl.float16))
            tl.store(_row_ptr(TABLE, b, 3 + 2 * which, tl.float16) + h * cap + pos, lo.to(tl.float16))
        else:
            NG: tl.constexpr = D // GROUP
            xg = tl.reshape(x, (NG, GROUP))
            lo = tl.min(xg, 1)
            scale = tl.maximum(libdevice.div_rn(tl.max(xg, 1) - lo, 15.0), 1e-8)
            codes = tl.minimum(tl.maximum(libdevice.rint(libdevice.div_rn(xg - lo[:, None], scale[:, None])), 0.0), 15.0)
            even, odd = tl.split(tl.reshape(codes.to(tl.uint8), (D // 2, 2)))
            packed = even | (odd << 4)
            tl.store(_row_ptr(TABLE, b, which, tl.uint8) + (h * cap + pos) * (D // 2) + tl.arange(0, D // 2), packed)
            g = tl.arange(0, NG)
            tl.store(_row_ptr(TABLE, b, 2 + 2 * which, tl.float16) + (h * cap + pos) * NG + g, scale.to(tl.float16))
            tl.store(_row_ptr(TABLE, b, 3 + 2 * which, tl.float16) + (h * cap + pos) * NG + g, lo.to(tl.float16))

    @triton.jit
    def _load_batch_rows(TABLE, b, which, h, cap, tok, valid, offs_d,
                         D: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr, ELEM: tl.constexpr):
        """Rows `tok` of KV head h of K (which=0) or V (which=1) of row b, as fp32 [BLOCK_N, D]."""
        if MODE == 0:
            x = tl.load(_row_ptr(TABLE, b, which, ELEM) + (h * cap + tok)[:, None] * D + offs_d[None, :],
                        mask=valid[:, None], other=0.0).to(tl.float32)
        elif MODE == 1:
            c = tl.load(_row_ptr(TABLE, b, which, tl.uint8) + (h * cap + tok)[:, None] * D + offs_d[None, :],
                        mask=valid[:, None], other=0).to(tl.float32)
            sc = tl.load(_row_ptr(TABLE, b, 2 + 2 * which, tl.float16) + h * cap + tok, mask=valid, other=0.0)
            z = tl.load(_row_ptr(TABLE, b, 3 + 2 * which, tl.float16) + h * cap + tok, mask=valid, other=0.0)
            x = c * sc.to(tl.float32)[:, None] + z.to(tl.float32)[:, None]
        else:
            NG: tl.constexpr = D // GROUP
            row = (h * cap + tok)[:, None]
            byte = tl.load(_row_ptr(TABLE, b, which, tl.uint8) + row * (D // 2) + (offs_d // 2)[None, :],
                           mask=valid[:, None], other=0)
            c = ((byte >> ((offs_d % 2) * 4)[None, :]) & 0xF).to(tl.float32)
            g = (offs_d // GROUP)[None, :]
            sc = tl.load(_row_ptr(TABLE, b, 2 + 2 * which, tl.float16) + row * NG + g, mask=valid[:, None], other=0.0)
            z = tl.load(_row_ptr(TABLE, b, 3 + 2 * which, tl.float16) + row * NG + g, mask=valid[:, None], other=0.0)
            x = c * sc.to(tl.float32) + z.to(tl.float32)
        return x

    @triton.jit
    def _kv_decode_batch_kernel(Q, TABLE, ACC, M, L, s_qb, s_qh, tokens_per_split, sm_scale,
                                G: tl.constexpr, BLOCK_G: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
                                MODE: tl.constexpr, GROUP: tl.constexpr, ELEM: tl.constexpr,
                                N_SPLITS: tl.constexpr, HKV: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        split = tl.program_id(2)
        length = tl.load(TABLE + b * 8 + 6) + 1  # the table holds the length before this step's append
        cap = tl.load(TABLE + b * 8 + 7)
        offs_g = tl.arange(0, BLOCK_G)
        offs_d = tl.arange(0, D)
        gmask = offs_g < G
        q = tl.load(Q + b * s_qb + (h * G + offs_g)[:, None] * s_qh + offs_d[None, :], mask=gmask[:, None], other=0.0)
        q = q.to(tl.float32)
        m_i = tl.full([BLOCK_G], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
        acc = tl.zeros([BLOCK_G, D], dtype=tl.float32)
        start = split * tokens_per_split
        end = tl.minimum(start + tokens_per_split, length)
        for n0 in range(start, end, BLOCK_N):
            tok = n0 + tl.arange(0, BLOCK_N)
            valid = tok < end
            k = _load_batch_rows(TABLE, b, 0, h, cap, tok, valid, offs_d, D, MODE, GROUP, ELEM)
            s = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_batch_rows(TABLE, b, 1, h, cap, tok, valid, offs_d, D, MODE, GROUP, ELEM)
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            m_i = m_new
        row = ((b * HKV + h) * N_SPLITS + split) * G + offs_g
        tl.store(M + row, m_i, mask=gmask)
        tl.store(L + row, l_i, mask=gmask)
        tl.store(ACC + row[:, None] * D + offs_d[None, :], acc, mask=gmask[:, None])

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
        out = num / den[:, None]
        tl.store(OUT + (bh * G + offs_g)[:, None] * D + offs_d[None, :], out.to(OUT.dtype.element_ty),
                 mask=gmask[:, None])


_FIELDS = ("k", "v", "k_scale", "k_zero", "v_scale", "v_zero")


def batched_decode_supported(caches, device: torch.device, head_dim: int) -> bool:
    """CUDA, every row allocated in one storage format, no Quest page tracking, a power-of-2 head_dim."""
    if not (HAS_TRITON and device.type == "cuda" and head_dim >= 16 and head_dim & (head_dim - 1) == 0):
        return False
    first = caches[0].layers[0]
    return all(c.length > 0 and not c.layers[0].track_pages and c.layers[0].kv_cache_dtype == first.kv_cache_dtype
               and c.layers[0].dtype == first.dtype for c in caches) and first.dtype in _ELEM_DTYPES


def build_decode_tables(caches, device: torch.device) -> torch.Tensor:
    """[layers, B, 8] int64: per layer and row, the cache's data pointers, length and capacity (one H2D copy)."""
    rows = []
    for i in range(len(caches[0].layers)):
        for c in caches:
            lc = c.layers[i]
            s = lc.store
            rows.append([s[n].data_ptr() if n in s else 0 for n in _FIELDS] + [lc.length, lc.capacity])
    return torch.tensor(rows, dtype=torch.int64).view(len(caches[0].layers), len(caches), 8).to(device)


def triton_kv_decode_batch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, rows, table: torch.Tensor) -> torch.Tensor:
    """q [B, H, 1, D], k, v [B, Hkv, 1, D] of one decode step: appends k, v to every row's cache, attends each row
    over its own cache, and returns [B, H, 1, D]. Capacity must already hold length + 1 (LayerKVCache.ensure)."""
    B, H, _, D = q.shape
    Hkv = k.shape[1]
    G = H // Hkv
    first = rows[0]
    mode, elem = _MODES[first.kv_cache_dtype], _ELEM_DTYPES[first.dtype]
    kv = torch.stack((k, v)).reshape(2, B, Hkv, D)
    _kv_append_batch_kernel[(B, Hkv, 2)](kv, table, kv.stride(0), kv.stride(1), kv.stride(2),
                                         D=D, MODE=mode, GROUP=INT4_GROUP, ELEM=elem)
    max_len = max(r.length for r in rows) + 1
    BLOCK_N = 64 if D <= 64 else 32
    n_splits = max(1, min(math.ceil(max_len / 256), max(1, 256 // (B * Hkv))))
    tokens_per_split = math.ceil(math.ceil(max_len / n_splits) / BLOCK_N) * BLOCK_N
    n_splits = math.ceil(max_len / tokens_per_split)
    acc = torch.empty(B * Hkv * n_splits * G, D, device=q.device, dtype=torch.float32)
    m = torch.empty(B * Hkv * n_splits * G, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    qc = q.reshape(B, H, D)
    BLOCK_G = max(16, triton.next_power_of_2(G))
    _kv_decode_batch_kernel[(B, Hkv, n_splits)](
        qc, table, acc, m, l, qc.stride(0), qc.stride(1), tokens_per_split, 1.0 / math.sqrt(D),
        G=G, BLOCK_G=BLOCK_G, D=D, BLOCK_N=BLOCK_N, MODE=mode, GROUP=INT4_GROUP, ELEM=elem,
        N_SPLITS=n_splits, HKV=Hkv, num_stages=2,
    )
    out = torch.empty(B, H, D, device=q.device, dtype=q.dtype)
    _kv_reduce_kernel[(B * Hkv,)](acc, m, l, out, G=G, BLOCK_G=BLOCK_G, D=D, N_SPLITS=n_splits)
    for r in rows:
        r.length += 1
    return out.view(B, H, 1, D)
