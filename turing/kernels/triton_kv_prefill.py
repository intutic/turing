"""
Paged prefill attention over the KV pool (speed plan P5.1): a prompt chunk's queries attend to every key the request
has (its cached prefix and the chunk itself, already appended), read through the page table with in-register int8/int4
dequantization or native rows, as the decode kernel reads them (triton_kv_decode._load_rows).

The reference path this replaces gathered and dequantized the request's whole cache per layer per chunk, repeated the
KV heads for GQA, and ran SDPA over it. Here each program serves BLOCK_M (query position, query head) rows of one KV
head: the G query heads sharing a KV head are rows of one tile, so no KV head is copied. Causal within the chunk, with
an optional sliding window and score soft-capping; online softmax in fp32; dot products in the model dtype (fp32 inputs
use IEEE fp32).
"""

from typing import Optional
import math

import torch

from ..core.kv_cache import INT4_GROUP, KVPool
from .triton_kv_decode import HAS_TRITON, _MODES, pool_decode_supported

if HAS_TRITON:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from .triton_kv_decode import _load_rows
    from .triton_w4a16 import _TUNE_CACHE

    # Tiles tuned per (shape, storage, chunk-length bucket) on first use, choices kept on disk where Triton allows. With
    # one fixed tile the kernel ran 1.6-1.7x FlashInfer's paged prefill on native fp16 pages (L4,
    # results_prefill_attention_flashinfer_cuda.json). fp32 (IEEE) keeps the small tiles its dequantized rows need.
    _PREFILL_CONFIGS = [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
                        for bm, bn, w, s in ((64, 32, 4, 2), (64, 64, 4, 2), (64, 64, 4, 3), (128, 32, 8, 2),
                                             (128, 64, 8, 2), (128, 64, 8, 3), (64, 32, 4, 3), (32, 32, 4, 2))]

    def _prefill_prune(configs, named_args, **kwargs):
        args = {**named_args, **kwargs}  # constexprs arrive in kwargs or named_args depending on the Triton version
        if not args["IEEE"]:
            return configs
        b = 64 if args["D"] <= 64 else 32  # fp32: the one tile that fits the L4's shared memory, as before tuning
        return [c for c in configs if (c.kwargs["BLOCK_M"], c.kwargs["BLOCK_N"], c.num_warps, c.num_stages) == (b, b, 4, 2)]

    @triton.autotune(configs=_PREFILL_CONFIGS, key=["LB", "G", "D", "MODE", "WINDOW", "SOFTCAP", "IEEE"],
                     prune_configs_by={"early_config_prune": _prefill_prune}, **_TUNE_CACHE)
    @triton.jit(do_not_specialize=["slot", "start", "L", "LB", "window"])  # no build per value of these
    def _pool_prefill_kernel(Q, PT, K, V, KS, KZ, VS, VZ, OUT, slot, start, L, LB, s_qh, s_ql, pt_stride,
                             sm_scale, softcap, window,
                             G: tl.constexpr, D: tl.constexpr, HKV: tl.constexpr, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr, MODE: tl.constexpr, GROUP: tl.constexpr, WINDOW: tl.constexpr,
                             SOFTCAP: tl.constexpr, IEEE: tl.constexpr):
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        r = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # (query position, query head of this KV head) rows
        i = r // G
        g = r % G
        rmask = r < L * G
        qpos = start + i
        offs_d = tl.arange(0, D)
        q = tl.load(Q + (h * G + g)[:, None] * s_qh + i[:, None] * s_ql + offs_d[None, :], mask=rmask[:, None],
                    other=0.0)
        pt_row = PT + slot * pt_stride  # no .to(): Triton passes an int argument equal to 1 as a constant
        last = start + tl.minimum((pid_m * BLOCK_M + BLOCK_M - 1) // G, L - 1)  # the block's last query position
        lo = 0
        if WINDOW:
            lo = tl.maximum(start + (pid_m * BLOCK_M) // G - window + 1, 0)
            lo = (lo // BLOCK_N) * BLOCK_N
        m_i = tl.full([BLOCK_M], -1.0e30, dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        for j0 in range(lo, last + 1, BLOCK_N):
            j = j0 + tl.arange(0, BLOCK_N)
            kvalid = j <= last
            page = tl.load(pt_row + j // 16, mask=kvalid, other=0).to(tl.int64)
            row = (page * HKV + h) * 16 + j % 16
            k = _load_rows(K, KS, KZ, row, kvalid, offs_d, D, MODE, GROUP)
            if IEEE:
                s = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee")
            else:
                s = tl.dot(q, tl.trans(k.to(q.dtype)))
            s = s * sm_scale
            if SOFTCAP:
                s = softcap * libdevice.tanh(s / softcap)
            see = (j[None, :] <= qpos[:, None]) & kvalid[None, :]
            if WINDOW:
                see = see & (j[None, :] > qpos[:, None] - window)
            s = tl.where(see, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = _load_rows(V, VS, VZ, row, kvalid, offs_d, D, MODE, GROUP)
            if IEEE:
                acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
            else:
                acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), v.to(q.dtype))
            m_i = m_new
        o = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
        tl.store(OUT + i[:, None] * (HKV * G * D) + (h * G + g)[:, None] * D + offs_d[None, :],
                 o.to(OUT.dtype.element_ty), mask=rmask[:, None])


def pool_prefill_supported(pool: KVPool, q: torch.Tensor) -> bool:
    """As the decode kernel: CUDA, a power-of-2 head_dim of at least 16, a pool dtype Triton reads."""
    return pool_decode_supported(pool, q)


def pool_prefill_preferred(pool: KVPool, layer: int, q: torch.Tensor, start: int) -> bool:
    """Whether to run the paged kernel for this chunk rather than gather + SDPA. The kernel dequantizes the cached
    prefix once per block of queries; for a layer stored in int4 that costs more than one gather once a prefix is cached
    (L4, results_prefill_attention_cuda.json: 15.3 against 10.1 ms per layer at 14K cached tokens, equal with none)."""
    return pool_prefill_supported(pool, q) and (start == 0 or pool.layer_fields(layer, "k")[3] != "int4")


def triton_pool_prefill(pool: KVPool, layer: int, q: torch.Tensor, slot: int, start: int,
                        scale: Optional[float] = None, window: int = 0, softcap: float = 0.0) -> torch.Tensor:
    """Attention of a chunk's queries q [1, H, L, D] at positions start .. start + L - 1 over one request's pages
    (page-table row `slot`, the chunk's own K/V already appended): -> [1, L, H * D] in q's dtype."""
    _, H, L, D = q.shape
    Hkv = pool.num_kv_heads
    G = H // Hkv
    kc, ks, kz, storage = pool.layer_fields(layer, "k")
    vc, vs, vz, _ = pool.layer_fields(layer, "v")
    mode = _MODES[storage]
    scales = [ks, kz, vs, vz] if mode else [kc, vc, kc, vc]
    if q.stride(-1) != 1:
        q = q.contiguous()
    out = torch.empty(L, H, D, device=q.device, dtype=q.dtype)
    grid = lambda meta: (triton.cdiv(L * G, meta["BLOCK_M"]), Hkv)
    # LB, the tuning bucket: the chunk's length as a power of 2 from 512 to 2,048. Shorter chunks share the first:
    # the tail a prompt leaves for the next forward can be of any length, and a bucket for each stalled a serving
    # process for seconds at its first use (eight builds and their timing)
    _pool_prefill_kernel[grid](q, pool.page_table, kc, vc, *scales, out, slot, start, L,
                               min(max(triton.next_power_of_2(L), 512), 2048), q.stride(1), q.stride(2),
                               pool.page_table.stride(0), scale if scale is not None else 1.0 / math.sqrt(D),
                               float(softcap), int(window), G=G, D=D, HKV=Hkv, MODE=mode, GROUP=INT4_GROUP,
                               WINDOW=window > 0, SOFTCAP=softcap > 0, IEEE=q.dtype == torch.float32)
    return out.view(1, L, H * D)
