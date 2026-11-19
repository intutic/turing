"""
Paged prefill attention through FlashInfer (speed plan decision D10): the fallback for layers stored in the model dtype
when FlashInfer's kernel is faster than Turing's (triton_kv_prefill): it ran 1.4-1.6x faster on native fp16 pages on an
L4 (results_prefill_attention_ckpt3_cuda.json against results_prefill_attention_flashinfer_cuda.json). Used only when
the optional `flashinfer` package imports, for long chunks without a sliding window or score soft-capping. The pool's
per-layer K and V pages [pages, Hkv, 16, D] are FlashInfer's HND layout, read in place through the page table.
"""

from typing import List, Optional
import warnings

import torch

from ..core.kv_cache import KVPool

try:
    with warnings.catch_warnings():  # 0.7 warns about its own deprecated imports; -W error must not turn that off
        warnings.simplefilter("ignore")
        import flashinfer
    HAS_FLASHINFER = True
except Exception:  # not installed, or built for another torch/CUDA: Turing's own kernel serves every layer
    flashinfer = None
    HAS_FLASHINFER = False

MIN_CHUNK = 512  # below this the plan's host work is a larger share than the attention it saves
_STATE = {}  # device -> [wrapper, plan key, last layer served]


def flashinfer_prefill_supported(pool: KVPool, layer: int, q: torch.Tensor, window: Optional[int],
                                 softcap: Optional[float]) -> bool:
    return (HAS_FLASHINFER and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16) and q.shape[2] >= MIN_CHUNK
            and not window and not softcap and pool.dtype == q.dtype and pool.layer_fields(layer, "k")[3] == "native")


def flashinfer_prefill(pool: KVPool, layer: int, q: torch.Tensor, slots: List[int], starts: List[int],
                       lens: List[int], scale: Optional[float] = None) -> torch.Tensor:
    """As triton_pool_prefill (no window or soft-cap), for one or more requests packed in one row: q [1, H, T, D] holds
    request i's lens[i] queries at positions starts[i] .. over page-table row slots[i] (its chunk's K/V already
    appended), T = sum(lens) -> [1, T, H * D]. One plan per forward serves every layer and every request."""
    _, H, T, D = q.shape
    st = _STATE.get(q.device)
    if st is None:
        ws = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)
        st = _STATE[q.device] = [flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "HND"), None, None]
    key = (id(pool), pool.page_table.data_ptr(), tuple(slots), tuple(starts), tuple(lens), H, D, scale, q.dtype)
    # the plan holds the page indices: plan again at each new forward (a layer no later than the last one served), as
    # a reused slot can hold other pages with the same (start, L)
    if st[1] != key or layer <= st[2]:
        totals = [s + n for s, n in zip(starts, lens)]
        pages = [-(-t // 16) for t in totals]
        qo, kv = [0], [0]
        for n, p in zip(lens, pages):
            qo.append(qo[-1] + n)
            kv.append(kv[-1] + p)
        i32 = dict(dtype=torch.int32)  # indptrs and last-page lengths on the host: plan reads them there
        indices = torch.cat([pool.page_table[s, :p] for s, p in zip(slots, pages)]).to(torch.int32)
        st[0].plan(torch.tensor(qo, **i32), torch.tensor(kv, **i32), indices,
                   torch.tensor([t - (p - 1) * 16 for t, p in zip(totals, pages)], **i32),
                   H, pool.num_kv_heads, D, 16, causal=True, sm_scale=scale if scale is not None else D ** -0.5,
                   q_data_type=q.dtype, kv_data_type=q.dtype)
        st[1] = key
    st[2] = layer
    kc = pool.layer_fields(layer, "k")[0]
    vc = pool.layer_fields(layer, "v")[0]
    out = st[0].run(q[0].transpose(0, 1).contiguous(), (kc, vc))  # [T, H, D]
    return out.reshape(1, T, H * D)
