import pytest
import torch

from turing.core.kv_cache import KVCache, KVPool
from turing.kernels.flashinfer_prefill import HAS_FLASHINFER

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_FLASHINFER),
                                reason="CUDA and the optional flashinfer package required")


def _filled(pool, start, L, g):
    cache = KVCache(pool)
    cache.reserve(start + L)
    if start:
        pool.append(0, cache, *(torch.randn(1, pool.num_kv_heads, start, pool.head_dim, device="cuda", generator=g)
                                .half() for _ in range(2)))
        cache.length = start
    pool.append(0, cache, *(torch.randn(1, pool.num_kv_heads, L, pool.head_dim, device="cuda", generator=g).half()
                            for _ in range(2)))
    return cache


@pytest.mark.parametrize("H,Hkv,D", [(12, 2, 128), (8, 8, 64)])
@pytest.mark.parametrize("start,L", [(0, 50), (37, 70), (300, 600)])  # off page boundaries; a cached prefix
def test_flashinfer_prefill_matches_gather_and_sdpa(H, Hkv, D, start, L):
    from tests.test_triton_kv_prefill import _reference
    from turing.kernels.flashinfer_prefill import flashinfer_prefill
    g = torch.Generator(device="cuda").manual_seed(0)
    pool = KVPool(1, Hkv, D, 128, torch.float16, "cuda", "native")
    cache = _filled(pool, start, L, g)
    q = torch.randn(1, H, L, D, device="cuda", generator=g).half()
    out = flashinfer_prefill(pool, 0, q, [cache.slot], [start], [L], D ** -0.5)
    ref = _reference(pool, 0, cache, q, start, L, D ** -0.5, 0, 0.0)
    torch.testing.assert_close(out.float(), ref, atol=1e-3, rtol=1e-2)


def test_a_reused_slot_with_other_pages_is_planned_again():
    """Same slot, start and length in the next forward, but other pages: the plan's page indices must not be reused."""
    from tests.test_triton_kv_prefill import _reference
    from turing.kernels.flashinfer_prefill import flashinfer_prefill
    g = torch.Generator(device="cuda").manual_seed(0)
    pool = KVPool(1, 2, 128, 128, torch.float16, "cuda", "native")
    q = torch.randn(1, 12, 64, 128, device="cuda", generator=g).half()
    first = _filled(pool, 0, 64, g)
    flashinfer_prefill(pool, 0, q, [first.slot], [0], [64])
    slot, old = first.slot, list(first.pages)
    hold = KVCache(pool)  # takes the freed pages first, so the next request gets other pages in the same slot
    first.free()
    hold.reserve(64)
    second = _filled(pool, 0, 64, g)
    assert second.slot == slot and second.pages != old
    out = flashinfer_prefill(pool, 0, q, [second.slot], [0], [64])
    torch.testing.assert_close(out.float(), _reference(pool, 0, second, q, 0, 64, 128 ** -0.5, 0, 0.0),
                               atol=1e-3, rtol=1e-2)


def test_packed_requests_share_one_call():
    """Several requests' chunks in one row (one cached prefix among them), one plan: each slice matches its reference."""
    from tests.test_triton_kv_prefill import _reference
    from turing.kernels.flashinfer_prefill import flashinfer_prefill
    g = torch.Generator(device="cuda").manual_seed(0)
    pool = KVPool(1, 2, 128, 256, torch.float16, "cuda", "native")
    spec = [(0, 600), (37, 70), (0, 17), (300, 513)]
    caches = [_filled(pool, s, n, g) for s, n in spec]
    q = torch.randn(1, 12, sum(n for _, n in spec), 128, device="cuda", generator=g).half()
    out = flashinfer_prefill(pool, 0, q, [c.slot for c in caches], [s for s, _ in spec], [n for _, n in spec])
    off = 0
    for c, (s, n) in zip(caches, spec):
        ref = _reference(pool, 0, c, q[:, :, off:off + n], s, n, 128 ** -0.5, 0, 0.0)
        torch.testing.assert_close(out[:, off:off + n].float(), ref, atol=1e-3, rtol=1e-2)
        off += n
