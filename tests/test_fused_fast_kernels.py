"""
Unit tests for Native C++20 AVX2 SIMD Serving Helpers & Fused Triton GPU Kernels.
Verifies numerical parity, deterministic hashing, zero-sync quadtrees, and spec parity.
"""

import pytest
import torch
import numpy as np

try:
    import turing.turing_csrc as turing_csrc
    HAS_CSRC = True
except ImportError:
    HAS_CSRC = False

from turing.serving.traffic import PrefixHashRouter
from turing.serving.spec_gate import SpecExactParityVerifier, ParityReport
from turing.experimental.lineage import _hash_kv_tensors
from turing.experimental.kernels.triton_vram_hash import fused_tensor_checksum_cuda, compute_fast_tensor_hash
from turing.experimental.kernels.triton_gated_zero_identity import fused_gated_zero_identity_cuda
from turing.experimental.kernels.triton_chunk_filter import fused_chunk_context_filter_cuda
from turing.experimental.kernels.triton_quadtree_mrp import fused_quadtree_mrp_cuda
from turing.experimental.speculation import QuadtreeMRPSpeculator, MatryoshkaDraftHead
from turing.experimental.kslot_pooling import GatedZeroIdentityHead
from turing.experimental.hybrid_attention import ChunkContextScorer


def test_prefix_router_fast_hash_parity():
    tokens = [128000, 15496, 995, 382, 1024, 2048, 4096, 8192, 16384, 32000]
    router = PrefixHashRouter(window=128)
    h_py = router.compute_prefix_hash(tokens)
    assert isinstance(h_py, int)
    assert h_py > 0

    if HAS_CSRC:
        tok_arr = np.array(tokens, dtype=np.int32)
        h_cpp = turing_csrc.compute_prefix_hash_fast(tok_arr, 128)
        assert h_py == h_cpp


def test_spec_verifier_fast_parity():
    spec = [10, 20, 30, 40, 50]
    plain = [10, 20, 30, 40, 50]

    rep = SpecExactParityVerifier.verify_greedy_parity(spec, plain)
    assert rep.passed is True
    assert rep.num_tokens_compared == 5

    spec_div = [10, 20, 99, 40, 50]
    rep_div = SpecExactParityVerifier.verify_greedy_parity(spec_div, plain)
    assert rep_div.passed is False
    assert rep_div.divergence_index == 2

    if HAS_CSRC:
        s_arr = np.array(spec, dtype=np.int32)
        p_arr = np.array(plain, dtype=np.int32)
        p, n, d = turing_csrc.verify_greedy_parity_fast(s_arr, p_arr)
        assert p is True
        assert n == 5

        s_div_arr = np.array(spec_div, dtype=np.int32)
        p_div, n_div, d_div = turing_csrc.verify_greedy_parity_fast(s_div_arr, p_arr)
        assert p_div is False
        assert d_div == 2


def test_fast_tensor_hash_consistency():
    t1 = torch.randn(4, 16, 64)
    t2 = torch.randn(4, 16, 64)

    h1 = compute_fast_tensor_hash([t1, t2])
    h2 = compute_fast_tensor_hash([t1, t2])
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 128 # Standard 512-bit BLAKE2b hex string


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton checksum runs on CUDA")
def test_vram_checksum_sees_every_bit():
    # converting floats to int (instead of bitcasting) gave 0.25 and 0.5 the same checksum
    a = torch.full((1000,), 0.25, device="cuda")
    assert fused_tensor_checksum_cuda(a) != fused_tensor_checksum_cuda(a * 2)
    assert fused_tensor_checksum_cuda(a.half()) != fused_tensor_checksum_cuda((a * 2).half())


def test_gated_zero_identity_forward_and_fused():
    head = GatedZeroIdentityHead(latent_dim=64, num_kv_heads=4, head_dim=32)
    feat = torch.randn(2, 8, 64)

    dk, dv = head(feat)
    assert dk.shape == (2, 8, 4, 32)
    assert dv.shape == (2, 8, 4, 32)
    # Since heads are zero-initialized, dk and dv must be exactly zero!
    assert torch.allclose(dk, torch.zeros_like(dk))
    assert torch.allclose(dv, torch.zeros_like(dv))


def test_chunk_context_filter_fused():
    scorer = ChunkContextScorer(hidden_dim=256, chunk_size=128, budget_tokens=512)
    k = torch.randn(1, 1024, 4, 64)
    v = torch.randn(1, 1024, 4, 64)
    q = torch.randn(1, 16, 4, 64)

    k_f, v_f = scorer.filter_context(k, v, q)
    assert k_f.shape[1] == 512
    assert v_f.shape[1] == 512


def test_quadtree_mrp_cuda_generator():
    hidden = torch.randn(1, 256)
    draft_weight = torch.randn(1000, 256)
    spatial_proj = torch.randn(2, 256)

    toks, parents, mask = fused_quadtree_mrp_cuda(hidden, draft_weight, spatial_proj, slice_width=128)
    assert toks.shape == (21,)
    assert parents.shape == (21,)
    assert mask.shape == (21, 21)
    assert parents[0].item() == -1
    assert parents[1].item() == 0


def test_quadtree_mrp_matches_the_masked_reference():
    # the sync-free quadrant pick must equal "first candidate in the quadrant, else the fallback"
    import torch.nn.functional as F
    torch.manual_seed(0)
    for _ in range(20):
        hidden, w, proj = torch.randn(1, 64), torch.randn(500, 64), torch.randn(2, 64) * 300
        toks, parents, mask = fused_quadtree_mrp_cuda(hidden, w, proj)
        top = torch.topk(F.linear(hidden, w).squeeze(0), 64).indices
        o = F.linear(hidden, proj).squeeze(0)
        tail = top[1:]
        dx = (tail % 7).float() - 3.0 - o[0] * 0.01
        dy = ((tail // 7) % 7).float() - 3.0 - o[1] * 0.01
        quads = [(dx >= 0) & (dy >= 0), (dx < 0) & (dy >= 0), (dx < 0) & (dy < 0), (dx >= 0) & (dy < 0)]
        want = [top[0]] + [tail[q][0] if q.any() else top[i + 1] for i, q in enumerate(quads)] + list(top[5:21])
        assert toks.tolist() == [int(t) for t in want]
        for i, p in enumerate(parents.tolist()):
            assert mask[i, i] == 0 and (p == -1 or mask[i, p] == 0)
        assert int((mask == 0).sum()) == 21 + 20 + 16  # self, the root for depth 1-2, the depth-1 parent for depth 2
