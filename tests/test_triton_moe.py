"""Grouped MoE GEMM (turing/kernels/triton_moe.py): the pair alignment on any device, the Triton kernels against
SparseMoE's per-expert loop on CUDA."""

import pytest
import torch

from turing.kernels.triton_moe import moe_align

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton MoE kernels")


@pytest.mark.parametrize("T,top_k,E,block_m", [(1, 2, 4, 16), (5, 8, 64, 16), (64, 8, 64, 16), (300, 4, 8, 64),
                                               (3, 2, 128, 16)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])  # torch operations; on CUDA three Triton kernels
def test_moe_align_groups_every_pair_once_by_expert(T, top_k, E, block_m, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    g = torch.Generator().manual_seed(T * E)
    experts = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(T)]).to(device)
    pairs, block_expert, used = moe_align(experts, E, block_m)
    n = T * top_k
    assert pairs.numel() == block_expert.numel() * block_m
    used = int(used)
    assert used % block_m == 0 and (block_expert[:used // block_m] < E).all() and (block_expert[used // block_m:] == E).all()
    pairs = pairs[:used]  # rows past the blocks in use are not written on CUDA
    block_expert = block_expert[:used // block_m]
    real = pairs[pairs < n]
    assert sorted(real.tolist()) == list(range(n))  # each pair exactly once
    flat = experts.reshape(-1)
    for b, e in enumerate(block_expert.tolist()):
        block = pairs[b * block_m:(b + 1) * block_m]
        block = block[block < n]
        if e == E:  # past the last group: nothing but padding
            assert block.numel() == 0
        else:
            assert block.numel() > 0 and (flat[block.long()] == e).all()


@cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("T,E,top_k,norm", [(1, 60, 4, False), (5, 64, 8, True), (33, 8, 2, True), (2, 128, 6, False)])
def test_route_kernel_equals_softmax_then_topk(dtype, T, E, top_k, norm):
    from turing.kernels.triton_moe import moe_route
    g = torch.Generator(device="cuda").manual_seed(T + E)
    logits = (torch.randn(T, E, device="cuda", generator=g) * 3).to(dtype)
    weights, experts = moe_route(logits, top_k, norm)
    want_w, want_e = torch.topk(torch.softmax(logits.float(), dim=-1), top_k, dim=-1)
    if norm:
        want_w = want_w / want_w.sum(dim=-1, keepdim=True)
    assert experts.dtype == torch.int64 and weights.dtype == torch.float32
    assert torch.equal(experts, want_e)  # best first, as torch.topk (random logits: no ties)
    torch.testing.assert_close(weights, want_w, atol=1e-6, rtol=1e-5)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gated_add_kernel_equals_the_three_operations(dtype):
    from turing.kernels.triton_moe import gated_add
    g = torch.Generator(device="cuda").manual_seed(3)
    a, b = (torch.randn(5, 1100, device="cuda", generator=g).to(dtype) for _ in range(2))
    gate = (torch.randn(5, 1, device="cuda", generator=g) * 3).to(dtype)
    torch.testing.assert_close(gated_add(a, gate, b).float(), (a + torch.sigmoid(gate) * b).float(), atol=2e-2, rtol=0)


@torch.no_grad()
@pytest.mark.parametrize("T", [1, 5, 300])
def test_gemm_moe_matches_the_expert_loop_on_any_device(T):
    """The per-expert GEMM path (prefill-sized inputs, and every device without the grouped kernels) against a plain
    loop over tokens and their experts."""
    from turing.kernels.triton_moe import gemm_moe
    torch.manual_seed(T)
    E, K, H, F = 6, 2, 32, 48
    gate_up, down = torch.randn(E, 2 * F, H) * 0.1, torch.randn(E, H, F) * 0.1
    x = torch.randn(T, H)
    experts = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    weights = torch.rand(T, K)
    ref = torch.zeros(T, H)
    for t in range(T):
        for j in range(K):
            e = int(experts[t, j])
            gate, up = (gate_up[e] @ x[t]).chunk(2)
            ref[t] += weights[t, j] * (down[e] @ (torch.nn.functional.silu(gate) * up))
    out = gemm_moe(x, weights, experts, E, lambda e: (gate_up[e], down[e]), torch.nn.functional.silu)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)


def _moe(dtype, E=8, H=256, F=320, shared=False):
    from turing.config import ModelConfig
    from turing.models.causal_lm import SparseMoE
    cfg = ModelConfig(name="moe-test", hidden_dim=H, ffn_dim=F, num_layers=1, num_heads=4, num_kv_heads=4, head_dim=H // 4,
                      vocab_size=128, num_experts=E, num_experts_per_tok=2, moe_ffn_dim=F, hidden_act="silu",
                      shared_expert_ffn_dim=F if shared else 0)
    torch.manual_seed(0)
    moe = SparseMoE(cfg)
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0.0, 0.05)
    return moe.to("cuda", dtype)


def _route_and_loop(moe, x):
    """(routing weights, experts, output of SparseMoE's per-expert loop) for x [T, H]."""
    probs = torch.softmax(moe.router(x).float(), -1)
    weights, experts = torch.topk(probs, moe.top_k, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for e in torch.unique(experts).tolist():
        token, slot = torch.where(experts == e)
        gate, up = torch.nn.functional.linear(x[token], moe.gate_up[e]).chunk(2, -1)
        y = torch.nn.functional.linear(moe.act(gate) * up, moe.down[e])
        ref.index_add_(0, token, (y * weights[token, slot, None]).to(ref.dtype))
    return weights, experts, ref


@cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2), (torch.float16, 2e-3)])
@pytest.mark.parametrize("T", [1, 3, 64, 600])  # 600 tokens x 2 = 1,200 pairs: the 64-row blocks
def test_fused_moe_matches_the_expert_loop(dtype, atol, T):
    from turing.kernels.triton_moe import fused_moe
    moe = _moe(dtype)
    x = torch.randn(T, 256, device="cuda", dtype=dtype)
    weights, experts, ref = _route_and_loop(moe, x)
    out = fused_moe(x, moe.gate_up, moe.down, weights, experts)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=0)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("block_m,block_n,block_k", [(16, 32, 128), (32, 128, 32), (64, 64, 64)])
@pytest.mark.parametrize("T", [1, 37])
def test_fused_moe_off_the_tile_grid_for_any_block_sizes(block_m, block_n, block_k, T):
    """Hidden and expert widths that are no multiple of a tile (a tensor-parallel shard's F / ranks), and every
    block and tile size giving the same result."""
    from turing.kernels.triton_moe import fused_moe
    moe = _moe(torch.float32, E=6, H=200, F=150)
    x = torch.randn(T, 200, device="cuda")
    weights, experts, ref = _route_and_loop(moe, x)
    out = fused_moe(x, moe.gate_up, moe.down, weights, experts, block_m=block_m, block_n=block_n, block_k=block_k)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=0)


@cuda
@torch.no_grad()
def test_sparse_moe_takes_the_gemm_path_for_prefill_sized_inputs_with_the_same_result():
    from turing.kernels.triton_moe import GEMM_MIN_ROWS, fused_moe
    moe = _moe(torch.float16)
    T = GEMM_MIN_ROWS * moe.num_experts // moe.top_k + 3
    x = torch.randn(T, 256, device="cuda", dtype=torch.float16)
    weights, experts = moe.route(x)
    kernel = fused_moe(x, moe.gate_up, moe.down, weights, experts)
    torch.testing.assert_close(moe(x).float(), kernel.float(), atol=2e-3, rtol=0)


# in inference mode, as DecodeGraphs captures: PyTorch's CUDA generator keeps the state of its first capture, and a
# capture in the other mode then fails on it (and leaves the generator unusable for the rest of the process)
@cuda
@torch.inference_mode()
def test_sparse_moe_on_cuda_uses_the_fused_path_without_host_syncs():
    moe = _moe(torch.float16, shared=True)
    x = torch.randn(4, 1, 256, device="cuda", dtype=torch.float16)
    eager = moe(x)
    graph = torch.cuda.CUDAGraph()
    static = x.clone()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):  # capture fails on a host sync (the per-expert loop's .tolist())
        out = moe(static)
    graph.replay()
    torch.testing.assert_close(out, eager, atol=1e-3, rtol=0)
