"""Grouped MoE GEMM (turing/kernels/triton_moe.py): the pair alignment on any device, the Triton kernels against
SparseMoE's per-expert loop on CUDA."""

import pytest
import torch

from turing.kernels.triton_moe import moe_align

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton MoE kernels")


@pytest.mark.parametrize("T,top_k,E,block_m", [(1, 2, 4, 16), (5, 8, 64, 16), (64, 8, 64, 16), (300, 4, 8, 64),
                                               (3, 2, 128, 16)])
def test_moe_align_groups_every_pair_once_by_expert(T, top_k, E, block_m):
    g = torch.Generator().manual_seed(T * E)
    experts = torch.stack([torch.randperm(E, generator=g)[:top_k] for _ in range(T)])
    pairs, block_expert = moe_align(experts, E, block_m)
    n = T * top_k
    assert pairs.numel() == block_expert.numel() * block_m
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


@cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-4), (torch.bfloat16, 2e-2), (torch.float16, 2e-3)])
@pytest.mark.parametrize("T", [1, 3, 64, 600])  # 600 tokens x 2 = 1,200 pairs: the 64-row blocks
def test_fused_moe_matches_the_expert_loop(dtype, atol, T):
    from turing.kernels.triton_moe import fused_moe
    moe = _moe(dtype)
    x = torch.randn(T, 256, device="cuda", dtype=dtype)
    probs = torch.softmax(moe.router(x).float(), -1)
    weights, experts = torch.topk(probs, moe.top_k, -1)
    weights = weights / weights.sum(-1, keepdim=True)
    ref = torch.zeros_like(x)
    for e in torch.unique(experts).tolist():
        token, slot = torch.where(experts == e)
        gate, up = torch.nn.functional.linear(x[token], moe.gate_up[e]).chunk(2, -1)
        y = torch.nn.functional.linear(moe.act(gate) * up, moe.down[e])
        ref.index_add_(0, token, (y * weights[token, slot, None]).to(ref.dtype))
    out = fused_moe(x, moe.gate_up, moe.down, weights, experts)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=0)


@cuda
@torch.no_grad()
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
