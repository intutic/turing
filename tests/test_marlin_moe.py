"""Marlin W4A16 for MoE experts (turing.kernels.marlin_moe, vendored from vLLM v0.31.0): on CUDA with the extension
built, marlin_moe against the per-expert loop on the dequantized experts, SparseMoE's Marlin path (every input size,
inside a CUDA graph, packed buffers moved to host memory) and a quantized MoE model served through it."""
import pytest
import torch

from tests.test_moe_w4 import _cfg, _dequantized_twin, _moe
from turing.kernels.marlin_moe import HAS_MARLIN_MOE, _block_rows, marlin_moe_supported
from turing.models.causal_lm import SubspaceCausalLM
from turing.models.quant import quantize_experts, quantize_model

needs_marlin = pytest.mark.skipif(not (torch.cuda.is_available() and HAS_MARLIN_MOE),
                                  reason="CUDA and the Marlin extensions (TURING_BUILD_MARLIN=1) required")
SHAPE = dict(E=8, H=256, F=384)  # Marlin's tiles: 2F and H in 64s, H and F in 128s


def test_block_rows_follow_the_rows_an_expert_gets():
    assert [_block_rows(n, 60) for n in (4, 32, 256, 600, 2048, 8192)] == [8, 8, 8, 16, 48, 64]


def test_shapes_outside_marlins_tiles_are_refused():
    assert not marlin_moe_supported(256, 384, 64, torch.float16, "cpu")
    if torch.cuda.is_available() and HAS_MARLIN_MOE:
        assert marlin_moe_supported(256, 384, 64, torch.bfloat16, "cuda")
        assert not marlin_moe_supported(256, 320, 64, torch.bfloat16, "cuda")  # F = 320 is not a multiple of 128
        assert not marlin_moe_supported(256, 384, 64, torch.float32, "cuda")


@needs_marlin
@torch.no_grad()
@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 2e-2), (torch.float16, 2e-3)])
@pytest.mark.parametrize("T", [1, 3, 64, 700])  # blocks of 8 rows up to 700 tokens x 2 / 8 experts: 64-row blocks
def test_marlin_moe_matches_the_expert_loop(dtype, atol, T):
    from turing.kernels.marlin_moe import marlin_moe, marlin_moe_pack
    moe = _moe(**SHAPE)
    quantize_experts(moe, "rtn")
    twin = _dequantized_twin(moe, **SHAPE).to("cuda", dtype)
    moe = moe.to("cuda", dtype)
    x = torch.randn(T, 256, device="cuda", dtype=dtype)
    weights, experts = moe.route(x)
    ref = torch.zeros_like(x)
    for e in torch.unique(experts).tolist():
        token, slot = torch.where(experts == e)
        gate, up = torch.nn.functional.linear(x[token], twin.gate_up[e]).chunk(2, -1)
        y = torch.nn.functional.linear(twin.act(gate) * up, twin.down[e])
        ref.index_add_(0, token, (y * weights[token, slot, None]).to(ref.dtype))
    w = marlin_moe_pack((moe.gate_up_q, moe.gate_up_scales, moe.gate_up_zeros),
                        (moe.down_q, moe.down_scales, moe.down_zeros), dtype)
    out = marlin_moe(x, w, weights, experts, moe.act)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=0)


# in inference mode, as DecodeGraphs captures (see tests/test_moe_w4.py)
@needs_marlin
@torch.inference_mode()
def test_sparse_moe_serves_packed_experts_through_marlin_in_a_cuda_graph():
    moe = _moe(shared=True, **SHAPE)
    quantize_experts(moe, "rtn")
    moe = moe.to("cuda", torch.float16)
    x = torch.randn(4, 1, 256, device="cuda", dtype=torch.float16)
    triton = moe(x)  # the Triton int4 kernels
    assert moe.use_marlin(torch.float16) and moe.graph_safe(x)
    assert all(not moe._buffers[name].is_cuda for name in moe._arrays())  # one copy of the experts on the device
    eager = moe(x)
    torch.testing.assert_close(eager, triton, atol=2e-3, rtol=0)
    graph = torch.cuda.CUDAGraph()
    static = x.clone()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out = moe(static)
    graph.replay()
    torch.testing.assert_close(out, eager, atol=1e-3, rtol=0)
    prefill = torch.randn(1, 300, 256, device="cuda", dtype=torch.float16)  # 600 pairs: 64-row blocks
    assert moe(prefill).shape == prefill.shape
    with pytest.raises(ValueError, match="offload before"):
        moe.offload(4)


@needs_marlin
@torch.inference_mode()
def test_a_quantized_moe_model_decodes_through_marlin_as_through_the_triton_kernels():
    import copy
    torch.manual_seed(0)
    cfg = _cfg(layers=2, **SHAPE)
    cfg.kv_pool_gib = 0.02
    base = quantize_model(SubspaceCausalLM(cfg).eval(), "rtn").to("cuda", torch.float16)
    ids = torch.randint(0, 128, (1, 24), device="cuda")
    outs = []
    for marlin in (False, True):
        model = copy.deepcopy(base)
        if not marlin:
            model._prepare_marlin = lambda dtype: None  # keep the Triton kernels
        logits, cache = model(ids)
        steps = [model(ids[:, :1] + t, past_key_values=[cache])[0] for t in range(3)]  # decode steps: CUDA graphs
        outs.append([logits[:, -1]] + [s[:, -1] for s in steps])
        used = [layer.mlp._marlin is not None for layer in model.layers]
        assert used == [marlin, marlin]
    for a, b in zip(*outs):
        torch.testing.assert_close(a.float(), b.float(), atol=5e-3, rtol=0)
