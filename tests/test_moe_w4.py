"""4-bit MoE experts: packing the stacked expert weights (turing.models.quant.quantize_experts), SparseMoE on packed
experts against the same layer on the dequantized weights, the .tgate4 round trip, and on CUDA the int4 grouped
kernels (turing.kernels.triton_moe.fused_moe_w4) against the per-expert loop."""

import pytest
import torch

from turing.config import ModelConfig
from turing.models.causal_lm import SparseMoE, SubspaceCausalLM
from turing.models.quant import (_expert_grams, dequantize, load_quantized, quantize_experts, quantize_model,
                                 save_quantized)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the Triton MoE kernels")


def _cfg(E=4, H=64, F=128, layers=1, shared=False):
    return ModelConfig(name="moe-w4-test", hidden_dim=H, ffn_dim=F, num_layers=layers, num_heads=4, num_kv_heads=4,
                       head_dim=H // 4, vocab_size=128, num_experts=E, num_experts_per_tok=2, moe_ffn_dim=F,
                       hidden_act="silu", shared_expert_ffn_dim=F if shared else 0)


def _moe(**kw):
    torch.manual_seed(0)
    moe = SparseMoE(_cfg(**kw))
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0.0, 0.05)
    return moe


def _dequantized_twin(packed: SparseMoE, **kw) -> SparseMoE:
    """An fp SparseMoE holding the packed layer's dequantized experts and the same router."""
    twin = _moe(**kw)
    with torch.no_grad():
        twin.router.weight.copy_(packed.router.weight)
        for e in range(twin.num_experts):
            gate_up, down = packed.expert_weights(e, torch.float32)
            twin.gate_up[e].copy_(gate_up)
            twin.down[e].copy_(down)
    return twin


@torch.no_grad()
def test_rtn_packs_every_expert_within_half_a_grid_step():
    moe = _moe()
    gate_up, down = moe.gate_up.clone(), moe.down.clone()
    quantize_experts(moe, "rtn")
    assert moe.gate_up is None and moe.down is None
    assert moe.gate_up_q.shape == (4, 256, 32) and moe.gate_up_q.dtype == torch.uint8
    assert moe.down_q.shape == (4, 64, 64) and moe.down_scales.shape == (4, 64, 2)
    for name, ref in (("gate_up", gate_up), ("down", down)):
        q, s, z = (getattr(moe, name + part) for part in ("_q", "_scales", "_zeros"))
        err = (dequantize(q, s, z, torch.float32) - ref).abs().unflatten(-1, (s.shape[-1], -1))
        assert (err <= s.float()[..., None] * 0.5 + 1e-6).all()
    assert not [n for n, _ in moe.named_parameters() if "gate_up" in n or n == "down"]
    assert {"gate_up_q", "gate_up_scales", "gate_up_zeros", "down_q", "down_scales", "down_zeros"} <= set(moe.state_dict())


@torch.no_grad()
@pytest.mark.parametrize("T", [1, 7, 40])
def test_packed_moe_equals_the_layer_on_its_dequantized_experts(T):
    moe = _moe()
    quantize_experts(moe, "rtn")
    twin = _dequantized_twin(moe)
    x = torch.randn(T, 64)
    torch.testing.assert_close(moe(x), twin(x), atol=1e-6, rtol=0)


@torch.no_grad()
def test_gptq_lowers_the_output_error_and_rounds_unseen_experts_to_nearest():
    moe = _moe(E=4, H=64, F=64)
    ref = _moe(E=4, H=64, F=64)
    rtn = _moe(E=4, H=64, F=64)
    quantize_experts(rtn, "rtn")
    torch.manual_seed(1)
    x = torch.randn(4000, 64) @ (torch.eye(64) + 0.5 * torch.randn(64, 64))  # correlated inputs
    grams = {"gate_up": torch.zeros(4, 64, 64), "down": torch.zeros(4, 64, 64), "tokens": torch.zeros(4, dtype=torch.int64)}
    _expert_grams(moe, x, grams)
    assert int(grams["tokens"].sum()) == 4000 * 2  # every token counted once per routed expert
    grams["tokens"][3] = 10  # as if expert 3 had seen 10 tokens: fewer than its 64 input columns
    quantize_experts(moe, "gptq", grams)
    assert torch.equal(moe.gate_up_q[3], rtn.gate_up_q[3]) and torch.equal(moe.down_q[3], rtn.down_q[3])
    assert not torch.equal(moe.gate_up_q[0], rtn.gate_up_q[0])
    for e in range(3):  # GPTQ minimises the error of gate_up's outputs on the tokens routed to the expert
        xe = x[(ref.route(x)[1] == e).any(-1)]
        want = xe @ ref.gate_up[e].T
        err = lambda m: float(((xe @ m.expert_weights(e, torch.float32)[0].T - want) ** 2).mean())  # noqa: E731
        assert err(moe) < err(rtn)


@torch.no_grad()
@pytest.mark.parametrize("method", ["rtn", "gptq"])
def test_quantize_model_packs_the_experts_and_tgate4_restores_them(method, tmp_path):
    torch.manual_seed(0)
    model = SubspaceCausalLM(_cfg(E=4, H=64, F=64, layers=2, shared=True)).eval()
    ids = torch.randint(0, 128, (2, 24))
    quantize_model(model, method, calib=torch.randint(0, 128, (4, 48)) if method == "gptq" else None)
    assert all(layer.mlp.gate_up is None for layer in model.layers)
    logits, _ = model(ids)
    assert torch.isfinite(logits).all()
    save_quantized(model, str(tmp_path))
    loaded, _ = load_quantized(str(tmp_path))
    assert all(layer.mlp.gate_up is None and layer.mlp.group == 64 for layer in loaded.layers)
    torch.testing.assert_close(loaded(ids)[0], logits, atol=1e-5, rtol=0)


def test_offload_and_tensor_parallel_refuse_packed_experts():
    from turing.core.heterogeneous_moe import HostExpertBank
    from turing.models.tensor_parallel import _shard_moe
    torch.manual_seed(0)
    model = SubspaceCausalLM(_cfg(E=4, H=64, F=64)).eval()
    quantize_model(model, "rtn")
    with pytest.raises(ValueError, match="packed in 4 bits"):
        HostExpertBank.from_model(model)
    with pytest.raises(ValueError, match="packed in 4 bits"):
        _shard_moe(model.layers[0].mlp, 0, 2, None)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 2e-2), (torch.float16, 2e-3)])
@pytest.mark.parametrize("T", [1, 3, 64, 600])  # 600 tokens x 2 = 1,200 pairs: the 64-row blocks
# blocks of 1, 2 and 4 rows: the kernels without a tensor-core tile, whose step covers 1, 2 or 8 quantization groups
# (512 columns per step is wider than the 256 there are); and the tile kernels with two groups per step
@pytest.mark.parametrize("block_m,block_k", [(None, None), (1, None), (4, 512), (2, 64), (1, 128), (None, 128)])
def test_fused_moe_w4_matches_the_expert_loop(dtype, atol, T, block_m, block_k):
    from turing.kernels.triton_moe import fused_moe_w4
    moe = _moe(E=8, H=256, F=320 - 64)
    quantize_experts(moe, "rtn")
    twin = _dequantized_twin(moe, E=8, H=256, F=320 - 64).to("cuda", dtype)
    moe = moe.to("cuda", dtype)
    x = torch.randn(T, 256, device="cuda", dtype=dtype)
    weights, experts = moe.route(x)
    ref = torch.zeros_like(x)
    for e in torch.unique(experts).tolist():
        token, slot = torch.where(experts == e)
        gate, up = torch.nn.functional.linear(x[token], twin.gate_up[e]).chunk(2, -1)
        y = torch.nn.functional.linear(twin.act(gate) * up, twin.down[e])
        ref.index_add_(0, token, (y * weights[token, slot, None]).to(ref.dtype))
    out = fused_moe_w4(x, (moe.gate_up_q, moe.gate_up_scales, moe.gate_up_zeros),
                       (moe.down_q, moe.down_scales, moe.down_zeros), weights, experts, moe.group,
                       block_m=block_m, block_k=block_k)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=0)


# in inference mode, as DecodeGraphs captures: PyTorch's CUDA generator keeps the state of its first capture, and a
# capture in the other mode then fails on it (and leaves the generator unusable for the rest of the process)
@cuda
@torch.inference_mode()
def test_packed_sparse_moe_on_cuda_is_captured_in_a_cuda_graph():
    moe = _moe(E=8, H=256, F=256, shared=True)
    quantize_experts(moe, "rtn")
    moe = moe.to("cuda", torch.float16)
    x = torch.randn(4, 1, 256, device="cuda", dtype=torch.float16)
    assert moe.graph_safe(x)
    eager = moe(x)
    graph = torch.cuda.CUDAGraph()
    static = x.clone()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):  # capture fails on a host sync (the per-expert loop's .tolist())
        out = moe(static)
    graph.replay()
    torch.testing.assert_close(out, eager, atol=1e-3, rtol=0)
