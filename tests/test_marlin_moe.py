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


@needs_marlin
@torch.no_grad()
def test_offloaded_layer_keeps_its_resident_experts_in_marlins_layout():
    """An offloaded layer after use_marlin: the resident experts through Marlin, the others on the CPU for a few
    tokens and through the device one at a time for a prefill chunk, with the all-resident layer's output; an expert
    that becomes resident is converted on the way in."""
    from turing.models import causal_lm
    full, moe = _moe(shared=True, **SHAPE), _moe(shared=True, **SHAPE)
    for m in (full, moe):
        quantize_experts(m, "rtn")
    moe.offload(5)
    full, moe = full.to("cuda", torch.float16), moe.to("cuda", torch.float16)
    assert moe.use_marlin(torch.float16) and all(moe._buffers[name] is None for name in moe._arrays())
    assert not moe.graph_safe(torch.zeros(1, 256, device="cuda", dtype=torch.float16))
    g = torch.Generator(device="cuda").manual_seed(5)
    for T in (1, 2, 8, 40, 300):
        x = torch.randn(T, 256, device="cuda", generator=g).half()
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=3e-3, rtol=2e-2)
    assert moe.offload_streamed > 0 and moe.offload_misses == 0 and moe._slot_expert == [0, 1, 2, 3, 4]
    if causal_lm.w4_expert_cpu is None:
        return
    assert moe.offload_cpu > 0
    moe.REBALANCE, moe._small_steps, moe._uses, cpu = 8, 0, [0] * 8, moe.offload_cpu  # the counts start over
    weights = torch.tensor([[0.6, 0.4]], device="cuda")
    for step in range(23):  # expert 7 in every step; expert 0 only once it has lost its slot to 7 (the 8th step)
        experts = torch.tensor([[7, 0 if step >= 16 else 1 + step % 4]], device="cuda")
        full.route = moe.route = lambda x, e=experts: (weights, e)
        x = torch.randn(1, 256, device="cuda", generator=g).half()
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=3e-3, rtol=2e-2)
    assert moe._expert_slot[7] >= 0 and moe._expert_slot[0] == -1 and moe.offload_misses == 1
    assert moe.offload_cpu - cpu == 7 + 7  # expert 7 before the swap, expert 0 from step 16 on


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


@needs_marlin
@torch.inference_mode()
def test_a_captured_step_runs_the_routed_experts_beside_the_shared_expert(monkeypatch):
    """SparseMoE._two_branches: inside a capture the routing and the routed experts are launched on a second stream.
    Three layers one after the other in one graph (each fork takes memory the one before freed), replayed on new
    inputs: the eager output every time. Outside a capture, and past BRANCH_MAX_ROWS rows, one stream as before."""
    from turing.models.causal_lm import SparseMoE
    layers = []
    for seed in range(3):
        moe = _moe(shared=True, **SHAPE)
        torch.manual_seed(seed)
        for p in moe.parameters():
            p.normal_(0.0, 0.05)
        quantize_experts(moe, "rtn")
        moe = moe.to("cuda", torch.float16)
        assert moe.use_marlin(torch.float16)
        layers.append(moe)
    forks, two = [], SparseMoE._two_branches
    monkeypatch.setattr(SparseMoE, "_two_branches", lambda self, x, w: forks.append(x.shape[0]) or two(self, x, w))

    def stack(x):
        for moe in layers:
            x = x + moe(x)
        return x
    g = torch.Generator(device="cuda").manual_seed(3)
    static = torch.randn(4, 1, 256, device="cuda", dtype=torch.float16, generator=g)
    eager = stack(static)
    assert not forks
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        out = stack(static)
    assert forks == [4, 4, 4]
    graph.replay()
    torch.testing.assert_close(out, eager, atol=2e-3, rtol=1e-3)
    for _ in range(5):
        static.copy_(torch.randn(4, 1, 256, device="cuda", dtype=torch.float16, generator=g))
        graph.replay()
        torch.testing.assert_close(out, stack(static), atol=2e-3, rtol=1e-3)
    wide = torch.randn(SparseMoE.BRANCH_MAX_ROWS + 1, 256, device="cuda", dtype=torch.float16, generator=g)
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        layers[0](wide)
    assert forks == [4, 4, 4]


@needs_marlin
@torch.inference_mode()
def test_a_model_with_shared_experts_decodes_from_graphs_as_it_does_eagerly():
    """The whole decode step of a 4-bit MoE model with shared experts, captured with its two branches a layer
    (DecodeGraphs): the eager step's logits for batches of one to three rows, step after step."""
    import copy
    torch.manual_seed(0)
    cfg = _cfg(layers=3, shared=True, **SHAPE)
    cfg.kv_pool_gib = 0.02
    base = quantize_model(SubspaceCausalLM(cfg).eval(), "rtn").to("cuda", torch.float16)
    prompts = [torch.randint(0, 128, (1, n), device="cuda") for n in (24, 9, 17)]
    steps = [torch.randint(0, 128, (B, 1), device="cuda") for B in (1, 1, 1, 2, 2, 3, 3, 3)]
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base)
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        outs.append([model(ids, past_key_values=caches[:ids.shape[0]])[0][:, -1].clone() for ids in steps])
        assert all(layer.mlp._marlin is not None for layer in model.layers)
        assert bool(model._graphs.graphs) == graphs
    for a, b in zip(*outs):
        torch.testing.assert_close(a.float(), b.float(), atol=5e-3, rtol=0)


@needs_marlin
@torch.inference_mode()
def test_an_offloaded_models_decode_step_is_a_chain_of_graphs_with_the_eager_logits():
    """DecodeGraphs for a model whose 4-bit MoE layers are offloaded: the step is captured as graphs cut at every
    layer (StepCuts), the host reads the routing between them and the CPU takes the experts that are not resident.
    The eager step's logits, for batches with and without padding rows, and after the resident experts change."""
    import copy
    from turing.models import causal_lm
    from turing.models.causal_lm import StepCuts
    if causal_lm.w4_expert_cpu is None:
        pytest.skip("the native extension is not built")
    torch.manual_seed(0)
    cfg = _cfg(layers=2, **SHAPE)
    cfg.kv_pool_gib = 0.02
    base = quantize_model(SubspaceCausalLM(cfg).eval(), "rtn")
    base.offload_experts(5)
    base = base.to("cuda", torch.float16)
    prompts = [torch.randint(0, 128, (1, n), device="cuda") for n in (24, 9, 17)]
    steps = [torch.randint(0, 128, (B, 1), device="cuda") for B in (1, 1, 1, 2, 2, 2, 3, 3, 3) for _ in range(2)]
    outs = []
    for graphs in (False, True):
        model = copy.deepcopy(base)
        model.config.cuda_graphs = graphs
        caches = [model(p)[1] for p in prompts]
        got = []
        for n, ids in enumerate(steps):
            if n == 3:  # expert 7 takes expert 0's slot in every layer: the graphs read the slots as they are now
                for layer in model.layers:
                    layer.mlp._load(7, 0)
            got.append(model(ids, past_key_values=caches[:ids.shape[0]])[0][:, -1])
        outs.append(got)
        chains = [entry[1] for entry in model._graphs.graphs.values()]
        if graphs:  # 1, 2 and 3 rows: three graph sizes (3 pads to 4), each cut twice per layer
            assert len(chains) == 3 and all(isinstance(c, StepCuts) and len(c.sites) == 2 and len(c.graphs) == 5
                                            for c in chains)
        else:
            assert not chains
        assert all(layer.mlp._marlin is not None and layer.mlp._cuts is None for layer in model.layers)
        assert model.expert_offload_stats()["cpu"] > 0
    for a, b in zip(*outs):
        torch.testing.assert_close(a.float(), b.float(), atol=5e-3, rtol=0)
