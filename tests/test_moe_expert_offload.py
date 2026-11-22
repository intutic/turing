"""Expert offload in the serving MoE layer (SparseMoE.offload): some experts on the device, all of them in host
memory, the same outputs as the layer with every expert resident; with fp and with packed 4-bit experts."""

import pytest
import torch

from turing.config import ModelConfig
from turing.models.causal_lm import SparseMoE, SubspaceCausalLM
from turing.models.quant import quantize_experts, quantize_model, save_quantized

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for the grouped MoE kernels")


def _cfg(E=8, H=64, F=64, layers=1, shared=False, top_k=2):
    return ModelConfig(name="moe-offload-test", hidden_dim=H, ffn_dim=F, num_layers=layers, num_heads=4,
                       num_kv_heads=4, head_dim=H // 4, vocab_size=128, num_experts=E, num_experts_per_tok=top_k,
                       moe_ffn_dim=F, hidden_act="silu", shared_expert_ffn_dim=F if shared else 0)


def _moe(packed=False, **kw):
    torch.manual_seed(0)
    moe = SparseMoE(_cfg(**kw))
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(0.0, 0.05)
    if packed:
        quantize_experts(moe, "rtn")
    return moe


@torch.no_grad()
@pytest.mark.parametrize("packed", [False, True])
def test_offloaded_layer_gives_the_outputs_of_the_resident_layer(packed):
    full, moe = _moe(packed, shared=True), _moe(packed, shared=True)
    moe.offload(3)
    assert moe.num_slots == 3 and all(getattr(moe, n).shape[0] == 3 for n in moe._arrays())
    assert not moe.graph_safe(torch.zeros(1, 64))
    for T in (1, 5, 40, 1):  # one token, a few, more experts than slots
        x = torch.randn(T, 64)
        torch.testing.assert_close(moe(x), full(x), atol=1e-6, rtol=0)
    assert moe.offload_streamed > 0  # experts that were not resident were read from the host copy


@torch.no_grad()
@pytest.mark.parametrize("packed", [False, True])
def test_ensure_loads_the_needed_experts_over_the_least_recently_used(packed):
    full, moe = _moe(packed), _moe(packed)
    moe.offload(3)

    def check():
        assert moe.slot_of.tolist() == moe._expert_slot
        for e, slot in enumerate(moe._expert_slot):
            if slot >= 0:
                assert moe._slot_expert[slot] == e
                for name in moe._arrays():
                    assert torch.equal(getattr(moe, name)[slot], getattr(full, name)[e])
        assert sorted(s for s in moe._expert_slot if s >= 0) == [0, 1, 2]
    check()                       # experts 0, 1, 2 resident at first
    moe._ensure([5])              # one miss: replaces a resident expert
    check()
    assert moe._expert_slot[5] >= 0 and (moe.offload_hits, moe.offload_misses) == (0, 1)
    moe._ensure([5, 6, 7])        # 5 is needed and resident: it must stay
    check()
    assert all(moe._expert_slot[e] >= 0 for e in (5, 6, 7)) and (moe.offload_hits, moe.offload_misses) == (1, 3)
    moe._ensure([6])
    moe._ensure([0])              # replaces the least recently used of 5, 6, 7: not 6, used a step ago
    check()
    assert moe._expert_slot[6] >= 0 and moe._expert_slot[0] >= 0


@torch.no_grad()
@pytest.mark.parametrize("packed", [False, True])
def test_offloaded_model_gives_the_same_logits_and_reports_its_cache(packed):
    torch.manual_seed(0)
    model = SubspaceCausalLM(_cfg(layers=2, shared=True)).eval()
    if packed:
        quantize_model(model, "rtn")
    ids = torch.randint(0, 128, (1, 20))
    logits, _ = model(ids)
    assert model.expert_offload_stats() is None
    assert model.offload_experts(4) == 2
    half = SubspaceCausalLM(model.config)
    with pytest.raises(ValueError, match="every layer"):  # 7 and 8 of 8: refused before any layer changes
        half.offload_experts(7.5)
    assert half.expert_offload_stats() is None
    assert half.offload_experts(4.5) == 2 and [layer.mlp.num_slots for layer in half.layers] == [4, 5]  # an average
    assert [layer.mlp._small_steps for layer in half.layers] == [0, 32]  # they look at their residents in turn
    out, cache = model(ids)
    torch.testing.assert_close(out, logits, atol=1e-5, rtol=0)
    step, _ = model(ids[:, :1], past_key_values=cache)  # a decode step on the offloaded layers
    assert torch.isfinite(step).all()
    stats = model.expert_offload_stats()
    assert stats["slots_per_layer"] == 4 and stats["experts_per_layer"] == 8


def test_offload_is_refused_where_it_cannot_be_honoured(tmp_path):
    moe = _moe()
    for bad in (0, 8, 9):
        with pytest.raises(ValueError, match="expert slots"):
            moe.offload(bad)
    moe.offload(2)
    with pytest.raises(ValueError, match="already offloaded"):
        moe.offload(2)
    torch.manual_seed(0)
    model = SubspaceCausalLM(_cfg()).eval()
    model.offload_experts(2)
    with pytest.raises(ValueError, match="before offloading"):
        quantize_model(model, "rtn")
    with pytest.raises(ValueError, match="before offloading"):
        save_quantized(model, str(tmp_path))
    with pytest.raises(ValueError, match="MoE layers"):
        SubspaceCausalLM(ModelConfig(name="dense", hidden_dim=64, ffn_dim=64, num_layers=1, num_heads=4,
                                     num_kv_heads=4, head_dim=16, vocab_size=128)).offload_experts(2)


@cuda
@torch.no_grad()
@pytest.mark.parametrize("packed", [False, True])
def test_offloaded_layer_on_cuda_uses_the_grouped_kernels_over_its_slots(packed):
    kw = dict(E=8, H=256, F=256, shared=True)
    full = _moe(packed, **kw).to("cuda", torch.float16)
    moe = _moe(packed, **kw)
    moe.offload(4)
    moe = moe.to("cuda", torch.float16)
    assert moe.slot_of.is_cuda and all(getattr(moe, n).is_cuda for n in moe._arrays())
    assert not any(t.is_cuda for t in moe._host.values())
    torch.manual_seed(1)
    for T in (1, 2, 1, 2, 64, 1):  # decode-sized steps that miss, then one that needs more experts than slots
        x = torch.randn(T, 256, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=4e-3, rtol=0)
    from turing.models import causal_lm
    on_cpu = packed and causal_lm.w4_expert_cpu is not None  # packed experts that miss run on the CPU: no eviction
    assert (moe.offload_cpu if on_cpu else moe.offload_misses) > 0 and moe.offload_hits > 0
    assert moe.offload_streamed > 0
    assert moe.slot_of.tolist() == moe._expert_slot


def _greedy_through_the_engine(model, n=12):
    import asyncio
    from turing.config import TuringConfig
    from turing.serving.engine import ContinuousBatchEngine
    model.config.kv_pool_gib = 0.01
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu", max_batch_size=4), model=model,
                                   prefix_caching=False)

    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(prompt_tokens=p, max_new_tokens=n, temperature=0.0, stop_token_ids=set())
                    for p in ([5, 9, 13, 7, 21, 33, 40, 2], [17, 8, 3, 11, 29, 64])]
            outs = []
            for r in reqs:  # not a nested comprehension: Python 3.10 has no async comprehension inside a plain one
                outs.append([t async for t in engine.stream_request(r)])
            return outs
        finally:
            await engine.stop()
    return asyncio.run(go()), engine.get_telemetry()


def test_a_converted_model_offloaded_after_loading_generates_the_same_tokens(tmp_path):
    """The flow of `turing serve <dir> --moe-expert-slots N`: packed weights loaded in host memory, experts offloaded,
    then served."""
    from turing.models.quant import load_quantized
    torch.manual_seed(0)
    model = SubspaceCausalLM(_cfg(layers=2, shared=True)).eval()
    quantize_model(model, "rtn")
    save_quantized(model, str(tmp_path))
    ref, telemetry = _greedy_through_the_engine(load_quantized(str(tmp_path))[0])
    assert "moe_offload" not in telemetry
    loaded, _ = load_quantized(str(tmp_path))
    loaded.offload_experts(3)
    out, telemetry = _greedy_through_the_engine(loaded)
    assert out == ref
    assert telemetry["moe_offload"]["slots_per_layer"] == 3


def _expert_reference(full, x, weights, experts, which):
    """sum over the pairs routed to the experts in `which` of weight * expert(x), from the dequantized weights."""
    out = torch.zeros(x.shape)
    for e in which:
        gate_up, down = full.expert_weights(e, torch.float32)
        rows, cols = (experts == e).nonzero(as_tuple=True)
        gate, up = (x[rows] @ gate_up.T).chunk(2, dim=-1)
        out[rows] += (torch.nn.functional.silu(gate) * up) @ down.T * weights[rows, cols, None]
    return out


@torch.no_grad()
def test_missing_packed_experts_run_on_the_cpu_from_host_memory():
    """SparseMoE._cpu_experts (csrc/turing_w4_expert_cpu.hpp): the experts that are not resident, computed from the
    packed host arrays, give what the dequantized experts give; nothing is copied and no slot changes."""
    from turing.models import causal_lm
    if causal_lm.w4_expert_cpu is None:
        pytest.skip("the native extension is not built")
    full, moe = _moe(True, H=128, F=192), _moe(True, H=128, F=192)
    moe.offload(5)
    x = torch.randn(4, 128)
    weights, experts = moe.route(x)
    experts[0] = torch.tensor([6, 1])  # make sure experts off the device are needed, one by two tokens
    experts[1] = torch.tensor([7, 6])
    needed = torch.unique(experts).tolist()
    missing = [e for e in needed if moe._expert_slot[e] < 0]
    assert set(missing) == {6, 7}
    out = moe._cpu_experts(x, weights.float(), experts, needed, missing)
    ref = _expert_reference(full, x, weights.float(), experts, missing)
    torch.testing.assert_close(out, ref, atol=1e-5 * float(ref.abs().max()) + 1e-7, rtol=1e-4)
    assert moe.offload_cpu == 2 and moe.offload_misses == 0 and moe._slot_expert == [0, 1, 2, 3, 4]


@torch.no_grad()
def test_the_experts_used_most_become_resident():
    from turing.models import causal_lm
    if causal_lm.w4_expert_cpu is None:
        pytest.skip("the native extension is not built")
    moe = _moe(True, H=128, F=192)
    moe.offload(5)
    moe.REBALANCE = 8
    x, weights = torch.randn(1, 128), torch.tensor([[0.5, 0.5]])
    for step in range(40):  # expert 7 in every step, expert 0 in none; as SparseMoE.forward: count, then look
        experts = torch.tensor([[7, 1 + step % 4]])
        needed = sorted(set(experts[0].tolist()))
        moe._count(needed)
        missing = [e for e in needed if moe._expert_slot[e] < 0]
        if missing:
            moe._cpu_experts(x, weights, experts, needed, missing)
    assert moe._expert_slot[7] >= 0 and moe._expert_slot[0] == -1  # 7 took the slot of the unused expert
    assert int(moe.slot_of[7]) == moe._expert_slot[7] and int(moe.slot_of[0]) == -1
    # one copy, and none after it: steps that only hit resident experts count too, so the residents keep their lead
    assert moe.offload_misses == 1 and moe.offload_cpu == 7 and sorted(moe._slot_expert) == [1, 2, 3, 4, 7]


@cuda
@torch.no_grad()
def test_a_rebalance_changes_the_slots_only_between_steps():
    """The step on which the most-used missing expert becomes resident, and the steps after it, give the resident
    layer's output: the expert swapped in is not added from the CPU as well, and the one swapped out is not dropped
    when a later step needs it."""
    from turing.models import causal_lm
    if causal_lm.w4_expert_cpu is None:
        pytest.skip("the native extension is not built")
    full, moe = _moe(True).cuda().half(), _moe(True)
    moe.offload(5)
    moe = moe.cuda().half()
    moe.REBALANCE = 8
    g = torch.Generator(device="cuda").manual_seed(4)
    weights = torch.tensor([[0.6, 0.4]], device="cuda")
    for step in range(23):  # expert 7 in every step; expert 0 only once it has lost its slot to 7 (the 8th step)
        experts = torch.tensor([[7, 0 if step >= 16 else 1 + step % 4]], device="cuda")
        full.route = moe.route = lambda x, e=experts: (weights, e)
        x = torch.randn(1, 64, device="cuda", generator=g).half()
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=2e-3, rtol=2e-2)
    assert moe._expert_slot[7] >= 0 and moe._expert_slot[0] == -1 and moe.offload_misses == 1
    assert moe.offload_cpu == 7 + 7  # expert 7 before the swap, expert 0 from step 16 on


@cuda
@torch.no_grad()
def test_offloaded_layer_on_cuda_runs_missing_packed_experts_on_the_cpu_for_a_few_tokens():
    from turing.models import causal_lm
    if causal_lm.w4_expert_cpu is None:
        pytest.skip("the native extension is not built")
    full, moe = _moe(True, shared=True).cuda().half(), _moe(True, shared=True)
    moe.offload(3)
    moe = moe.cuda().half()
    g = torch.Generator(device="cuda").manual_seed(3)
    for T in (1, 2, 8):
        x = torch.randn(T, 64, device="cuda", generator=g).half()
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=2e-3, rtol=2e-2)
    assert moe.offload_cpu > 0 and moe.offload_misses == 0 and moe._slot_expert == [0, 1, 2]
    x = torch.randn(40, 64, device="cuda", generator=g).half()  # more rows than CPU_ROWS: the slots are used as before
    torch.testing.assert_close(moe(x).float(), full(x).float(), atol=2e-3, rtol=2e-2)
