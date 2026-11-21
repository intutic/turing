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
    for T in (1, 2, 1, 2, 64, 1):  # decode-sized steps that evict, then one that needs more experts than slots
        x = torch.randn(T, 256, device="cuda", dtype=torch.float16)
        torch.testing.assert_close(moe(x).float(), full(x).float(), atol=4e-3, rtol=0)
    assert moe.offload_misses > 0 and moe.offload_hits > 0 and moe.offload_streamed > 0
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
