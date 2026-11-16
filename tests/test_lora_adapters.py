"""Multi-LoRA: adapters loaded from PEFT checkpoints reproduce PEFT's logits, alone and mixed per batch row."""
import pytest
import torch

peft = pytest.importorskip("peft")

from turing.models.adapters import load_lora_adapter
from turing.models.hf_loader import RealHuggingFaceLoader


def _adapter(hf, path, seed, targets, rslora=False):
    torch.manual_seed(seed)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, target_modules=targets, use_rslora=rslora, init_lora_weights=False)
    m = peft.get_peft_model(hf, cfg)
    m.save_pretrained(path, save_embedding_layers=False)  # a tiny local model has no hub config to compare against
    return m


@pytest.mark.parametrize("arch,targets", [
    ("llama", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    ("qwen2_qkv_bias", ["q_proj", "v_proj"]),
    ("phi3", ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]),
    ("gemma2", ["q_proj", "down_proj"]),
])
@torch.no_grad()
def test_lora_matches_peft_alone_and_mixed_per_row(tmp_path, arch, targets):
    import copy
    from tests.test_hf_parity import _tiny
    base = _tiny(arch)
    model = RealHuggingFaceLoader.convert_hf_model(base)
    ids = torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(3))
    refs = {}
    for name, seed, rs in (("a", 1, False), ("b", 2, True)):
        pm = _adapter(copy.deepcopy(base), str(tmp_path / name), seed, targets, rs).eval()
        refs[name] = pm(ids).logits
        load_lora_adapter(model, name, str(tmp_path / name))
    ref_base = base(ids).logits
    assert set(model.adapter_names) == {"a", "b"}
    for name in ("a", "b"):
        torch.testing.assert_close(model(ids, adapters=[name])[0], refs[name], atol=1e-4, rtol=0)
    torch.testing.assert_close(model(ids)[0], ref_base, atol=1e-4, rtol=0)  # no adapter: the base model
    # one batched decode step with three requests on base, a and b
    caches = [model(ids[:, :8], adapters=[n])[1] for n in (None, "a", "b")]
    step, _ = model(ids[:, 8:9].repeat(3, 1), past_key_values=caches, adapters=[None, "a", "b"])
    for row, ref in enumerate((ref_base, refs["a"], refs["b"])):
        torch.testing.assert_close(step[row, -1], ref[0, 8], atol=1e-4, rtol=0)


def test_unsupported_adapters_raise(tmp_path):
    from tests.test_hf_parity import _tiny
    base = _tiny("gpt2")
    import warnings
    with warnings.catch_warnings():  # PEFT notes that GPT-2's Conv1D is transposed; irrelevant here
        warnings.simplefilter("ignore", UserWarning)
        peft.get_peft_model(base, peft.LoraConfig(r=2, target_modules=["c_attn"])).save_pretrained(
            tmp_path / "g", save_embedding_layers=False)
    with pytest.raises(ValueError, match="not supported"):
        load_lora_adapter(RealHuggingFaceLoader.convert_hf_model(_tiny("gpt2")), "g", str(tmp_path / "g"))


def test_engine_routes_requests_to_adapters(tmp_path):
    import asyncio
    from tests.test_hf_parity import _tiny
    from turing.config import TuringConfig
    from turing.serving.engine import ContinuousBatchEngine, RequestRejected
    base = _tiny("llama")
    model = RealHuggingFaceLoader.convert_hf_model(base)
    _adapter(base, str(tmp_path / "a"), 1, ["q_proj", "v_proj"])
    load_lora_adapter(model, "a", str(tmp_path / "a"))
    model.config.kv_pool_gib = 0.01
    engine = ContinuousBatchEngine(model.config, TuringConfig(device="cpu"), model=model)
    prompt = list(range(1, 40))

    async def go():
        await engine.start()
        try:
            reqs = [await engine.add_request(prompt_tokens=prompt, max_new_tokens=6, temperature=0.0, adapter=a,
                                             stop_token_ids=set()) for a in (None, "a", None, "a")]
            outs = await asyncio.gather(*[_all(engine, r) for r in reqs])
            with pytest.raises(RequestRejected):
                await engine.add_request(prompt_tokens=prompt, adapter="missing")
            return outs
        finally:
            await engine.stop()
    outs = asyncio.run(go())
    assert outs[0] == outs[2] and outs[1] == outs[3] and outs[0] != outs[1]
    with torch.no_grad():
        ref_a = model.generate(prompt, max_new_tokens=6, temperature=0.0, adapter="a")[len(prompt):]
    assert outs[1] == ref_a


async def _all(engine, req):
    return [t async for t in engine.stream_request(req)]
