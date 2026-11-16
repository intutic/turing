"""
GGUF loader parity against the reference implementations, no mocks:
- dequantization of every supported GGML type, in Python and in C++, bit for bit against gguf-py (ggml's Python port);
- tiny random HF models (Llama with GQA, Qwen2 with biases and tied embeddings, Qwen3 with q/k norms, Llama with
  llama3 RoPE scaling) written to GGUF the way llama.cpp's convert_hf_to_gguf.py does, loaded through
  GGUFModelLoader, and matched to the HF logits (fp32, CPU) for a prefill and decode steps, in F32 and Q8_0;
- the standalone C++ engine on the same Llama file;
- files that do not match (a tensor removed, cut short, unsupported architecture or RoPE scaling) fail loudly.
"""

import math
import shutil

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as QT  # noqa: E402
from gguf import quants  # noqa: E402
from transformers import (LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM,  # noqa: E402
                          Qwen3Config, Qwen3ForCausalLM)

from turing.models.gguf_loader import GGMLType, GGUFDequantizer, GGUFModelLoader  # noqa: E402

try:
    from turing import turing_csrc
except ImportError:
    turing_csrc = None

native = pytest.mark.skipif(turing_csrc is None, reason="native extension not built")

FLOAT_TYPES = ["F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]  # gguf-py quantizes these
K_TYPES = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"]                 # gguf-py only dequantizes (not Q8_K)
# byte offsets of the fp16 scales in each K-quant block (random bytes there could be inf / NaN)
K_FP16_FIELDS = {"Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}


def _encoded(type_name: str, n_blocks: int = 16, seed: int = 0):
    """(bytes, number of values) of random data in a GGML type."""
    rng = np.random.default_rng(seed)
    qt = QT[type_name]
    per_block, block_bytes = GGML_QUANT_SIZES[qt]
    if type_name in FLOAT_TYPES:
        x = (rng.standard_normal((n_blocks, 256)) * 3).astype(np.float32)
        return quants.quantize(x, qt).tobytes(), x.size
    raw = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
    for off in K_FP16_FIELDS.get(type_name, ()):
        raw[:, off:off + 2] = rng.standard_normal((n_blocks, 1)).astype(np.float16).view(np.uint8)
    if type_name == "Q8_K":
        raw[:, 0:4] = rng.standard_normal((n_blocks, 1)).astype(np.float32).view(np.uint8)
    return raw.tobytes(), n_blocks * per_block


def _reference(type_name: str, data: bytes) -> np.ndarray:
    raw = np.frombuffer(data, dtype=np.uint8)
    if type_name == "Q8_K":  # not in gguf-py: ggml's dequantize_row_q8_K is y = d * qs (block: f32 d, 256 int8, 16 int16)
        b = raw.reshape(-1, 292)
        return (b[:, 0:4].view(np.float32) * b[:, 4:260].view(np.int8).astype(np.float32)).reshape(-1)
    return quants.dequantize(raw, QT[type_name]).reshape(-1)


def _assert_bit_exact(got: np.ndarray, ref: np.ndarray):
    assert got.dtype == np.float32 and got.shape == ref.shape
    assert np.array_equal(got.view(np.uint32), ref.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize("type_name", FLOAT_TYPES + K_TYPES)
def test_python_dequant_bit_exact(type_name):
    data, n = _encoded(type_name)
    _assert_bit_exact(GGUFDequantizer.dequantize_numpy(data, GGMLType[type_name], n), _reference(type_name, data))


@native
@pytest.mark.parametrize("type_name", FLOAT_TYPES + K_TYPES)
def test_cpp_dequant_bit_exact(type_name):
    data, n = _encoded(type_name)
    _assert_bit_exact(turing_csrc.dequantize_gguf_simd(data, int(GGMLType[type_name]), [n]), _reference(type_name, data))


def test_unsupported_or_short_data_raises():
    with pytest.raises(ValueError, match="IQ2_XXS is not supported"):
        GGUFDequantizer.dequantize(bytes(66), GGMLType.IQ2_XXS, [256])
    with pytest.raises(ValueError, match="truncated"):
        GGUFDequantizer.dequantize_numpy(bytes(100), GGMLType.Q4_K, 256)
    if turing_csrc is not None:
        with pytest.raises(RuntimeError, match="not supported"):
            turing_csrc.dequantize_gguf_simd(bytes(66), int(GGMLType.IQ2_XXS), [256])
        with pytest.raises(RuntimeError, match="truncated"):
            turing_csrc.dequantize_gguf_simd(bytes(100), int(GGMLType.Q4_K), [256])


# ---- end to end: HF model -> GGUF (convert_hf_to_gguf conventions) -> Turing ----

VOCAB = 96
HF_TO_GGUF = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
              "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
              "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
              "self_attn.q_proj.bias": "attn_q.bias", "self_attn.k_proj.bias": "attn_k.bias",
              "self_attn.v_proj.bias": "attn_v.bias", "self_attn.q_norm.weight": "attn_q_norm.weight",
              "self_attn.k_norm.weight": "attn_k_norm.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
              "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight"}


def _gguf_name(hf_name: str) -> str:
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"
    _, _, layer, rest = hf_name.split(".", 3)  # model.layers.{l}.{rest}
    return f"blk.{layer}.{HF_TO_GGUF[rest]}"


def _llama_permute(w: np.ndarray, n_head: int) -> np.ndarray:
    """convert_hf_to_gguf.py LlamaModel.permute."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def _llama3_rope_factors(rp: dict, dim: int) -> np.ndarray:
    """convert_hf_to_gguf.py LlamaModel.generate_extra_tensors (rope_freqs.weight)."""
    base = rp["rope_theta"]
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    factor, low, high, old = rp["factor"], rp["low_freq_factor"], rp["high_freq_factor"], rp["original_max_position_embeddings"]
    out = []
    for freq in freqs.tolist():
        wavelen = 2 * math.pi / freq
        if wavelen < old / high:
            out.append(1)
        elif wavelen > old / low:
            out.append(factor)
        else:
            smooth = (old / wavelen - low) / (high - low)
            out.append(1 / ((1 - smooth) / factor + smooth))
    return np.array(out, dtype=np.float32)


def _hf_model(kind: str, seed: int = 0):
    torch.manual_seed(seed)
    common = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512, rms_norm_eps=1e-5)
    if kind == "llama":
        model = LlamaForCausalLM(LlamaConfig(**common, rope_parameters={"rope_type": "default", "rope_theta": 10000.0}))
    elif kind == "llama3":
        common.update(hidden_size=128, num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=131072)
        rp = {"rope_type": "llama3", "rope_theta": 10000.0, "factor": 8.0, "low_freq_factor": 1.0,
              "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
        model = LlamaForCausalLM(LlamaConfig(**common, rope_parameters=rp))
    elif kind == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**{**common, "rms_norm_eps": 1e-6}, tie_word_embeddings=True,
                                             rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}))
    elif kind == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**{**common, "rms_norm_eps": 1e-6}, head_dim=32,
                                             rope_parameters={"rope_type": "default", "rope_theta": 1000000.0}))
    # weights large enough that logits and attention patterns are far from uniform (HF's 0.02 init is near-uniform)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2:
                p.normal_(0.0, 1.0 / math.sqrt(p.shape[1]))
            else:
                p.copy_((1.0 if "norm" in name else 0.0) + 0.2 * torch.randn_like(p))
    return model.eval()


def _write_gguf(path: str, hf, quant: str = "F32", drop: str = None, arch: str = None) -> str:
    """Writes an HF model as llama.cpp's convert_hf_to_gguf.py does: names, q/k permutation (llama), metadata keys,
    tokenizer metadata, 1-D tensors in F32 and 2-D ones in `quant`. Returns the file path."""
    cfg = hf.config
    arch = arch or {"llama": "llama", "qwen2": "qwen2", "qwen3": "qwen3"}[cfg.model_type]
    n_head, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_head
    rp = cfg.rope_parameters
    w = gguf.GGUFWriter(path, arch)
    w.add_name("tiny")
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(cfg.hidden_size)
    w.add_block_count(cfg.num_hidden_layers)
    w.add_feed_forward_length(cfg.intermediate_size)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_freq_base(rp["rope_theta"])
    w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
    w.add_rope_dimension_count(head_dim)
    w.add_vocab_size(cfg.vocab_size)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if quant == "Q8_0" else gguf.LlamaFileType.ALL_F32)
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"<tok_{i}>" for i in range(cfg.vocab_size)])
    w.add_token_types([1] * cfg.vocab_size)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    if rp.get("rope_type") == "llama3":
        w.add_tensor("rope_freqs.weight", _llama3_rope_factors(rp, head_dim))
    for name, p in hf.named_parameters():
        if name == "lm_head.weight" and cfg.tie_word_embeddings:
            continue
        gname = _gguf_name(name)
        if gname == drop:
            continue
        t = p.detach().numpy().astype(np.float32)
        if arch == "llama" and gname.endswith(("attn_q.weight", "attn_q.bias")):
            t = _llama_permute(t, n_head)
        elif arch == "llama" and gname.endswith(("attn_k.weight", "attn_k.bias")):
            t = _llama_permute(t, n_kv)
        if quant == "F32" or t.ndim == 1:
            w.add_tensor(gname, t)
        else:
            w.add_tensor(gname, quants.quantize(t, QT[quant]), raw_dtype=QT[quant])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _q8_roundtrip(hf):
    """The HF model with every 2-D weight replaced by its Q8_0 round trip: what a Q8_0 GGUF of it holds."""
    with torch.no_grad():
        for p in hf.parameters():
            if p.ndim == 2:
                deq = quants.dequantize(quants.quantize(p.numpy(), QT.Q8_0), QT.Q8_0)
                p.copy_(torch.from_numpy(deq.reshape(p.shape)))
    return hf


def _turing_logits(path: str, prompt: list, decode: list) -> torch.Tensor:
    """Turing's logits for a prefill of `prompt`, then one decode step per token of `decode`: [len(prompt + decode), V]."""
    model, tokenizer = GGUFModelLoader(path).load(device="cpu")
    assert torch.get_default_dtype() == torch.float32 and model.embed_tokens.weight.dtype == torch.float32
    assert tokenizer.vocab_size == VOCAB
    with torch.inference_mode():
        logits, cache = model(torch.tensor([prompt]))
        rows = [logits[0]]
        for tok in decode:
            step, cache = model(torch.tensor([[tok]]), past_key_values=cache)
            rows.append(step[0])
    return torch.cat(rows)


def _hf_logits(hf, ids: list) -> torch.Tensor:
    with torch.no_grad():
        return hf(torch.tensor([ids])).logits[0]


PROMPT, DECODE = [1, 17, 42, 5, 88, 23, 64, 9, 31, 77, 2, 50], [11, 60, 3, 45]


@pytest.mark.parametrize("kind", ["llama", "qwen2", "qwen3"])
@pytest.mark.parametrize("quant", ["F32", "Q8_0"])
def test_gguf_matches_hf_logits(tmp_path, kind, quant):
    hf = _hf_model(kind)
    path = _write_gguf(str(tmp_path / f"{kind}-{quant}.gguf"), hf, quant)
    if quant == "Q8_0":
        hf = _q8_roundtrip(hf)
    ref = _hf_logits(hf, PROMPT + DECODE)
    assert ref.std() > 0.5  # far from uniform, so a wrong weight layout cannot pass at atol 1e-3
    got = _turing_logits(path, PROMPT, DECODE)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=0)


def test_gguf_config_from_metadata(tmp_path):
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "q3.gguf"), _hf_model("qwen3"))).extract_config()
    assert (cfg.head_dim, cfg.rope_theta, cfg.qk_norm, cfg.attention_bias) == (32, 1e6, True, False)
    assert cfg.rms_norm_eps == pytest.approx(1e-6)  # stored as float32
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "q2.gguf"), _hf_model("qwen2"))).extract_config()
    assert cfg.attention_bias and not cfg.qk_norm and cfg.rope_scaling is None


def test_gguf_llama3_rope_scaling_matches_hf(tmp_path):
    hf = _hf_model("llama3")
    path = _write_gguf(str(tmp_path / "llama3.gguf"), hf)
    assert GGUFModelLoader(path).extract_config().rope_scaling == {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}
    rng = np.random.default_rng(1)
    prompt, decode = rng.integers(0, VOCAB, 200).tolist(), rng.integers(0, VOCAB, 3).tolist()
    ref = _hf_logits(hf, prompt + decode)
    plain = LlamaForCausalLM(LlamaConfig(**{**hf.config.to_dict(), "rope_parameters": {"rope_type": "default",
                                                                                       "rope_theta": 10000.0}}))
    plain.load_state_dict(hf.state_dict())
    assert (_hf_logits(plain.eval(), prompt + decode) - ref).abs().max() > 1e-2  # the scaling shows in the logits
    torch.testing.assert_close(_turing_logits(path, prompt, decode), ref, atol=1e-3, rtol=0)


@native
def test_cpp_engine_matches_hf_on_llama(tmp_path):
    """The standalone C++ engine reads q/k in the file's (permuted) layout and rotates interleaved pairs, which together
    equal HF's half rotation on the original weights. Its config is fixed at RoPE base 1e4 and RMS eps 1e-5."""
    hf = _hf_model("llama")
    path = _write_gguf(str(tmp_path / "llama.gguf"), hf)
    ids = PROMPT + DECODE
    got = torch.from_numpy(turing_csrc.gguf_cpp_engine_logits(path, ids))
    torch.testing.assert_close(got, _hf_logits(hf, ids), atol=1e-3, rtol=0)


def test_gguf_missing_tensor_fails_loudly(tmp_path):
    path = _write_gguf(str(tmp_path / "missing.gguf"), _hf_model("llama"), drop="blk.1.ffn_down.weight")
    with pytest.raises(ValueError, match=r"missing tensors \['blk.1.ffn_down.weight'\]"):
        GGUFModelLoader(path).load(device="cpu")


def test_gguf_truncated_file_fails_loudly(tmp_path):
    path = _write_gguf(str(tmp_path / "full.gguf"), _hf_model("llama"))
    cut = str(tmp_path / "cut.gguf")
    shutil.copy(path, cut)
    with open(cut, "r+b") as f:
        f.truncate(f.seek(0, 2) - 4096)
    with pytest.raises(ValueError, match="truncated"):
        GGUFModelLoader(cut).load(device="cpu")
    if turing_csrc is not None:
        with pytest.raises(RuntimeError, match="truncated"):
            turing_csrc.gguf_cpp_engine_logits(cut, [1, 2])


def test_gguf_unsupported_arch_and_rope_scaling_fail_loudly(tmp_path):
    path = _write_gguf(str(tmp_path / "gemma.gguf"), _hf_model("llama"), arch="gemma")
    with pytest.raises(ValueError, match="architecture 'gemma' is not supported"):
        GGUFModelLoader(path).load(device="cpu")
    linear = str(tmp_path / "linear.gguf")
    hf = _hf_model("llama")
    w = gguf.GGUFWriter(linear, "llama")
    w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    w.add_rope_scaling_factor(2.0)
    w.add_tensor("token_embd.weight", hf.model.embed_tokens.weight.detach().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with pytest.raises(ValueError, match="RoPE scaling type 'linear'"):
        GGUFModelLoader(linear).extract_config()
