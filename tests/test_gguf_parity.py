"""
GGUF loader parity against the reference implementations, no mocks:
- dequantization of every supported GGML type, in Python and in C++, bit for bit against gguf-py (ggml's Python port);
- a tiny random HF model of every architecture GGUFModelLoader reads (llama with linear / yarn / llama3 RoPE and
  Mixtral experts, qwen2, qwen3, qwen2moe, qwen3moe, olmoe, gemma, gemma2, gemma3, phi2, phi3 with LongRoPE, gpt2,
  gptneox) written to GGUF the way llama.cpp's convert_hf_to_gguf.py does, loaded through GGUFModelLoader, and matched
  to the HF logits (fp32, CPU) for a prefill and decode steps, in F32 and Q8_0;
- the standalone C++ engine on the same Llama file;
- files that do not match (a tensor removed, cut short, unsupported architecture or RoPE scaling) fail loudly.
Real files against llama.cpp itself: scripts/eval_gguf_parity.py.
"""

import math
import shutil

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as QT  # noqa: E402
from gguf import quants  # noqa: E402
import transformers as tf  # noqa: E402

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
_COMMON = dict(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=512, rms_norm_eps=1e-5)
_MOE = dict(num_experts=4, num_experts_per_tok=2)


def _rope(theta: float = 10000.0, **kw) -> dict:
    return {"rope_type": "default", "rope_theta": theta, **kw}


def _longrope(head_dim: int) -> dict:
    n = head_dim // 2
    return {"rope_type": "longrope", "rope_theta": 10000.0, "original_max_position_embeddings": 16,
            "short_factor": [1.0 + 0.1 * i for i in range(n)], "long_factor": [1.0 + 0.5 * i for i in range(n)]}


_LLAMA3 = {"rope_type": "llama3", "rope_theta": 10000.0, "factor": 8.0, "low_freq_factor": 1.0,
           "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
_NO_RMS = {k: v for k, v in _COMMON.items() if k not in ("rms_norm_eps", "num_key_value_heads")}

# HF models llama.cpp converts, in the shapes the GGUF conventions can express (Gemma 1/2's converters write no RoPE
# base, so those use llama.cpp's default 1e4; Gemma 2/3's query_pre_attn_scalar is the head size, as in every model but
# the 27B ones, whose scale llama.cpp derives from n_embd / n_head; the MoE top-k renormalisation is llama.cpp's per arch)
HF_MODELS = {
    "llama": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**_COMMON, rope_parameters=_rope())),
    "llama3": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**{**_COMMON, "hidden_size": 128, "num_attention_heads": 2,
                                                            "num_key_value_heads": 1, "max_position_embeddings": 131072},
                                                         rope_parameters=_LLAMA3)),
    "llama_linear": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**_COMMON, rope_parameters=_rope(factor=4.0, rope_type="linear"))),
    "llama_yarn": lambda: tf.LlamaForCausalLM(tf.LlamaConfig(**{**_COMMON, "max_position_embeddings": 16384},
                                                             rope_parameters=_rope(rope_type="yarn", factor=4.0,
                                                                                   original_max_position_embeddings=4096))),
    "mixtral": lambda: tf.MixtralForCausalLM(tf.MixtralConfig(**_COMMON, num_local_experts=4, num_experts_per_tok=2,
                                                              rope_parameters=_rope(1e6))),
    "qwen2": lambda: tf.Qwen2ForCausalLM(tf.Qwen2Config(**{**_COMMON, "rms_norm_eps": 1e-6}, tie_word_embeddings=True,
                                                        rope_parameters=_rope(1e6))),
    "qwen3": lambda: tf.Qwen3ForCausalLM(tf.Qwen3Config(**{**_COMMON, "rms_norm_eps": 1e-6}, head_dim=32,
                                                        rope_parameters=_rope(1e6))),
    "qwen2_moe": lambda: tf.Qwen2MoeForCausalLM(tf.Qwen2MoeConfig(
        **_COMMON, **_MOE, moe_intermediate_size=32, shared_expert_intermediate_size=96, norm_topk_prob=False,
        rope_parameters=_rope(1e6))),
    "qwen3_moe": lambda: tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(**_COMMON, **_MOE, head_dim=32, moe_intermediate_size=32,
                                                                  norm_topk_prob=True, rope_parameters=_rope(1e6))),
    "olmoe": lambda: tf.OlmoeForCausalLM(tf.OlmoeConfig(**_COMMON, **_MOE, norm_topk_prob=False)),
    "gemma": lambda: tf.GemmaForCausalLM(tf.GemmaConfig(**_COMMON, head_dim=32)),
    "gemma2": lambda: tf.Gemma2ForCausalLM(tf.Gemma2Config(  # HF soft-caps attention scores only in eager attention
        **_COMMON, head_dim=32, query_pre_attn_scalar=32, sliding_window=6, attn_logit_softcapping=5.0,
        final_logit_softcapping=3.0, attn_implementation="eager")),
    "gemma3": lambda: tf.Gemma3ForCausalLM(tf.Gemma3TextConfig(
        **{**_COMMON, "num_hidden_layers": 6}, head_dim=32, query_pre_attn_scalar=32, sliding_window=6,
        rope_parameters={"full_attention": _rope(100000.0, rope_type="linear", factor=4.0), "sliding_attention": _rope()})),
    "phi3": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(**_COMMON, pad_token_id=0, sliding_window=6)),
    "phi3_longrope": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(**{**_COMMON, "max_position_embeddings": 64}, pad_token_id=0,
                                                              original_max_position_embeddings=16,
                                                              rope_parameters=_longrope(16))),
    "phi2": lambda: tf.PhiForCausalLM(tf.PhiConfig(**{**_NO_RMS, "intermediate_size": 256}, partial_rotary_factor=0.4)),
    "gpt2": lambda: tf.GPT2LMHeadModel(tf.GPT2Config(vocab_size=VOCAB, n_embd=64, n_layer=2, n_head=4, n_positions=128,
                                                     bos_token_id=1, eos_token_id=2)),
    "gpt_neox": lambda: tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(**_NO_RMS, rope_parameters=_rope(partial_rotary_factor=0.25))),
    "gpt_neox_sequential": lambda: tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(**_NO_RMS, use_parallel_residual=False)),
}
GGUF_ARCH = {"llama": "llama", "mixtral": "llama", "qwen2": "qwen2", "qwen3": "qwen3", "qwen2_moe": "qwen2moe",
             "qwen3_moe": "qwen3moe", "olmoe": "olmoe", "gemma": "gemma", "gemma2": "gemma2", "gemma3_text": "gemma3",
             "phi3": "phi3", "phi": "phi2", "gpt2": "gpt2", "gpt_neox": "gptneox"}
# HF module (in a layer, or at the top) -> llama.cpp tensor name (gguf-py tensor_mapping.py)
LAYER_NAMES = {
    "input_layernorm": "attn_norm", "ln_1": "attn_norm", "post_attention_layernorm": "ffn_norm", "ln_2": "ffn_norm",
    "pre_feedforward_layernorm": "ffn_norm", "post_feedforward_layernorm": "post_ffw_norm",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.qkv_proj": "attn_qkv", "attn.c_attn": "attn_qkv", "attention.query_key_value": "attn_qkv",
    "self_attn.o_proj": "attn_output", "self_attn.dense": "attn_output", "attn.c_proj": "attn_output",
    "attention.dense": "attn_output", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down", "mlp.gate_up_proj": "ffn_up",
    "mlp.fc1": "ffn_up", "mlp.fc2": "ffn_down", "mlp.c_fc": "ffn_up", "mlp.c_proj": "ffn_down",
    "mlp.dense_h_to_4h": "ffn_up", "mlp.dense_4h_to_h": "ffn_down", "mlp.gate": "ffn_gate_inp",
    "mlp.experts.down_proj": "ffn_down_exps", "mlp.shared_expert.gate_proj": "ffn_gate_shexp",
    "mlp.shared_expert.up_proj": "ffn_up_shexp", "mlp.shared_expert.down_proj": "ffn_down_shexp",
    "mlp.shared_expert_gate": "ffn_gate_inp_shexp",
}
TOP_NAMES = {"embed_tokens": "token_embd", "wte": "token_embd", "embed_in": "token_embd", "wpe": "position_embd",
             "norm": "output_norm", "final_layernorm": "output_norm", "ln_f": "output_norm",
             "final_layer_norm": "output_norm", "lm_head": "output", "embed_out": "output"}
F32_TENSORS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "position_embd.weight")  # never quantized


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


def _hf_model(kind: str, seed: int = 0, **overrides):
    torch.manual_seed(seed)
    model = HF_MODELS[kind]()
    if overrides:
        model = type(model)(type(model.config)(**{**model.config.to_dict(), **overrides}))
    # weights large enough that logits and attention patterns are far from uniform (HF's 0.02 init is near-uniform)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 1.0 / math.sqrt(p.shape[-1]))
            else:
                is_norm = name.endswith("weight") and ("norm" in name or name.split(".")[-2] in ("ln_1", "ln_2", "ln_f"))
                p.copy_(float(is_norm) + 0.2 * torch.randn_like(p))
    return model.eval()


def _gguf_tensors(hf, arch: str, fuse: bool = False) -> dict:
    """GGUF name -> float32 array, as convert_hf_to_gguf.py writes the HF model's weights for `arch`: llama.cpp names,
    q/k permutation (llama), Gemma norms stored as 1 + w, GPT-2 Conv1D weights transposed, GPT-NeoX's per-head q/k/v
    regrouped into q, k, v, expert stacks split into gate and up. fuse: write attn_qkv and ffn_gate_up_exps (the
    converter's fused layouts) instead of the separate tensors."""
    cfg = hf.config
    n_head = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_head
    out = {}
    for name, p in hf.named_parameters():
        t = p.detach().numpy().astype(np.float32)
        parts = name.split(".")
        if "layers" in parts or "h" in parts:
            i = next(j for j, s in enumerate(parts) if s.isdigit())
            layer, rest = parts[i], ".".join(parts[i + 1:])
            module, kind = (rest.rsplit(".", 1) if rest.endswith((".weight", ".bias")) else (rest, "weight"))
            prefix = f"blk.{layer}."
            if module == "mlp.experts.gate_up_proj":
                F = t.shape[1] // 2
                if fuse:
                    out[prefix + "ffn_gate_up_exps.weight"] = t
                else:
                    out[prefix + "ffn_gate_exps.weight"], out[prefix + "ffn_up_exps.weight"] = t[:, :F], t[:, F:]
                continue
            base = LAYER_NAMES[module]
            if module == "post_attention_layernorm" and arch in ("gemma2", "gemma3"):
                base = "post_attention_norm"
            gname = f"{prefix}{base}.{kind}"
        else:
            module, kind = name.rsplit(".", 1)
            if module == "lm_head" and arch in ("gemma", "gemma2"):  # the Gemma converters skip it (tied)
                continue
            gname, base = f"{TOP_NAMES[module.split('.')[-1]]}.{kind}", None
        if arch == "gpt2" and module.endswith(("c_attn", "c_proj", "c_fc")) and kind == "weight":
            t = t.T  # Conv1D
        if arch == "gptneox" and module == "attention.query_key_value":
            t = t.reshape(n_head, 3, -1, *t.shape[1:]).swapaxes(0, 1).reshape(t.shape)
        if arch == "llama" and base in ("attn_q", "attn_k"):
            t = _llama_permute(t, n_head if base == "attn_q" else n_kv)
        if arch in ("gemma", "gemma2", "gemma3") and name.endswith("norm.weight"):
            t = t + 1
        out[gname] = t
    if fuse:
        for l in range(cfg.num_hidden_layers):
            for kind in ("weight", "bias"):
                qkv = [out.pop(f"blk.{l}.attn_{p}.{kind}", None) for p in "qkv"]
                if qkv[0] is not None:
                    out[f"blk.{l}.attn_qkv.{kind}"] = np.concatenate(qkv)
    return out


def _write_gguf(path: str, hf, quant: str = "F32", drop: str = None, arch: str = None, fuse: bool = False) -> str:
    """Writes an HF model as llama.cpp's convert_hf_to_gguf.py does: tensors (see _gguf_tensors), the metadata keys its
    model class writes, tokenizer metadata, norms / 1-D / router tensors in F32 and the other weights in `quant`."""
    cfg = hf.config
    mt = cfg.model_type
    arch = arch or GGUF_ARCH[mt]
    tensors = _gguf_tensors(hf, GGUF_ARCH[mt], fuse)
    rp = getattr(cfg, "rope_parameters", None) or {}
    full = rp.get("full_attention", rp)
    hidden, n_head = cfg.hidden_size, cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_head
    rot = int(full.get("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 1.0)) * hidden) // n_head
    w = gguf.GGUFWriter(path, arch)
    w.add_name("tiny")
    w.add_block_count(cfg.num_hidden_layers)
    w.add_context_length(cfg.max_position_embeddings)
    w.add_embedding_length(hidden)
    w.add_head_count(n_head)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_Q8_0 if quant == "Q8_0" else gguf.LlamaFileType.ALL_F32)
    if mt in ("gpt2", "phi", "gpt_neox"):
        w.add_layer_norm_eps(cfg.layer_norm_epsilon if mt == "gpt2" else cfg.layer_norm_eps)
        w.add_feed_forward_length(4 * hidden if mt != "gpt_neox" else cfg.intermediate_size)
        if mt == "phi":
            w.add_head_count_kv(n_head)
        if mt != "gpt2":
            w.add_rope_dimension_count(rot)
        if mt == "gpt_neox":
            w.add_parallel_residual(cfg.use_parallel_residual)
    else:
        w.add_feed_forward_length(cfg.intermediate_size)
        w.add_head_count_kv(cfg.num_key_value_heads)
        w.add_layer_norm_rms_eps(cfg.rms_norm_eps)
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        if mt not in ("gemma", "gemma2"):  # those converters write no RoPE keys
            w.add_rope_freq_base(full["rope_theta"])
        if mt == "phi3":  # Phi3MiniModel: no scaling type key; LongRoPE as factor tensors and an attention factor
            w.add_rope_dimension_count(rot)
            w.add_rope_scaling_orig_ctx_len(full.get("original_max_position_embeddings", cfg.max_position_embeddings))
            w.add_sliding_window(cfg.sliding_window or 0)
            if full.get("rope_type") == "longrope":
                orig = full["original_max_position_embeddings"]
                scale = cfg.max_position_embeddings / orig
                w.add_rope_scaling_attn_factors(math.sqrt(1 + math.log(scale) / math.log(orig)) if scale > 1.0 else 1.0)
                tensors["rope_factors_long.weight"] = np.array(full["long_factor"], dtype=np.float32)
                tensors["rope_factors_short.weight"] = np.array(full["short_factor"], dtype=np.float32)
        elif full.get("rope_type") == "linear":
            w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
            w.add_rope_scaling_factor(full["factor"])
        elif full.get("rope_type") == "yarn":
            w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            w.add_rope_scaling_factor(full["factor"])
            w.add_rope_scaling_orig_ctx_len(full["original_max_position_embeddings"])
        elif full.get("rope_type") == "llama3":
            tensors["rope_freqs.weight"] = _llama3_rope_factors(full, head_dim)
        if "sliding_attention" in rp:
            w.add_rope_freq_base_swa(rp["sliding_attention"]["rope_theta"])
        if mt == "gemma2":
            w.add_attn_logit_softcapping(cfg.attn_logit_softcapping)
            w.add_final_logit_softcapping(cfg.final_logit_softcapping)
            w.add_sliding_window(cfg.sliding_window)
        if mt == "gemma3_text":
            w.add_sliding_window(cfg.sliding_window)
        experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
        if experts:
            w.add_expert_count(experts)
            w.add_expert_used_count(cfg.num_experts_per_tok)
        if mt in ("qwen2_moe", "qwen3_moe"):
            w.add_expert_feed_forward_length(cfg.moe_intermediate_size)
            if getattr(cfg, "shared_expert_intermediate_size", None):
                w.add_expert_shared_feed_forward_length(cfg.shared_expert_intermediate_size)
    w.add_tokenizer_model("gpt2")
    w.add_token_list([f"<tok_{i}>" for i in range(cfg.vocab_size)])
    w.add_token_types([1] * cfg.vocab_size)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    for gname, t in tensors.items():
        if gname == drop:
            continue
        if quant == "F32" or t.ndim == 1 or gname.endswith(("_norm.weight",) + F32_TENSORS):
            w.add_tensor(gname, t)
        else:
            w.add_tensor(gname, quants.quantize(t, QT[quant]), raw_dtype=QT[quant])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def _q8_roundtrip(hf):
    """The HF model with every quantized weight replaced by its Q8_0 round trip: what a Q8_0 GGUF of it holds (blocks
    run along each GGUF row, which is a GPT-2 Conv1D weight's column)."""
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if p.ndim < 2 or name.endswith(("mlp.gate.weight", "shared_expert_gate.weight", "wpe.weight")):
                continue
            conv1d = hf.config.model_type == "gpt2" and ".c_" in name
            x = p.T if conv1d else p
            deq = quants.dequantize(quants.quantize(x.contiguous().numpy(), QT.Q8_0), QT.Q8_0)
            x.copy_(torch.from_numpy(deq.reshape(x.shape)))
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
# LongRoPE picks its factors from each forward's length (HF): one 24-token prefill past the 16-token original context
LONG_PROMPT = np.random.default_rng(3).integers(0, VOCAB, 24).tolist()


@pytest.mark.parametrize("kind", sorted(HF_MODELS))
@pytest.mark.parametrize("quant", ["F32", "Q8_0"])
def test_gguf_matches_hf_logits(tmp_path, kind, quant):
    hf = _hf_model(kind)
    path = _write_gguf(str(tmp_path / f"{kind}-{quant}.gguf"), hf, quant, fuse=kind == "mixtral")
    if quant == "Q8_0":
        hf = _q8_roundtrip(hf)
    prompt, decode = (LONG_PROMPT, []) if kind == "phi3_longrope" else (PROMPT, DECODE)
    ref = _hf_logits(hf, prompt + decode)
    assert ref.std() > 0.5  # far from uniform, so a wrong weight layout cannot pass at atol 1e-3
    got = _turing_logits(path, prompt, decode)
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=0)


def test_every_gguf_arch_has_a_parity_case():
    assert {GGUF_ARCH[_hf_model(k).config.model_type] for k in HF_MODELS} == set(GGUFModelLoader.ARCHS)


def test_gguf_config_from_metadata(tmp_path):
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "q3.gguf"), _hf_model("qwen3"))).extract_config()
    assert (cfg.head_dim, cfg.rope_theta, cfg.qk_norm, cfg.attention_bias) == (32, 1e6, True, False)
    assert cfg.rms_norm_eps == pytest.approx(1e-6)  # stored as float32
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "q2.gguf"), _hf_model("qwen2"))).extract_config()
    assert cfg.attention_bias and not cfg.qk_norm and cfg.rope_scaling is None
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "y.gguf"), _hf_model("llama_yarn"))).extract_config()
    assert cfg.rope_scaling == {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
                                "attention_factor": pytest.approx(0.1 * math.log(4.0) + 1.0)}
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "g3.gguf"), _hf_model("gemma3"))).extract_config()
    assert cfg.layer_types == ["sliding_attention"] * 5 + ["full_attention"]
    assert (cfg.sliding_window, cfg.rope_theta, cfg.rope_local_theta) == (6, 1e5, 1e4)
    assert cfg.rope_scaling == {"rope_type": "linear", "factor": 4.0} and cfg.rope_local_scaling is None
    # llama.cpp scales Gemma 2 / 3 queries by 1/sqrt(n_embd / n_head) in the 27B models (46 / 62 layers)
    for layers, scale in ((2, 32 ** -0.5), (46, 16 ** -0.5)):
        hf = _hf_model("gemma2", num_hidden_layers=layers, layer_types=None)
        assert GGUFModelLoader(_write_gguf(str(tmp_path / f"g2-{layers}.gguf"), hf)).extract_config().attn_scale == scale
    cfg = GGUFModelLoader(_write_gguf(str(tmp_path / "q2m.gguf"), _hf_model("qwen2_moe"))).extract_config()
    assert (cfg.num_experts, cfg.num_experts_per_tok, cfg.moe_ffn_dim, cfg.shared_expert_ffn_dim, cfg.norm_topk_prob,
            cfg.moe_layers) == (4, 2, 32, 96, False, None)


def test_gguf_llama3_rope_scaling_matches_hf(tmp_path):
    hf = _hf_model("llama3")
    path = _write_gguf(str(tmp_path / "llama3.gguf"), hf)
    assert GGUFModelLoader(path).extract_config().rope_scaling == {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192}
    rng = np.random.default_rng(1)
    prompt, decode = rng.integers(0, VOCAB, 200).tolist(), rng.integers(0, VOCAB, 3).tolist()
    ref = _hf_logits(hf, prompt + decode)
    plain = tf.LlamaForCausalLM(tf.LlamaConfig(**{**hf.config.to_dict(), "rope_parameters": _rope()}))
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
    path = _write_gguf(str(tmp_path / "deepseek2.gguf"), _hf_model("llama"), arch="deepseek2")
    with pytest.raises(ValueError, match="architecture 'deepseek2' is not supported"):
        GGUFModelLoader(path).load(device="cpu")
    embed = _hf_model("llama").model.embed_tokens.weight.detach().numpy()
    for kind, ext, error in (("yarn", 2.0, "ext_factor"), ("ntk", None, "RoPE scaling type 'ntk'")):
        path = str(tmp_path / f"{kind}.gguf")
        w = gguf.GGUFWriter(path, "llama")
        for key, value in (("embedding_length", 64), ("block_count", 2), ("attention.head_count", 4)):
            w.add_uint32(f"llama.{key}", value)
        w.add_string("llama.rope.scaling.type", kind)
        w.add_rope_scaling_factor(4.0)
        if ext is not None:
            w.add_rope_scaling_yarn_ext_factor(ext)
        w.add_tensor("token_embd.weight", embed)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        with pytest.raises(ValueError, match=error):
            GGUFModelLoader(path).extract_config()
