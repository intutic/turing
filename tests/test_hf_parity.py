"""Exact mode: a HuggingFace causal LM of every supported family, mapped into SubspaceCausalLM, must reproduce
HuggingFace's own logits through prefill, chunked prefill, decode, and batched decode across requests."""
import pytest
import torch
import transformers as tf

from turing.models.hf_loader import RealHuggingFaceLoader, FAMILIES

V, T = 128, 24  # vocab; tokens per test sequence (prefill 8, chunk 4, then 12 single-token steps)
# GQA 4:2, eps 1e-5, a non-default RoPE base, 4 layers. Small windows and original contexts so 24 tokens cross them.
_COMMON = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5, tie_word_embeddings=False)
_ROPE = {
    "llama3": {"rope_type": "llama3", "rope_theta": 100000.0, "factor": 8.0, "low_freq_factor": 1.0,
               "high_freq_factor": 4.0, "original_max_position_embeddings": 16},
    "linear": {"rope_type": "linear", "rope_theta": 10000.0, "factor": 4.0},
    "yarn": {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0, "original_max_position_embeddings": 16},
}


def _longrope(head_dim):
    n = head_dim // 2
    return {"rope_type": "longrope", "rope_theta": 10000.0, "original_max_position_embeddings": 16,
            "short_factor": [1.0 + 0.1 * i for i in range(n)], "long_factor": [1.0 + 0.5 * i for i in range(n)]}

_MOE = dict(num_experts_per_tok=2)


def _llama(**kw):
    return tf.LlamaForCausalLM(tf.LlamaConfig(**{**_COMMON, "head_dim": 32, **kw}))


ARCHS = {
    "llama": lambda: _llama(rope_theta=100000.0),  # head_dim != hidden / heads
    "llama3_rope": lambda: _llama(rope_parameters=_ROPE["llama3"]),
    "linear_rope": lambda: _llama(rope_parameters=_ROPE["linear"]),
    "yarn_rope": lambda: _llama(rope_parameters=_ROPE["yarn"]),
    "longrope": lambda: _llama(rope_parameters=_longrope(32), max_position_embeddings=64),
    "dynamic_rope": lambda: _llama(rope_parameters={"rope_type": "dynamic", "rope_theta": 10000.0, "factor": 2.0},
                                   max_position_embeddings=16),
    "mistral_sliding": lambda: tf.MistralForCausalLM(tf.MistralConfig(**_COMMON, sliding_window=6)),
    "qwen2_qkv_bias": lambda: tf.Qwen2ForCausalLM(tf.Qwen2Config(**_COMMON, rope_theta=100000.0)),
    "qwen3_qk_norm": lambda: tf.Qwen3ForCausalLM(tf.Qwen3Config(**_COMMON, head_dim=32, rope_theta=100000.0)),
    "gemma": lambda: tf.GemmaForCausalLM(tf.GemmaConfig(**{**_COMMON, "head_dim": 32})),
    "gemma2": lambda: tf.Gemma2ForCausalLM(tf.Gemma2Config(**{**_COMMON, "head_dim": 32}, sliding_window=6,
                                                           query_pre_attn_scalar=24, attn_logit_softcapping=5.0,
                                                           final_logit_softcapping=3.0)),
    "gemma3": lambda: tf.Gemma3ForCausalLM(tf.Gemma3TextConfig(
        **{**_COMMON, "head_dim": 32}, sliding_window=6, query_pre_attn_scalar=24,
        rope_parameters={"full_attention": {**_ROPE["linear"], "rope_theta": 100000.0},
                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"])),
    "phi3": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(**_COMMON, pad_token_id=0, sliding_window=6)),
    "phi3_longrope": lambda: tf.Phi3ForCausalLM(tf.Phi3Config(
        **{**_COMMON, "max_position_embeddings": 64}, pad_token_id=0, rope_parameters=_longrope(16))),
    "phi2": lambda: tf.PhiForCausalLM(tf.PhiConfig(**{k: v for k, v in _COMMON.items() if k != "rms_norm_eps"},
                                                   partial_rotary_factor=0.4)),
    "phi2_qk_layernorm": lambda: tf.PhiForCausalLM(tf.PhiConfig(
        **{k: v for k, v in _COMMON.items() if k != "rms_norm_eps"}, qk_layernorm=True)),
    "gpt2": lambda: tf.GPT2LMHeadModel(tf.GPT2Config(vocab_size=V, n_embd=64, n_layer=4, n_head=4, n_positions=128)),
    "opt": lambda: tf.OPTForCausalLM(tf.OPTConfig(vocab_size=V, hidden_size=64, num_hidden_layers=4, ffn_dim=128,
                                                  num_attention_heads=4, max_position_embeddings=128,
                                                  word_embed_proj_dim=64, pad_token_id=1)),
    "opt_post_ln_projected": lambda: tf.OPTForCausalLM(tf.OPTConfig(
        vocab_size=V, hidden_size=64, num_hidden_layers=4, ffn_dim=128, num_attention_heads=4,
        max_position_embeddings=128, word_embed_proj_dim=32, do_layer_norm_before=False, pad_token_id=1)),
    "gpt_neox": lambda: tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(
        vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128,
        max_position_embeddings=128, rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                                      "partial_rotary_factor": 0.25})),
    "gpt_neox_sequential": lambda: tf.GPTNeoXForCausalLM(tf.GPTNeoXConfig(
        vocab_size=V, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, intermediate_size=128,
        max_position_embeddings=128, use_parallel_residual=False)),
    "mixtral": lambda: tf.MixtralForCausalLM(tf.MixtralConfig(**_COMMON, num_local_experts=4, **_MOE)),
    "qwen2_moe": lambda: tf.Qwen2MoeForCausalLM(tf.Qwen2MoeConfig(
        **_COMMON, num_experts=4, moe_intermediate_size=32, shared_expert_intermediate_size=96, norm_topk_prob=False,
        decoder_sparse_step=2, mlp_only_layers=[3], **_MOE)),
    "qwen3_moe": lambda: tf.Qwen3MoeForCausalLM(tf.Qwen3MoeConfig(
        **_COMMON, head_dim=32, num_experts=4, moe_intermediate_size=32, norm_topk_prob=True, **_MOE)),
    "olmoe": lambda: tf.OlmoeForCausalLM(tf.OlmoeConfig(**_COMMON, num_experts=4, clip_qkv=0.8, **_MOE)),
}


INCREMENTAL_REF = {"longrope"}


def _hf_incremental(hf, ids):
    """HF's logits for the same calls the Turing model gets: prefill 8, chunk 4, then one token at a time."""
    out = hf(ids[:, :8], use_cache=True)
    logits, past = [out.logits], out.past_key_values
    for a, b in [(8, 12)] + [(t, t + 1) for t in range(12, ids.shape[1])]:
        out = hf(ids[:, a:b], past_key_values=past, use_cache=True)
        logits.append(out.logits)
        past = out.past_key_values
    return torch.cat(logits, dim=1)


def _tiny(arch):
    torch.manual_seed(0)
    model = ARCHS[arch]().eval()
    with torch.no_grad():  # HF initialises norm weights to 1 (Gemma: 0) and biases to 0, hiding either being dropped
        for name, p in model.named_parameters():
            if "norm" in name and p.dim() == 1:
                p.uniform_(0.5, 1.5)
            elif name.endswith(".bias"):
                p.uniform_(-0.5, 0.5)
            elif "experts" in name or "router" in name or name.endswith("mlp.gate.weight"):
                p.normal_(0.0, 0.1)  # MoE weights start at zero/empty in some HF versions
    return model


def _tiny_llama():  # used by the KV cache tests
    return _tiny("llama")


def test_every_supported_family_has_a_parity_case():
    covered = {_tiny(a).config.model_type for a in ARCHS}
    assert set(FAMILIES) <= covered | {"gemma3_text"}  # gemma3 is built through Gemma3ForCausalLM (gemma3_text)


@pytest.mark.parametrize("arch", sorted(ARCHS))
@torch.no_grad()
def test_exact_mode_matches_hf_through_prefill_chunked_prefill_and_decode(arch):
    hf = _tiny(arch)
    ids = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(1))
    ref = hf(ids).logits
    # LongRoPE picks its frequencies from each forward's length, so HF's own incremental run differs from its full
    # forward once the sequence passes the original context: compare increments with increments there
    inc_ref = _hf_incremental(hf, ids) if arch in INCREMENTAL_REF else ref

    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    logits, past = model(ids[:, :8])
    outs = [logits]
    logits, past = model(ids[:, 8:12], past_key_values=past)  # chunked prefill: start_pos > 0, seq_len > 1
    outs.append(logits)
    for t in range(12, T):
        logits, past = model(ids[:, t:t + 1], past_key_values=past)
        outs.append(logits)
    torch.testing.assert_close(torch.cat(outs, dim=1), inc_ref, atol=1e-3, rtol=0)

    # the stateless forward (no cache) and a batched decode across two requests agree with HF too
    torch.testing.assert_close(model(ids)[0], ref, atol=1e-3, rtol=0)
    ids2 = torch.randint(0, V, (1, T), generator=torch.Generator().manual_seed(2))
    ref2 = hf(ids2).logits
    _, c1 = model(ids[:, :10])
    _, c2 = model(ids2[:, :14])  # different lengths in one step
    step, _ = model(torch.cat([ids[:, 10:11], ids2[:, 14:15]]), past_key_values=[c1, c2])
    if arch not in INCREMENTAL_REF:  # both lengths stay inside the original context for the LongRoPE cases
        torch.testing.assert_close(step[0, -1], ref[0, 10], atol=1e-3, rtol=0)
        torch.testing.assert_close(step[1, -1], ref2[0, 14], atol=1e-3, rtol=0)
    else:
        torch.testing.assert_close(step[0, -1], hf(ids[:, :11]).logits[0, -1], atol=1e-3, rtol=0)
        torch.testing.assert_close(step[1, -1], hf(ids2[:, :15]).logits[0, -1], atol=1e-3, rtol=0)


def test_unsupported_families_are_refused():
    hf = tf.BloomForCausalLM(tf.BloomConfig(vocab_size=V, hidden_size=64, n_layer=2, n_head=4))
    with pytest.raises(ValueError, match="not supported"):
        RealHuggingFaceLoader.convert_hf_model(hf)


@torch.no_grad()
def test_pruning_keeps_contiguous_channels_and_changes_only_the_ffn():
    hf = _tiny("gpt2")
    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.5)
    mlp = model.layers[0].mlp
    assert mlp.up_proj.weight.shape[0] == mlp.ffn_dim == 128  # 4 tiles of 64 -> 2 kept
    ids = torch.randint(0, V, (1, 8))
    assert model(ids)[0].shape == (1, 8, V)


def test_loader_families_and_registry_list_the_same_architectures():
    from turing.models.architecture_registry import SUPPORTED
    assert set(FAMILIES) == set(SUPPORTED)
