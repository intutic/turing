"""Exact mode: a HuggingFace causal LM mapped into SubspaceCausalLM must reproduce HuggingFace's logits."""
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from turing.models.hf_loader import RealHuggingFaceLoader

# GQA 4:2, eps 1e-5, a non-default RoPE base, and 4 layers (recirculation would fire).
_COMMON = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
               num_key_value_heads=2, max_position_embeddings=128, rms_norm_eps=1e-5, tie_word_embeddings=False)
# Llama-3.1 scaling with a 16-token original context, so a 24-token sequence spans its high, medium and low bands.
_LLAMA3 = {"rope_type": "llama3", "rope_theta": 100000.0, "factor": 8.0, "low_freq_factor": 1.0,
           "high_freq_factor": 4.0, "original_max_position_embeddings": 16}

ARCHS = {
    "llama": lambda: LlamaForCausalLM(LlamaConfig(**_COMMON, head_dim=32, rope_theta=100000.0)),  # head_dim != hidden/heads
    "llama3_rope": lambda: LlamaForCausalLM(LlamaConfig(**_COMMON, head_dim=32, rope_parameters=_LLAMA3)),
    "qwen2_qkv_bias": lambda: Qwen2ForCausalLM(Qwen2Config(**_COMMON, rope_theta=100000.0)),
    "qwen3_qk_norm": lambda: Qwen3ForCausalLM(Qwen3Config(**_COMMON, head_dim=32, rope_theta=100000.0)),
}


def _tiny(arch):
    torch.manual_seed(0)
    model = ARCHS[arch]().eval()
    with torch.no_grad():  # HF initialises norm weights to 1 and biases to 0, which would hide either being dropped
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
            elif name.endswith(".bias"):
                p.uniform_(-0.5, 0.5)
    return model


def _tiny_llama():  # used by the KV cache tests
    return _tiny("llama")


@pytest.mark.parametrize("arch", sorted(ARCHS))
@torch.no_grad()
def test_exact_mode_matches_hf_through_prefill_chunked_prefill_and_decode(arch):
    hf = _tiny(arch)
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(1))
    ref = hf(ids).logits

    model = RealHuggingFaceLoader.convert_hf_model(hf, sparsity_ratio=0.0, device="cpu", enable_recirculation=False)
    logits, past = model(ids[:, :8])
    outs = [logits]
    logits, past = model(ids[:, 8:12], past_key_values=past)  # chunked prefill: start_pos > 0, seq_len > 1
    outs.append(logits)
    for t in range(12, 24):
        logits, past = model(ids[:, t:t + 1], past_key_values=past)
        outs.append(logits)

    torch.testing.assert_close(torch.cat(outs, dim=1), ref, atol=1e-3, rtol=0)
