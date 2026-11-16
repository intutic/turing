"""Exact mode: a HuggingFace Llama mapped into SubspaceCausalLM must reproduce HuggingFace's logits."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from turing.models.hf_loader import RealHuggingFaceLoader


def _tiny_llama():
    torch.manual_seed(0)
    # GQA 4:2, head_dim 32 != hidden/heads, eps 1e-5, a non-default RoPE base, and 4 layers (recirculation would fire).
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=128,
                      rms_norm_eps=1e-5, rope_theta=100000.0, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():  # HF initialises norm weights to 1, which would hide a norm that is never copied
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model


@torch.no_grad()
def test_exact_mode_matches_hf_through_prefill_chunked_prefill_and_decode():
    hf = _tiny_llama()
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
