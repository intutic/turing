# Copyright 2026 Ishan Gupta
# SPDX-License-Identifier: Apache-2.0
import torch
import transformers as tf

from kv_layer_guard import RoundTripCache, choose_layers, roundtrip


def _qwen2(outlier_layer=None, bias=300.0):
    torch.manual_seed(0)
    cfg = tf.Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    model = tf.Qwen2ForCausalLM(cfg).eval()
    if outlier_layer is not None:
        with torch.no_grad():  # Qwen2's layer 0: large query and key biases in the same channels dominate the scores
            attn = model.model.layers[outlier_layer].self_attn
            attn.k_proj.bias[::16] = bias
            attn.q_proj.bias[::16] = bias / 10
    return model


@torch.no_grad()
def test_no_round_trip_layers_is_the_exact_model():
    model = _qwen2()
    ids = torch.randint(0, 128, (1, 20))
    plain = model(ids).logits
    torch.testing.assert_close(model(ids, past_key_values=RoundTripCache(model.config, (), "fp8"),
                                     use_cache=True).logits, plain, rtol=0, atol=0)


def test_round_trips():
    x = torch.randn(2, 3, 5, 64) * 4
    for fmt, levels in (("int8", 255), ("int4", 15)):
        step = (x.amax(-1, keepdim=True) - x.amin(-1, keepdim=True)) / levels
        assert ((roundtrip(x, fmt) - x).abs() <= step / 2 + 1e-5).all()
    big = torch.tensor([1000.0, -1000.0, 3.0])
    assert roundtrip(big, "fp8").tolist() == [448.0, -448.0, 3.0]  # e4m3 saturates; 3.0 is exact


def test_finds_the_layer_a_key_outlier_breaks():
    model = _qwen2(outlier_layer=1)
    # a tiny random model's next-token distribution is near uniform, so its KL is small in absolute terms; the
    # default 0.02 is set for real models (Qwen2.5-1.5B layer 0: 0.13, median layer 0.0006)
    layers, kls = choose_layers(model, "fp8", threshold=1e-3, n=2, tokens=64)
    assert layers == [1], kls
    assert kls[1] > 10 * max(k for i, k in enumerate(kls) if i != 1)
