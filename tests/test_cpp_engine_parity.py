"""
Logit parity of the standalone C++ engine (turing-cli logits) against Turing's PyTorch model loaded from the same
tiny F16 GGUF: prefill and decode steps, with the fp32 KV cache and with the int8 KV cache.
"""

import json
import subprocess

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from turing.core.rope import apply_rotary_pos_emb
from turing.models.causal_lm import RMSNorm
from turing.models.gguf_loader import GGMLType, GGUFModelLoader, GGUFReader, create_test_gguf_file

HEADS, HIDDEN = 4, 64
TOKENS = [3, 17, 42, 99, 5, 64, 7, 120, 31, 2]
PREFILL = 6  # the PyTorch side prefills these in one forward, then decodes the rest one token at a time


def _unpermute(w: torch.Tensor, n_head: int) -> torch.Tensor:
    """GGUF stores llama q/k rows permuted for interleaved-pair RoPE; undo it for the half-split RoPE in PyTorch."""
    return w.reshape(n_head, w.shape[0] // n_head // 2, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    np.random.seed(1234)
    path = str(tmp_path_factory.mktemp("parity") / "tiny.gguf")
    create_test_gguf_file(path, hidden_dim=HIDDEN, ffn_dim=128, num_layers=2, num_heads=HEADS, num_kv_heads=HEADS,
                          vocab_size=128, max_position_embeddings=256, quant_type=GGMLType.F16)
    return path


@pytest.fixture(scope="module")
def torch_model(tiny_gguf):
    model, _ = GGUFModelLoader(tiny_gguf).load(device="cpu", dtype=torch.float32)
    reader = GGUFReader(tiny_gguf)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            for name, proj in (("attn_q", layer.self_attn.q_proj), ("attn_k", layer.self_attn.k_proj)):
                raw = reader.get_tensor_tensor(f"blk.{i}.{name}.weight", torch.float32)
                proj.weight.copy_(_unpermute(raw, HEADS))
    reader.close()
    for m in model.modules():  # the GGUF has no layer_norm_rms_epsilon: both engines use llama.cpp's 1e-5
        if isinstance(m, RMSNorm):
            m.eps = 1e-5
    return model.eval()


def _cpp_logits(binary, gguf, kv_dtype):
    res = subprocess.run([binary, "logits", "--model", gguf, "--tokens", ",".join(map(str, TOKENS)),
                          "--kv-cache-dtype", kv_dtype], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    out = json.loads(res.stdout)
    assert out["kv_cache_dtype"] == kv_dtype
    return torch.tensor(out["logits"], dtype=torch.float32)


def _int8_round_trip(t: torch.Tensor) -> torch.Tensor:
    """Per token and KV head, asymmetric round-to-nearest over head_dim with fp32 scale and zero."""
    lo, hi = t.amin(-1, keepdim=True), t.amax(-1, keepdim=True)
    scale = ((hi - lo) / torch.full_like(hi, 255.0)).clamp(min=1e-8)
    return torch.round((t - lo) / scale).clamp(0, 255) * scale + lo


@torch.no_grad()
def _reference_logits(model, ids, int8_kv: bool) -> torch.Tensor:
    """The PyTorch model's own modules, run as one causal forward, with an optional int8 round trip of K and V."""
    cfg = model.config
    x = model.embed_tokens(ids)
    L = ids.shape[1]
    for layer in model.layers:
        att = layer.self_attn
        h = layer.input_layernorm(x)
        q = att.q_proj(h).view(1, L, cfg.num_heads, cfg.head_dim).transpose(1, 2)
        k = att.k_proj(h).view(1, L, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        v = att.v_proj(h).view(1, L, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = att.rope.compute_freqs(L, device=ids.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if int8_kv:
            k, v = _int8_round_trip(k), _int8_round_trip(v)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + att.o_proj(o.transpose(1, 2).reshape(1, L, -1))
        h = layer.post_attention_layernorm(x)
        x = x + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h))
    return model.lm_head(model.norm(x))[0]


@torch.no_grad()
def _model_logits(model) -> torch.Tensor:
    """SubspaceCausalLM's serving path: one prefill forward through a KVCache, then single-token decode steps."""
    ids = torch.tensor([TOKENS])
    logits, cache = model(ids[:, :PREFILL])
    rows = [logits[0]]
    for t in TOKENS[PREFILL:]:
        logits, cache = model(torch.tensor([[t]]), past_key_values=cache)
        rows.append(logits[0])
    return torch.cat(rows)


def test_cpp_engine_matches_pytorch_fp32_kv(turing_cli_binary, tiny_gguf, torch_model):
    expected = _model_logits(torch_model)
    got = _cpp_logits(turing_cli_binary, tiny_gguf, "native")
    assert got.shape == expected.shape == (len(TOKENS), 128)
    assert torch.allclose(got, expected, atol=1e-3, rtol=0), (got - expected).abs().max()


def test_reference_forward_matches_model(torch_model):
    """The hand-written reference used for the int8 comparison is the model's own forward."""
    ref = _reference_logits(torch_model, torch.tensor([TOKENS]), int8_kv=False)
    assert torch.allclose(ref, _model_logits(torch_model), atol=1e-3, rtol=0)


def test_cpp_engine_matches_pytorch_int8_kv(turing_cli_binary, tiny_gguf, torch_model):
    expected = _reference_logits(torch_model, torch.tensor([TOKENS]), int8_kv=True)
    got = _cpp_logits(turing_cli_binary, tiny_gguf, "int8")
    assert torch.allclose(got, expected, atol=1e-3, rtol=0), (got - expected).abs().max()
    # and the int8 cache really is lossy: it differs from the fp32 engine
    assert not torch.allclose(got, _cpp_logits(turing_cli_binary, tiny_gguf, "native"), atol=1e-3, rtol=0)
