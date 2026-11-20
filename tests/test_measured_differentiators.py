"""Core computations behind the W6 evaluation scripts (scripts/eval_xkv_transfer.py, eval_lineage.py, eval_kslot.py,
eval_debate.py) and the cross_model_kv fixes they rely on."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from turing.config import ModelConfig
from turing.experimental.cross_model_kv import CrossModelKVPipeline, RoPEContentDecoupler


def test_rope_4d_matches_per_slice():
    """[B, S, H, D] must rotate like each [S, D] slice (the native op used to return 4-D input unchanged)."""
    k = torch.randn(2, 12, 3, 16)
    for fn in (RoPEContentDecoupler.strip_rope, RoPEContentDecoupler.apply_rope):
        out = fn(k, base=10000.0)
        ref = torch.stack([torch.stack([fn(k[b, :, h], base=10000.0) for h in range(3)], 1) for b in range(2)])
        assert not torch.equal(out, k)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_pipeline_fit_recovers_linear_map_across_head_dims():
    torch.manual_seed(0)
    src = ModelConfig(name="s", hidden_dim=32, ffn_dim=64, num_heads=4, num_kv_heads=2, head_dim=8, num_layers=3,
                      rope_theta=10000.0)
    tgt = ModelConfig(name="t", hidden_dim=64, ffn_dim=64, num_heads=4, num_kv_heads=2, head_dim=16, num_layers=2,
                      rope_theta=50000.0)
    pipe = CrossModelKVPipeline(src, tgt, top_k_layers=2, ridge_lambda=1e-4)
    B, S = 6, 40

    def make(n):
        sk = [torch.randn(n, S, 2, 8) for _ in range(3)]
        sv = [torch.randn(n, S, 2, 8) for _ in range(3)]
        tk, tv = [], []
        for t in range(2):
            idx = pipe.select_source_layer_indices(t)
            fk = torch.cat([sk[i].reshape(n, S, -1) for i in idx], -1)
            fv = torch.cat([sv[i].reshape(n, S, -1) for i in idx], -1)
            tk.append(RoPEContentDecoupler.apply_rope((fk @ W[t][0]).view(n, S, 2, 16), base=50000.0))
            tv.append((fv @ W[t][1] + 0.5).view(n, S, 2, 16))
        sk = [RoPEContentDecoupler.apply_rope(k, base=10000.0) for k in sk]
        return sk, sv, tk, tv

    W = [(torch.randn(32, 32) * 0.3, torch.randn(32, 32) * 0.3) for _ in range(2)]
    pipe.fit(*make(B))
    sk, sv, tk, tv = make(1)
    ok, ov = pipe.transfer_cache(sk, sv)
    for a, b in zip(ok + ov, tk + tv):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    from turing.models.hf_loader import RealHuggingFaceLoader
    torch.manual_seed(0)
    hf = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)).eval()
    model = RealHuggingFaceLoader.convert_hf_model(hf, device="cpu", dtype=torch.float32)
    model.config.kv_pool_gib = 0.01
    return model


def test_injected_cache_reproduces_own_prefill(tiny_model):
    from eval_xkv_transfer import cache_from_kv, continuation_nll, prefill_kv
    ids = torch.randint(0, 128, (1, 40))
    cont = torch.randint(0, 128, (1, 9))
    cache, k, v = prefill_kv(tiny_model, ids)
    own = continuation_nll(tiny_model, cache, cont)
    injected = continuation_nll(tiny_model, cache_from_kv(tiny_model, k, v), cont)
    none = continuation_nll(tiny_model, None, cont)
    assert abs(own - injected) < 1e-5
    assert abs(own - none) > 1e-4


def test_retained_fraction():
    from eval_xkv_transfer import retained
    r, se = retained([3.0, 3.2, 2.8], [2.0, 2.2, 1.8], [2.5, 2.7, 2.3])
    assert abs(r - 0.5) < 1e-12 and se < 1e-9


def test_debate_answer_extraction_and_vote():
    from eval_debate import correct, extract, vote
    assert extract("so 3 + 4 = 7. The answer is 1,250.") == 1250.0
    assert extract("We get $18 in total") == 18.0
    assert extract("no number") is None
    assert extract("so the total is \\boxed{36} and 2 more") == 36.0
    assert extract("The answer is \\boxed{12}.") == 12.0
    assert correct(extract("The answer is 42."), 42.0)
    assert vote([5.0, 7.0, 7.0]) == 7.0
    assert vote([5.0, 7.0, None]) == 5.0
    assert vote([None, None, None]) is None
