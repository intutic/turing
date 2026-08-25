import pytest
import torch
from turing.models.registry import get_model_config
from turing.models.causal_lm import SubspaceCausalLM
from turing.models.tensor_parallel import ColumnParallelLinear, RowParallelLinear

def test_subspace_causal_lm_forward_and_generate():
    config = get_model_config("test-tiny")
    model = SubspaceCausalLM(config).eval()

    input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with torch.inference_mode():
        logits, past_kv = model(input_ids)

    assert logits.shape == (1, 4, config.vocab_size)
    assert len(past_kv) == config.num_layers

    # Test generation
    prompt = [10, 20, 30]
    out_tokens = model.generate(prompt, max_new_tokens=8, temperature=0.0)
    assert len(out_tokens) == len(prompt) + 8

def test_tensor_parallel_linear():
    col_lin = ColumnParallelLinear(in_features=64, out_features=128, tp_world_size=1)
    row_lin = RowParallelLinear(in_features=128, out_features=64, tp_world_size=1)

    x = torch.randn(2, 64)
    h = col_lin(x)
    assert h.shape == (2, 128)
    out = row_lin(h)
    assert out.shape == (2, 64)
