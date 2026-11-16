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

def test_tensor_parallel_linear_slices():
    """Single-process: column slices concatenate to the full output; row slices' partial products sum to it (the
    all-reduce is covered by the multi-process tests in test_distributed_inference.py)."""
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 128)
    x = torch.randn(2, 64)
    cols = [ColumnParallelLinear(lin, i * 64, (i + 1) * 64) for i in range(2)]
    torch.testing.assert_close(torch.cat([c(x) for c in cols], -1), lin(x))
    down = torch.nn.Linear(128, 64)
    h = lin(x)
    rows = [RowParallelLinear(down, i * 64, (i + 1) * 64, reduce=False, first=i == 0) for i in range(2)]
    torch.testing.assert_close(rows[0](h[:, :64]) + rows[1](h[:, 64:]), down(h))
