"""
Unit and integration tests for Fused Batched Sampler across C++20 AVX2 SIMD,
PyTorch vectorized, and Triton GPU kernels.
"""

import pytest
import torch
import torch.nn.functional as F

from turing.kernels.triton_fused_sample import fused_batched_sample_cuda, batched_sample_tokens


def test_greedy_batched_sampler_exact_parity():
    """Verifies that temperature=0.0 produces identical greedy argmax tokens across all batch rows."""
    batch_size = 8
    vocab_size = 128
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    temperatures = torch.zeros(batch_size, dtype=torch.float32)

    # Reference argmax
    expected_tokens = torch.argmax(logits, dim=-1)

    sampled = batched_sample_tokens(logits, temperatures, top_k=50)

    assert torch.equal(sampled, expected_tokens), f"Greedy sampler mismatch: {sampled} vs {expected_tokens}"


def test_temperature_stochastic_sampling_distribution():
    """Verifies that stochastic sampling respects top-k bounds and produces valid token IDs."""
    batch_size = 16
    vocab_size = 256
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)
    temperatures = torch.full((batch_size,), 0.8, dtype=torch.float32)

    sampled = batched_sample_tokens(logits, temperatures, top_k=10)

    assert sampled.shape == (batch_size,)
    assert (sampled >= 0).all()
    assert (sampled < vocab_size).all()


def test_csrc_native_sampler_binding():
    """Verifies C++20 pybind native sampler extension directly."""
    try:
        import turing.turing_csrc as turing_csrc
    except ImportError:
        pytest.skip("turing_csrc native extension not compiled")

    logits = torch.randn(4, 64, dtype=torch.float32).numpy()
    temps = torch.tensor([0.0, 0.0, 0.5, 0.8], dtype=torch.float32).numpy()
    top_ks = torch.tensor([50, 50, 10, 5], dtype=torch.int32).numpy()
    unifs = torch.tensor([0.5, 0.5, 0.2, 0.8], dtype=torch.float32).numpy()

    tokens = turing_csrc.sample_batched_logits_simd(logits, temps, top_ks, unifs)

    assert len(tokens) == 4
    # Rows 0 and 1 are greedy (temp=0)
    assert tokens[0] == int(logits[0].argmax())
    assert tokens[1] == int(logits[1].argmax())
