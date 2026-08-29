"""
Unit tests for Fused Shannon Entropy Kernel and Epistemic Gate.
Verifies numerical equivalence between fused entropy calculation and PyTorch softmax reduction.
"""

import pytest
import torch
import torch.nn.functional as F

from turing.kernels.shannon_entropy_cuda import fused_shannon_entropy_cuda
from turing.demo.epistemic_gate import EpistemicUncertaintyGate


def test_shannon_entropy_numerical_parity():
    torch.manual_seed(42)
    batch_size, vocab_size = 4, 32000
    logits = torch.randn(batch_size, vocab_size, dtype=torch.float32)

    # Reference PyTorch
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    ref_entropy = -torch.sum(probs * log_probs, dim=-1)

    # Fused entropy function (handles CPU fallback and GPU)
    entropy_out = fused_shannon_entropy_cuda(logits)

    torch.testing.assert_close(entropy_out, ref_entropy, rtol=1e-4, atol=1e-4)


def test_epistemic_uncertainty_gate():
    gate = EpistemicUncertaintyGate(uncertainty_threshold=2.0)

    # Confident logits (low entropy)
    sharp_logits = torch.zeros(1, 1000)
    sharp_logits[0, 42] = 50.0 # Extreme confidence
    res_confident = gate.evaluate_step_uncertainty(sharp_logits)
    assert res_confident["action"] == "CONFIDENT_EXECUTION"
    assert res_confident["is_uncertain"] is False
    assert res_confident["entropy"] < 0.1

    # Uniform logits (high entropy)
    uniform_logits = torch.ones(1, 1000)
    res_uncertain = gate.evaluate_step_uncertainty(uniform_logits)
    assert res_uncertain["action"] == "TRIGGER_EPISTEMIC_EXPLORATION"
    assert res_uncertain["is_uncertain"] is True
    assert res_uncertain["entropy"] > 6.0
