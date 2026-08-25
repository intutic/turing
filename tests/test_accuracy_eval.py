"""
Unit & Integration Tests for Live Accuracy Evaluator.
"""

import pytest
import torch

from turing.serving.accuracy_eval import LiveAccuracyEvaluator, GSM8K_SAMPLE_PROBLEMS

def test_live_accuracy_evaluator_init_and_extraction():
    evaluator = LiveAccuracyEvaluator(model_id="gpt2", sparsity_ratio=0.5, device="cpu")
    assert evaluator.model is not None
    assert evaluator.tokenizer is not None

    # Test numerical answer extraction
    assert evaluator.extract_numeric_answer("The final answer is 72 clips.") == "72"
    assert evaluator.extract_numeric_answer("Total is 10.") == "10"
    assert evaluator.extract_numeric_answer("No numbers here") is None

def test_live_accuracy_evaluator_gsm8k_run():
    evaluator = LiveAccuracyEvaluator(model_id="gpt2", sparsity_ratio=0.5, device="cpu")
    res = evaluator.evaluate_gsm8k(max_samples=2)

    assert "pass_at_1_accuracy" in res
    assert res["total_samples"] == 2
    assert "avg_latency_ms" in res
    assert len(res["results"]) == 2
