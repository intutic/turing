import pytest
import torch
from turing.models.registry import get_model_config
from turing.serving.benchmark import TuringBenchmarkSuite
from turing.serving.niah import LongContextNIAHEvaluator

def test_niah_evaluator():
    config = get_model_config("test-tiny")
    evaluator = LongContextNIAHEvaluator(config, rank=config.rank_sub, device=torch.device("cpu"))

    results = evaluator.evaluate_retrieval(
        context_lengths=[512],
        depth_fractions=[0.5],
        page_size=256
    )
    assert len(results) == 1
    assert results[0]["retrieval_status"] == "SUCCESS (Top-1 Match)"

def test_benchmark_suite_execution():
    config = get_model_config("test-tiny")
    suite = TuringBenchmarkSuite(config, device=torch.device("cpu"))
    results = suite.run_all()

    assert "simd_gemv" in results
    assert "paged_attention" in results
    assert "mmap_hardware_profile" in results
    assert "tco_financial_model" in results
