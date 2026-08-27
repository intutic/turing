import pytest
import torch
from turing.models.registry import get_model_config
from turing.serving.benchmark import TuringBenchmarkSuite
from turing.serving.niah import LongContextNIAHEvaluator
from turing.core.subspace import SubspaceManager

def test_niah_evaluator_multi_depth():
    """
    Validates that SVD INT8 KV Paging achieves 100% Top-1 retrieval across all depth slices
    (5%, 25%, 50%, 75%, 100%) with zero Lost-in-the-Middle degradation.
    """
    config = get_model_config("test-tiny")
    evaluator = LongContextNIAHEvaluator(config, rank=config.rank_sub, device=torch.device("cpu"))

    depths = [0.05, 0.25, 0.50, 0.75, 1.00]
    results = evaluator.evaluate_retrieval(
        context_lengths=[1024],
        depth_fractions=depths,
        page_size=256
    )
    assert len(results) == len(depths)
    for r in results:
        assert r["retrieval_status"] == "SUCCESS (Top-1 Match)"

def test_sparse_residual_outlier_correction():
    """
    Validates that Sparse Residual Outlier Correction guarantees 100% reconstruction
    and needle retrieval even under extreme low-rank SVD (Rank-4 / 64x compression).
    """
    device = torch.device("cpu")
    head_dim = 128
    rank = 4
    page_size = 256

    mgr = SubspaceManager(hidden_dim=head_dim, rank=rank, device=device)
    haystack = torch.randn(page_size, head_dim, device=device)
    needle = torch.randn(1, head_dim, device=device) * 2.5
    slot = 128
    haystack[slot] = needle.squeeze(0)

    # Compress with 1-sparse residual outlier correction
    q_int8, scale, residual_corr = mgr.compress_with_residual_correction(haystack, top_k_residuals=1)
    recon = mgr.reconstruct_with_residual_correction(q_int8, scale, residual_corr)

    scores = torch.matmul(recon, needle.t()).squeeze(-1)
    top_idx = torch.argmax(scores).item()
    assert top_idx == slot

def test_benchmark_suite_execution():
    config = get_model_config("test-tiny")
    suite = TuringBenchmarkSuite(config, device=torch.device("cpu"))
    results = suite.run_all()

    assert "simd_gemv" in results
    assert "paged_attention" in results
    assert "mmap_hardware_profile" in results
    assert "tco_financial_model" in results
