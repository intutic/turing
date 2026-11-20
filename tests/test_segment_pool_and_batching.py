import torch

from turing.serving.traffic import (
    ComputeCostEstimator,
    AdaptiveComputeBatcher,
)


def test_compute_cost_estimator_scaling():
    """Verify task-specific compute cost calculations."""
    # Linear for embeddings
    c_embed_100 = ComputeCostEstimator.estimate_cost("embedding", seq_len=100, hidden_dim=4096)
    c_embed_200 = ComputeCostEstimator.estimate_cost("embedding", seq_len=200, hidden_dim=4096)
    assert c_embed_200 == 2 * c_embed_100

    # Quadratic for reranking
    c_rerank_10 = ComputeCostEstimator.estimate_cost("rerank", seq_len=10, num_candidates=1)
    c_rerank_20 = ComputeCostEstimator.estimate_cost("rerank", seq_len=20, num_candidates=1)
    assert c_rerank_20 == 4 * c_rerank_10

    # Generation prefill + decode
    c_gen = ComputeCostEstimator.estimate_cost("generate", seq_len=128, max_new_tokens=64)
    assert c_gen == 192


def test_adaptive_compute_batcher():
    """Verify adaptive batcher clusters cost-homogeneous requests."""
    batcher = AdaptiveComputeBatcher(max_batch_cost=500000, max_batch_size=4, cost_tolerance_ratio=0.50)

    requests = [
        {"id": "r1", "task": "embedding", "seq_len": 16, "hidden_dim": 1024},
        {"id": "r2", "task": "embedding", "seq_len": 20, "hidden_dim": 1024},
        {"id": "r3", "task": "embedding", "seq_len": 512, "hidden_dim": 1024},
        {"id": "r4", "task": "embedding", "seq_len": 520, "hidden_dim": 1024},
    ]

    batches = batcher.bucket_requests(requests)
    assert len(batches) >= 2
    # Short sequences clustered together
    assert batches[0][0]["id"] in ("r1", "r2")
    assert batches[0][1]["id"] in ("r1", "r2")


def test_adaptive_batcher_pack_varlen_tokens():
    """Verify token packing and cu_seqlens generation."""
    token_lists = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    flat, cu_seqlens = AdaptiveComputeBatcher.pack_varlen_tokens(token_lists)
    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert cu_seqlens == [0, 3, 5, 9]
