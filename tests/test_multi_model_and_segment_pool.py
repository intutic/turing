import pytest
import torch
import numpy as np

from turing.kernels.triton_segment_pool import (
    fused_segment_mean_pool_cuda,
    fused_segment_cls_pool_cuda,
)
from turing.serving.traffic import (
    ComputeCostEstimator,
    AdaptiveComputeBatcher,
)
from turing.serving.multi_model import (
    ElasticMultiModelManager,
    TaskType,
    ModelPipelineMetadata,
)
from turing.serving.server import (
    create_app,
    EmbeddingRequest,
    RerankRequest,
    ExtractRequest,
)
from turing.serving.engine import ContinuousBatchEngine
from turing.config import ModelConfig


def test_fused_segment_mean_pool_parity():
    """Verify PyTorch fallback in fused_segment_mean_pool_cuda matches reference mean."""
    torch.manual_seed(42)
    hidden_dim = 128
    lengths = [10, 25, 40, 5, 80]
    num_seqs = len(lengths)
    total_tokens = sum(lengths)

    hidden = torch.randn((total_tokens, hidden_dim), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 10, 35, 75, 80, 160], dtype=torch.int32)

    # Reference mean computation
    ref_list = []
    for i in range(num_seqs):
        st = cu_seqlens[i].item()
        en = cu_seqlens[i + 1].item()
        ref_list.append(hidden[st:en].mean(dim=0))
    ref_output = torch.stack(ref_list)

    # Fused zero-sync implementation
    fused_output = fused_segment_mean_pool_cuda(hidden, cu_seqlens, num_seqs)

    assert torch.allclose(ref_output, fused_output, atol=1e-5, rtol=1e-5)


def test_fused_segment_cls_pool_parity():
    """Verify CLS pool extracts exact first token of each segment."""
    torch.manual_seed(42)
    hidden_dim = 64
    lengths = [8, 16, 32]
    num_seqs = len(lengths)
    total_tokens = sum(lengths)

    hidden = torch.randn((total_tokens, hidden_dim), dtype=torch.float32)
    cu_seqlens = torch.tensor([0, 8, 24, 56], dtype=torch.int32)

    cls_output = fused_segment_cls_pool_cuda(hidden, cu_seqlens, num_seqs)
    for i in range(num_seqs):
        st = cu_seqlens[i].item()
        assert torch.allclose(cls_output[i], hidden[st])


def test_native_csrc_simd_segment_pool():
    """Verify native C++20 SIMD segment pooling binding."""
    try:
        import turing.turing_csrc as turing_csrc
    except ImportError:
        pytest.skip("turing_csrc extension not built")

    np.random.seed(42)
    hidden_dim = 128
    lengths = [12, 24, 48]
    num_seqs = len(lengths)
    total_tokens = sum(lengths)

    hidden = np.random.randn(total_tokens, hidden_dim).astype(np.float32)
    cu_seqlens = np.array([0, 12, 36, 84], dtype=np.int32)

    # C++ SIMD mean pool
    out_simd = turing_csrc.segment_mean_pool_fast(hidden, cu_seqlens)
    assert out_simd.shape == (num_seqs, hidden_dim)

    # Reference numpy
    for i in range(num_seqs):
        st = cu_seqlens[i]
        en = cu_seqlens[i + 1]
        ref = hidden[st:en].mean(axis=0)
        np.testing.assert_allclose(out_simd[i], ref, rtol=1e-4, atol=1e-4)

    # C++ SIMD CLS pool
    out_cls_simd = turing_csrc.segment_cls_pool_fast(hidden, cu_seqlens)
    assert out_cls_simd.shape == (num_seqs, hidden_dim)
    for i in range(num_seqs):
        st = cu_seqlens[i]
        np.testing.assert_allclose(out_cls_simd[i], hidden[st], rtol=1e-4, atol=1e-4)


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


def test_elastic_multi_model_manager_lru_lifecycle():
    """Verify dynamic loading, memory budget compliance, and LRU eviction."""
    # Budget fits 2 models (each 500MB)
    mgr = ElasticMultiModelManager(vram_budget_bytes=1000 * 1024**2, device="cpu")

    mgr.register_model("model-embed", TaskType.EMBEDDING, estimated_vram_bytes=450 * 1024**2)
    mgr.register_model("model-rerank", TaskType.RERANK, estimated_vram_bytes=450 * 1024**2)
    mgr.register_model("model-extract", TaskType.EXTRACT, estimated_vram_bytes=450 * 1024**2)

    # Load first 2 models
    mgr.get_or_load_model("model-embed")
    mgr.get_or_load_model("model-rerank")
    assert mgr.stats["allocated_bytes"] == 900 * 1024**2
    assert "model-embed" in mgr.stats["loaded_models"]
    assert "model-rerank" in mgr.stats["loaded_models"]

    # Access model-embed to make model-rerank the LRU
    mgr.get_or_load_model("model-embed")

    # Load 3rd model -> should evict model-rerank
    mgr.get_or_load_model("model-extract")
    assert mgr.stats["eviction_count"] == 1
    assert "model-rerank" not in mgr.stats["loaded_models"]
    assert "model-embed" in mgr.stats["loaded_models"]
    assert "model-extract" in mgr.stats["loaded_models"]


def test_elastic_multi_model_manager_encode_score_extract():
    """Verify multi-model manager functional execution."""
    mgr = ElasticMultiModelManager(device="cpu")

    # 1. Encode
    texts = ["Quantum computing revolution", "Artificial intelligence systems"]
    embeddings = mgr.encode("bge-small-en", texts=texts, pool_type="mean", normalize=True)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768
    # Check normalized (L2 norm ~ 1.0)
    norm = sum(x**2 for x in embeddings[0]) ** 0.5
    assert norm == pytest.approx(1.0, rel=1e-3)

    # 2. Score / Rerank
    query = "flood insurance policy requirements"
    docs = [
        "Proof of flood insurance loss policy requirements must be submitted within 60 days.",
        "The weather today in Paris is sunny and warm.",
        "General news coverage about sports.",
    ]
    ranked = mgr.score("bge-reranker-v2", query=query, candidates=docs)
    assert len(ranked) == 3
    assert ranked[0]["rank"] == 0
    # Top ranked should have highest keyword overlap with query
    assert "flood" in ranked[0]["text"].lower()

    # 3. Extract
    doc_text = "Policyholder John Doe filed claim number POL-98432 for property damage."
    entities = mgr.extract("gliner-extract", text=doc_text, labels=["John Doe", "POL-98432"])
    assert len(entities) == 2
    assert entities[0]["label"] == "John Doe"
    assert entities[1]["label"] == "POL-98432"


def test_server_multi_task_endpoints():
    """Verify FastAPI routes for /v1/embeddings, /v1/rerank, and /v1/extract."""
    from fastapi.testclient import TestClient
    from turing.models.registry import get_model_config
    from turing.config import TuringConfig

    cfg = get_model_config("test-tiny")
    jcfg = TuringConfig(device="cpu")
    engine = ContinuousBatchEngine(cfg, jcfg)
    app = create_app(engine)
    client = TestClient(app)

    # 1. Test /v1/embeddings
    resp = client.post("/v1/embeddings", json={"input": ["hello world", "test embedding"], "model": "text-embed-3"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    assert len(data["data"][0]["embedding"]) == 1024

    # 2. Test /v1/rerank
    resp = client.post(
        "/v1/rerank",
        json={
            "query": "deep learning framework",
            "documents": ["PyTorch is a machine learning framework.", "The stock market went up today."],
            "top_n": 1,
        }
    )
    assert resp.status_code == 200
    rdata = resp.json()
    assert len(rdata["results"]) == 1
    assert rdata["meta"]["top_n"] == 1

    # 3. Test /v1/extract
    resp = client.post(
        "/v1/extract",
        json={
            "text": "Dr. Alice Smith at Oxford University announced the finding.",
            "labels": ["Dr. Alice Smith", "Oxford University"],
        }
    )
    assert resp.status_code == 200
    edata = resp.json()
    assert edata["count"] == 2
