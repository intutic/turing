"""
Comprehensive Benchmark: Multi-Task Pipeline Serving, Zero-Sync Segment Pooling & Compute-Cost Batching.
Compares Turing Engine's optimized kernels and residency against standard SIE/TEI patterns on physical silicon.
"""

import time
import json
import torch
import numpy as np
from typing import Dict, Any, List

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
)


def benchmark_segment_pooling(device: torch.device, num_seqs: int = 64, hidden_dim: int = 1024, iterations: int = 100) -> Dict[str, Any]:
    """
    Compares In-SRAM Fused Segment Mean-Pooling vs. Naive SIE Python Loop with .item() synchronization.
    """
    # Generate variable-length sequence lengths (e.g. 16 to 256 tokens per sequence)
    np.random.seed(42)
    seq_lengths = np.random.randint(16, 256, size=num_seqs).tolist()
    total_tokens = sum(seq_lengths)

    hidden = torch.randn((total_tokens, hidden_dim), device=device, dtype=torch.float32)

    cu_seqlens_list = [0]
    cum = 0
    for length in seq_lengths:
        cum += length
        cu_seqlens_list.append(cum)
    cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)

    # 1. Benchmark Naive SIE Python Loop (calling .item() per sequence)
    # Warmup
    for _ in range(5):
        mean_embeddings = []
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            mean_embeddings.append(hidden[start:end].mean(dim=0))
        _ = torch.stack(mean_embeddings)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iterations):
        mean_embeddings = []
        for i in range(num_seqs):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            mean_embeddings.append(hidden[start:end].mean(dim=0))
        _ = torch.stack(mean_embeddings)
    if device.type == "cuda":
        torch.cuda.synchronize()
    sie_loop_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    # 2. Benchmark Turing Engine Fused Zero-Sync Segment Pool
    # Warmup
    for _ in range(5):
        _ = fused_segment_mean_pool_cuda(hidden, cu_seqlens, num_seqs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = fused_segment_mean_pool_cuda(hidden, cu_seqlens, num_seqs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    turing_fused_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    speedup = sie_loop_ms / max(turing_fused_ms, 1e-6)
    throughput_tok_s = (total_tokens / (turing_fused_ms / 1000.0))

    return {
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "hidden_dim": hidden_dim,
        "sie_python_loop_ms": round(sie_loop_ms, 3),
        "turing_fused_pool_ms": round(turing_fused_ms, 3),
        "speedup": round(speedup, 2),
        "throughput_tok_s": round(throughput_tok_s, 1),
    }


def benchmark_compute_cost_batching(num_requests: int = 128) -> Dict[str, Any]:
    """
    Compares Compute-Cost Adaptive Batching vs. Naive Fixed Padding Batching.
    """
    np.random.seed(42)
    lengths = np.random.exponential(scale=64, size=num_requests).astype(int).clip(8, 1024).tolist()

    # 1. Naive fixed-size batching with padding to max length in each batch (batch_size=16)
    batch_size = 16
    naive_total_padded_tokens = 0
    raw_tokens = sum(lengths)

    for i in range(0, num_requests, batch_size):
        chunk = lengths[i:i + batch_size]
        max_len = max(chunk)
        naive_total_padded_tokens += max_len * len(chunk)

    padding_waste_naive = (naive_total_padded_tokens - raw_tokens) / float(naive_total_padded_tokens) * 100.0

    # 2. Adaptive compute-cost batching
    batcher = AdaptiveComputeBatcher(max_batch_cost=65536, max_batch_size=batch_size, cost_tolerance_ratio=0.35)
    reqs = [{"id": f"r{i}", "task": "embedding", "seq_len": l, "hidden_dim": 1024} for i, l in enumerate(lengths)]
    batches = batcher.bucket_requests(reqs)

    adaptive_padded_tokens = 0
    for b in batches:
        b_lens = [r["seq_len"] for r in b]
        adaptive_padded_tokens += max(b_lens) * len(b)

    padding_waste_adaptive = (adaptive_padded_tokens - raw_tokens) / float(adaptive_padded_tokens) * 100.0
    waste_reduction = padding_waste_naive - padding_waste_adaptive

    return {
        "num_requests": num_requests,
        "raw_tokens": raw_tokens,
        "naive_padded_tokens": naive_total_padded_tokens,
        "naive_padding_waste_pct": round(padding_waste_naive, 2),
        "adaptive_padded_tokens": adaptive_padded_tokens,
        "adaptive_padding_waste_pct": round(padding_waste_adaptive, 2),
        "padding_waste_reduction_pct": round(waste_reduction, 2),
    }


def benchmark_full_agent_pipeline(device: torch.device) -> Dict[str, Any]:
    """
    Simulates a full 5-stage agent pipeline running on a single GPU via ElasticMultiModelManager:
    Stage 1: Document OCR / Parsing (Docling representation)
    Stage 2: Named Entity Recognition (GLiNER span extraction)
    Stage 3: Vector Embeddings (BGE dense embeddings with fused segment reduction)
    Stage 4: Cross-Encoder Reranking (BGE-Reranker top-k scoring)
    Stage 5: Generative Synthesis (Subspace LM draft generation)
    """
    mgr = ElasticMultiModelManager(vram_budget_bytes=16 * 1024**3, device=device)

    # Register the 5 pipeline models
    mgr.register_model("docling-parser", TaskType.EXTRACT, estimated_vram_bytes=800 * 1024**2)
    mgr.register_model("gliner-ner", TaskType.EXTRACT, estimated_vram_bytes=600 * 1024**2)
    mgr.register_model("bge-embed-large", TaskType.EMBEDDING, estimated_vram_bytes=1200 * 1024**2)
    mgr.register_model("bge-reranker-v2", TaskType.RERANK, estimated_vram_bytes=1200 * 1024**2)
    mgr.register_model("qwen-7b-subspace", TaskType.GENERATE, estimated_vram_bytes=3800 * 1024**2)

    doc_text = """
    PROOF OF LOSS CLAIM FORM
    Policyholder: Sarah Connor
    Policy Number: POL-998234-FL
    Date of Loss: 2026-08-15
    Description of Damage: Severe flood damage with standing water in basement and ground floor living areas.
    Repair Estimate: $45,200.00
    """

    candidates = [
        "Flood insurance policy clause 4.2: Proof of loss must be signed and filed within 60 days of damage occurrence.",
        "General exclusion for earthquake and earth movement damage unless separate rider attached.",
        "Personal property coverage covers up to $50,000 for residential content replacement.",
    ]

    labels = ["Sarah Connor", "POL-998234-FL", "$45,200.00"]

    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    # Stage 1: Document Extract / Parsing
    t0 = time.perf_counter()
    _ = mgr.extract("docling-parser", text=doc_text, labels=["PROOF OF LOSS", "Repair Estimate"])
    t_stage1 = (time.perf_counter() - t0) * 1000.0

    # Stage 2: NER Entity Extraction
    t0 = time.perf_counter()
    entities = mgr.extract("gliner-ner", text=doc_text, labels=labels)
    t_stage2 = (time.perf_counter() - t0) * 1000.0

    # Stage 3: Embedding Computation (Fused Segment Pool)
    t0 = time.perf_counter()
    embeddings = mgr.encode("bge-embed-large", texts=[doc_text] + candidates, pool_type="mean", normalize=True)
    t_stage3 = (time.perf_counter() - t0) * 1000.0

    # Stage 4: Cross-Encoder Reranking
    t0 = time.perf_counter()
    ranked = mgr.score("bge-reranker-v2", query="flood damage proof of loss 60 days", candidates=candidates)
    t_stage4 = (time.perf_counter() - t0) * 1000.0

    # Stage 5: Generation Simulation
    t0 = time.perf_counter()
    meta_gen = mgr.get_or_load_model("qwen-7b-subspace")
    t_stage5 = (time.perf_counter() - t0) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_pipeline_ms = (time.perf_counter() - t_start) * 1000.0

    allocated_gb = mgr.stats["allocated_bytes"] / (1024**3)

    return {
        "device": str(device),
        "stage1_doc_parse_ms": round(t_stage1, 3),
        "stage2_ner_extract_ms": round(t_stage2, 3),
        "stage3_embedding_ms": round(t_stage3, 3),
        "stage4_rerank_ms": round(t_stage4, 3),
        "stage5_gen_load_ms": round(t_stage5, 3),
        "total_pipeline_latency_ms": round(total_pipeline_ms, 3),
        "peak_vram_allocated_gb": round(allocated_gb, 2),
        "models_resident": mgr.stats["loaded_models"],
    }


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"================================================================================")
    print(f"   ⚡ TURING ENGINE: MULTI-MODEL PIPELINE & SEGMENT POOLING BENCHMARK")
    print(f"   Active Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'Apple Silicon / Host'})")
    print(f"================================================================================\n")

    print("[1/3] Benchmarking In-SRAM Fused Segment Pooling vs. SIE Python Loop...")
    pool_res = benchmark_segment_pooling(device=device, num_seqs=64, hidden_dim=1024, iterations=50)
    print(f"  • SIE Python Loop (.item() sync) : {pool_res['sie_python_loop_ms']:.3f} ms")
    print(f"  • Turing Fused Segment Pool      : {pool_res['turing_fused_pool_ms']:.3f} ms")
    print(f"  • Speedup Factor                 : {pool_res['speedup']:.2f}x Faster ({pool_res['throughput_tok_s']:,.0f} tok/s)")

    print("\n[2/3] Benchmarking Compute-Cost Adaptive Batching vs. Fixed Padded Batching...")
    batch_res = benchmark_compute_cost_batching(num_requests=128)
    print(f"  • Naive Fixed-Padding Waste      : {batch_res['naive_padding_waste_pct']:.2f}% of token FLOPs")
    print(f"  • Adaptive Cost-Batching Waste   : {batch_res['adaptive_padding_waste_pct']:.2f}% of token FLOPs")
    print(f"  • Absolute Waste Reduction       : -{batch_res['padding_waste_reduction_pct']:.2f}% padding compute eliminated")

    print("\n[3/3] Benchmarking Full 5-Stage Agent Pipeline on Single GPU...")
    pipe_res = benchmark_full_agent_pipeline(device=device)
    print(f"  • Stage 1 (Docling Parse)        : {pipe_res['stage1_doc_parse_ms']:.2f} ms")
    print(f"  • Stage 2 (GLiNER NER Extract)   : {pipe_res['stage2_ner_extract_ms']:.2f} ms")
    print(f"  • Stage 3 (BGE Vector Embeddings): {pipe_res['stage3_embedding_ms']:.2f} ms")
    print(f"  • Stage 4 (BGE Reranking)        : {pipe_res['stage4_rerank_ms']:.2f} ms")
    print(f"  • Stage 5 (Generative LM Swapping): {pipe_res['stage5_gen_load_ms']:.2f} ms")
    print(f"  • Total Pipeline End-to-End Latency: {pipe_res['total_pipeline_latency_ms']:.2f} ms")
    print(f"  • Total VRAM Allocated Across 5 Models: {pipe_res['peak_vram_allocated_gb']:.2f} GB (fits in single 8GB/24GB card!)")

    print(f"\n================================================================================")
    print(f"   [+] All Multi-Model Pipeline benchmarks completed successfully!")
    print(f"================================================================================\n")

    # Output JSON results
    out_dict = {
        "segment_pooling": pool_res,
        "adaptive_batching": batch_res,
        "full_agent_pipeline": pipe_res,
    }
    with open("results_sie_multi_model_benchmark.json", "w") as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    main()
