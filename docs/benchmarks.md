# 📊 Benchmarks

Sections 2 and 3 are measured on real pretrained weights (SmolLM2 135M–1.7B) on an NVIDIA L4 and an Apple M4 Pro, by scripts in this repository whose raw results are committed (`results_*.json`). Section 1 is a single-layer timing on random weights. Nothing else is claimed here.

Earlier versions of this page carried about 25 further sections (latent decode, hybrid prefill, LoRA hot-swap, gateway latency, prefill/decode duality, MoE pinning, ingest tiers, multi-model pipeline, kernel fusions and more). An audit traced every number to the code that was supposed to produce it and found that none had been measured on a real model, many had no committed result, and several were constants or formulas printed as results. Those sections were removed; [§4](#4-removed-results) summarises why, and the [claims audit](claims-audit.md) lists every number with its producing script and line.

---

## 1. FFN Channel Pruning: Layer Time on Random Weights

One FFN layer (gate, up, down; batch 1, fp16) with random weights, timed dense and with the FFN width cut to the pruned size, on one NVIDIA L4 (`scripts/benchmark_comprehensive_real_gpu.py`, part 3; recorded in `results_gcp_master_benchmark_v4.json`). The kernels are plain cuBLAS `F.linear` on smaller matrices, so the speedup only reflects the share of channels removed.

| Layer shape (hidden × FFN → kept FFN) | Dense | Pruned | Ratio | Channels removed |
| :--- | :---: | :---: | :---: | :---: |
| 4,096 × 14,336 → 6,144 (Llama-3-8B shape) | 1.36 ms | 0.61 ms | 2.24× | 57.1% |
| 8,192 × 28,672 → 12,288 (Llama-3.1-70B shape) | 5.41 ms | 2.33 ms | 2.32× | 57.1% |
| 8,192 × 29,696 → 12,288 (Qwen-2.5-72B shape) | 5.60 ms | 2.33 ms | 2.41× | 58.6% |

> **What pruning costs.** Pruning without retraining is not lossless: at 50% it raises next-token NLL by +6.8 to +15.8 nats on SmolLM2 (§2), so it is off by default (`--sparsity 0.0`). The table times one layer, not a model; no 8B or 70B checkpoint was run.

---

## 2. Quality on Real Weights

Every number below is a paired next-token NLL difference on real pretrained weights, measured by `scripts/eval_kv_quality.py`: the same text windows are scored by the exact model and by each setting, and the table gives the mean difference per window ± its standard error. The exact model reproduces HuggingFace's logits (`tests/test_hf_parity.py`). Earlier versions of this page reported GSM8K/HumanEval/MMLU retention and throughput against vLLM that no script in the repository produced; they were removed.

**A. Serving defaults** (Apple M4 Pro, fp32, WikiText-2 test, 8 windows of 2,048 tokens, last 256 positions scored; `results_serving_defaults_mps.json`). Δ is the mean paired NLL change ± its standard error.

| Model | Exact model (= HuggingFace) | FFN pruning 50% (old default) | Recirculation (old default) |
| :--- | :---: | :---: | :---: |
| SmolLM2-135M | NLL 2.606 (Δ vs HF -0.00000) | **+15.78 ± 1.73** | +0.00008 ± 0.00005 |
| SmolLM2-360M | NLL 2.309 (Δ vs HF +0.00000) | **+8.92 ± 0.43** | +0.00085 ± 0.00058 |
| SmolLM2-1.7B | NLL 1.921 (Δ vs HF -0.00000) | **+6.85 ± 0.21** | +0.00060 ± 0.00096 |

**B. KV cache settings** (NVIDIA L4, fp16, WikiText-2 test, 16 windows of 2,048 tokens, last 128 positions scored; `results_kv_quality_cuda.json`). Low-bit arms are paired against the exact model; Quest arms run token by token and are paired against exact token-by-token decode.

| Setting | SmolLM2-135M | SmolLM2-360M | SmolLM2-1.7B |
| :--- | :---: | :---: | :---: |
| HuggingFace vs Turing exact (sanity) | -0.0003 ± 0.0001 | -0.0000 ± 0.0002 | +0.0002 ± 0.0002 |
| Token-by-token vs one pass (sanity) | -0.0001 ± 0.0001 | -0.0001 ± 0.0001 | +0.0001 ± 0.0001 |
| int8 KV (per token) | +0.0004 ± 0.0003 | -0.0001 ± 0.0005 | -0.0005 ± 0.0007 |
| int4 KV (per 32 values) | +0.0246 ± 0.0072 | +0.0184 ± 0.0049 | +0.0245 ± 0.0051 |
| Quest, 25% of pages | +0.0354 ± 0.0105 | +0.0235 ± 0.0050 | +0.0132 ± 0.0065 |
| Quest, 10% of pages | +0.1472 ± 0.0224 | +0.1062 ± 0.0156 | +0.0450 ± 0.0139 |
| int8 + Quest 25% | +0.0385 ± 0.0103 | +0.0246 ± 0.0053 | +0.0133 ± 0.0068 |

The int8 cache costs nothing measurable (every Δ is within two standard errors of zero), which is why `--kv-cache-dtype auto` uses it on CUDA. int4 with per-32-value scales costs about +0.02 nats; Quest page selection at 25% costs +0.013 to +0.035, less on the larger model; these reproduce the published results (KIVI-style grouped 4-bit, Quest) at this scale.

### Reproduce
```bash
python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-135M,HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B \
  --arms hf_native,exact_forward,exact_decode,int8,int4,quest25,quest10,int8_quest25 --context 2048 --decode 128 --windows 16
```

---

## 3. KV Cache: Memory and Decode Throughput

**Memory.** K and V bytes per token, all layers, for a 70B-shaped model (80 layers, 8 KV heads, head_dim 128) at a 32K context: fp16 10.0 GiB; int8 5.16 GiB (−48%: head_dim + 4 bytes per head); int4 3.13 GiB (−69%). The cache grows by a quarter when full, so up to 25% of the latest growth can be unused capacity.

**Decode throughput** on one NVIDIA L4 (fp16), measured by `scripts/benchmark_kv_decode.py`: one WikiText-2 prompt is prefilled, its cache cloned B times, and 32 batched decode steps are timed through the serving path (`SubspaceCausalLM.forward` with one cache per request, as the engine runs it). The baseline is the same weights in HuggingFace transformers with its own cache, same GPU, batch and context.

**SmolLM2-360M** (32 layers, 15/5 heads, head_dim 64; `results_kv_decode_cuda_smollm2_360m.json`). Generated tokens per second; OOM = out of memory; contexts above 8,192 exceed the model's trained length (speed only).

| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 2K × B=64 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 8K × B=64 | 32K × B=1 | 32K × B=8 | 32K × B=32 | 32K × B=64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 30 | 231 | 792 | 833 | 30 | 204 | 215 | OOM | 30 | 56 | OOM | OOM |
| Turing, native cache | 25 | 192 | 705 | 1274 | 25 | 191 | 360 | OOM | 25 | 89 | OOM | OOM |
| Turing, int8 | 25 | 187 | 652 | 1155 | 25 | 188 | 324 | 349 | 25 | 79 | OOM | OOM |
| Turing, int4 | 26 | 195 | 690 | 1245 | 25 | 192 | 448 | 472 | 25 | 106 | OOM | OOM |
| Turing, int8 + Quest 25% | 9 | 19 | 20 | 20 | 9 | 19 | 20 | 20 | 9 | 19 | OOM | OOM |

**SmolLM2-1.7B** (24 layers, 32/32 heads, head_dim 64; `results_kv_decode_cuda_smollm2_1p7b.json`). Generated tokens per second; OOM = out of memory; contexts above 8,192 exceed the model's trained length (speed only).

| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 2K × B=64 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 8K × B=64 | 32K × B=1 | 32K × B=8 | 32K × B=32 | 32K × B=64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 41 | 137 | 159 | OOM | 29 | 43 | OOM | OOM | 10 | OOM | OOM | OOM |
| Turing, native cache | 33 | 193 | OOM | OOM | 33 | OOM | OOM | OOM | 16 | OOM | OOM | OOM |
| Turing, int8 | 33 | 181 | 266 | OOM | 33 | 64 | OOM | OOM | 14 | OOM | OOM | OOM |
| Turing, int4 | 33 | 215 | 345 | 377 | 32 | 84 | OOM | OOM | 18 | OOM | OOM | OOM |
| Turing, int8 + Quest 25% | 11 | 25 | 27 | OOM | 11 | 25 | OOM | OOM | 16 | OOM | OOM | OOM |


What the tables show:
* **Batched decode:** one fused append + attention launch per layer serves every request. At batch 64 and 2K context SmolLM2-360M decodes 1,274 tok/s against 833 for HuggingFace transformers (1.53×), and at 8K and batch 32, 360 against 215.
* **Capacity:** the int8 and int4 caches fit batches the native cache and HuggingFace do not (SmolLM2-360M at 8K × 64; SmolLM2-1.7B int4 at 2K × 64).
* **Batch 1 at short context** is limited by per-layer Python and kernel-launch overhead: SmolLM2-1.7B at 2K decodes 33 tok/s against 41 for HuggingFace.
* **Quest page selection** runs on the per-request path today (about 20 tok/s regardless of batch), so it only pays where memory, not speed, is the limit.
* **Not measured:** other engines (vLLM, SGLang, TensorRT-LLM, llama.cpp); this repository does not run them.

### Reproduce
```bash
python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M \
  --settings hf_transformers,native,int8,int4,int8_quest25 --contexts 2048,8192,32768 --batches 1,8,32,64
```

> **About the earlier NIAH section.** `scripts/stress_test_niah_breaking_point.py` quantizes one 512-token page of random vectors per row; the "context length" only labelled the row, and no model was involved. Its 1,000,000-token and rank-table claims were removed; the script now says what it tests.

---

## 4. Removed Results

The [claims audit](claims-audit.md) has the full table (claim, producing script and line, classification). In summary, the removed sections fell into four groups:

* **Constants or formulas printed as results.** Speculation-gating throughput (`50 × c` plain, `50 × (1.82 − 0.3(c−1)) × c` speculative, so "1.82× at c=1" was an input); lineage residual norms (≈30.7 is the norm of the injected noise, and the naive collapse was a hard-coded `× 1.5` per turn); the 5,500 ms cold-start baseline and its 21.6× speedup; ingest Tier 3 (`Tier 2 × 0.25`), Tier 4 (45.2 ms) and the Tier 5 floor (0.042 ms); the 7.85× cross-model transfer speedup (`baseline = measured × 7.85`); "75.0% VRAM savings" (a literal in the print string); the 7.42 GB multi-model footprint (a sum of declared estimates); "100% parity / parse success / OK" strings that were never checked. These scripts now print only what they compute.
* **Random tensors or stubs presented as model or system results.** The 131K-token NIAH scored one 512-row page of random keys; the "5-model pipeline" (Docling, GLiNER, BGE, reranker, LLM) loads no model (substring match, random embeddings, word overlap, no generation); serving, gateway and prefill/decode throughput used a random-initialised 4-layer toy model; MoE hit-rate and eviction counts come from a seeded synthetic Zipf trace; associative recall used toy models trained from scratch.
* **No committed result, or contradicted by the committed one.** Most values in the old §4–§9, §11–§18 and §20–§26 appear in no results file, and the committed runs of the same scripts disagree, sometimes in direction (for example semantic-anchor restore on MPS was slower than the baseline, the fused speculative verify ran at 0.28× on the L4, and the batched option select was 0.15×).
* **Comparisons that were not like for like.** A dense baseline without SDPA against a one-query hybrid layer; Python loops against vectorised kernels; cold page cache against warm; baselines whose first timed iteration paid kernel start-up.

**Not measured by this repository:** other engines (vLLM, SGLang, TensorRT-LLM, llama.cpp), MoE checkpoints of any size, and any model larger than SmolLM2-1.7B. Earlier versions also showed a serving-concurrency table against those engines; it was computed from formulas in `scripts/run_all_benchmarks.py` (`18.5 × c^0.72 × 2.35` for Turing Engine) and was removed in v1.6.0.
