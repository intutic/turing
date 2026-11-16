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
| Token-by-token vs one pass (sanity) | -0.0000 ± 0.0001 | -0.0002 ± 0.0001 | +0.0001 ± 0.0001 |
| int8 KV (per token) | +0.0004 ± 0.0003 | -0.0001 ± 0.0005 | -0.0005 ± 0.0007 |
| int4 KV (per 32 values) | +0.0246 ± 0.0072 | +0.0184 ± 0.0049 | +0.0245 ± 0.0051 |
| Quest, 25% of pages | +0.0356 ± 0.0105 | +0.0235 ± 0.0052 | +0.0136 ± 0.0065 |
| Quest, 10% of pages | +0.1474 ± 0.0223 | +0.1071 ± 0.0158 | +0.0450 ± 0.0137 |
| int8 + Quest 25% | +0.0378 ± 0.0105 | +0.0246 ± 0.0051 | +0.0132 ± 0.0067 |

**C. Quest at 8K context** (SmolLM2-1.7B, its trained length; 8 windows of 8,192 tokens, last 128 positions scored; `results_kv_quality_8k_cuda.json`): 25% of pages +0.0014 ± 0.0039, 10% of pages +0.0172 ± 0.0062. Page selection costs less as the context grows (+0.013 at 2K for the same model), as the Quest paper reports.

The int8 cache costs nothing measurable (every Δ is within two standard errors of zero), which is why `--kv-cache-dtype auto` uses it on CUDA. int4 with per-32-value scales costs about +0.02 nats; Quest page selection at 25% costs +0.013 to +0.035 at 2K, less on the larger model and at longer context; these reproduce the published results (KIVI-style grouped 4-bit, Quest) at this scale.

### Reproduce
```bash
python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-135M,HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B \
  --arms hf_native,exact_forward,exact_decode,int8,int4,quest25,quest10,int8_quest25 --context 2048 --decode 128 --windows 16
```

---

## 3. KV Cache: Memory and Decode Throughput

**Memory.** K and V bytes per token, all layers, for a 70B-shaped model (80 layers, 8 KV heads, head_dim 128) at a 32K context: fp16 10.0 GiB; int8 5.16 GiB (−48%: head_dim + 4 bytes per head); int4 3.13 GiB (−69%). The cache is paged in 16-token pages from one pool, so a request leaves at most 15 token slots of a page unused.

**Decode throughput** on one NVIDIA L4 (fp16), measured by `scripts/benchmark_kv_decode.py`: one WikiText-2 prompt is prefilled (in 2,048-token chunks, as the engine does), its cache is copied into B independent requests, and 32 batched decode steps are timed through the serving path (`SubspaceCausalLM.forward` with one cache per request, as the engine runs it). The paged KV pool takes 85% of the free GPU memory; each setting sizes its own. The baseline is the same weights in HuggingFace transformers with its own cache, same GPU, batch and context.

**SmolLM2-360M** (32 layers, 15/5 heads, head_dim 64; `results_kv_decode_cuda_smollm2_360m.json`). Generated tokens per second; 'pool full' = the KV pool (85% of free GPU memory) cannot hold the batch, OOM = CUDA out of memory, — = not run after a smaller batch did not fit; contexts above 8,192 exceed the model's trained length (speed only).

| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 2K × B=64 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 8K × B=64 | 32K × B=1 | 32K × B=8 | 32K × B=32 | 32K × B=64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 29 | 228 | 783 | 827 | 30 | 202 | 213 | OOM | 30 | 56 | OOM | — |
| Turing, native, no CUDA graph | 24 | 190 | 755 | 1501 | 24 | 191 | 463 | pool full | OOM | — | — | — |
| Turing, native cache | 162 | 888 | 1506 | 1715 | 131 | 382 | 468 | pool full | OOM | — | — | — |
| Turing, int8 | 163 | 882 | 1518 | 1713 | 129 | 383 | 465 | 496 | OOM | — | — | — |
| Turing, int4 | 163 | 825 | 1431 | 1597 | 127 | 353 | 440 | 458 | OOM | — | — | — |
| Turing, int8 + Quest 25% | 24 | 188 | 743 | 1453 | 16 | 121 | 478 | 594 | OOM | — | — | — |

**SmolLM2-1.7B** (24 layers, 32/32 heads, head_dim 64; `results_kv_decode_cuda_smollm2_1p7b.json`). Generated tokens per second; 'pool full' = the KV pool (85% of free GPU memory) cannot hold the batch, OOM = CUDA out of memory, — = not run after a smaller batch did not fit; contexts above 8,192 exceed the model's trained length (speed only).

| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 2K × B=64 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 8K × B=64 | 32K × B=1 | 32K × B=8 | 32K × B=32 | 32K × B=64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 40 | 137 | 159 | OOM | 29 | 43 | OOM | — | 10 | OOM | — | — |
| Turing, native, no CUDA graph | 31 | 228 | 364 | pool full | 32 | 89 | pool full | — | 19 | pool full | — | — |
| Turing, native cache | 56 | 234 | 366 | pool full | 41 | 90 | pool full | — | 19 | pool full | — | — |
| Turing, int8 | 56 | 230 | 361 | 394 | 41 | 88 | pool full | — | 19 | pool full | — | — |
| Turing, int4 | 55 | 215 | 329 | 356 | 39 | 81 | 97 | pool full | 17 | 23 | pool full | — |
| Turing, int8 + Quest 25% | 31 | 217 | 346 | 382 | 19 | 109 | pool full | — | 20 | pool full | — | — |

What the tables show:
* **CUDA graphs** remove the per-layer launch overhead: SmolLM2-360M at batch 1 and 2K decodes 162 tok/s against 24 eagerly and 29 for HuggingFace transformers; SmolLM2-1.7B 56 against 40. At batch 64 and 2K SmolLM2-360M reaches 1,715 tok/s against 827 (2.1×).
* **Capacity:** the pool wastes at most one partial page per request, and int8/int4 hold 1.87×/3.17× the tokens of the native cache in the same memory. int8 fits SmolLM2-360M at 8K × 64 and SmolLM2-1.7B at 2K × 64, int4 SmolLM2-1.7B at 32K × 8; HuggingFace runs out of memory on all three.
* **Long context, one request:** SmolLM2-1.7B at 32K decodes 19 tok/s against 10.
* **Quest page selection** runs outside the CUDA graph (the page lists change every step) and pays only where reading the cache dominates:

Quest page selection at 25% against reading every page, both int8, SmolLM2-360M (`results_quest_threshold_cuda_smollm2_360m.json`; selection forced on at every length). Tokens per second; ratio > 1 means Quest is faster.

| Batch | 1K | 2K | 4K | 8K | 16K | 32K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.09× | 0.09× | 0.10× | 0.13× | 0.15× | 0.21× |
| 8 | 0.12× | 0.14× | 0.20× | 0.31× | 0.55× | 1.03× |
| 64 | 0.31× | 0.54× | 0.99× | 1.19× | OOM | OOM |


  So `kv_select_min_tokens` defaults to 8,192, and Quest stays opt-in (`--kv-page-budget`). At batch 1 it is 5–10× slower than reading every page.
* **Not measured:** other engines (vLLM, SGLang, TensorRT-LLM, llama.cpp); this repository does not run them.

### Reproduce
```bash
python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M \
  --settings hf_transformers,native_eager,native,int8,int4,int8_quest25 --contexts 2048,8192,32768 --batches 1,8,32,64
# Quest against reading every page, selection forced on at every length:
python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --settings int8,int8_quest25 \
  --contexts 1024,2048,4096,8192,16384,32768 --batches 1,8,64 --quest-min-tokens 0
```

> **About the earlier NIAH section.** `scripts/stress_test_niah_breaking_point.py` quantizes one 512-token page of random vectors per row; the "context length" only labelled the row, and no model was involved. Its 1,000,000-token and rank-table claims were removed; the script now says what it tests.

---

## 4. Storage Ingestion Ladder (measured cold, Apple M4 Pro SSD)

`scripts/benchmark_ingest_ladder_all_tiers.py` reads a 2048 MiB file laid out as tensor-sized segments (one 7B-layer pattern repeated, 42 segments) into host memory. Before every cold run the file is written fresh through an `F_NOCACHE` descriptor (macOS) or dropped with `posix_fadvise(POSIX_FADV_DONTNEED)` (Linux), and `mincore` confirms that none of it is in the file cache; the warm row reads it again once it is fully cached. GB/s is bytes / wall time with 1 GB = 10⁹ bytes; ± is the standard error over n runs. The machine was shared with other jobs (load average about 10), so the spread is wide. Results: `results_ingest_ladder.json`.

| Row | Cache | GB/s | ms | n |
| :--- | :---: | ---: | ---: | :---: |
| Tier 0: mmap demand paging | cold | 0.75 ± 0.01 | 2,874 | 3 |
| Tier 1: mmap + MADV_WILLNEED | cold | 2.25 ± 0.23 | 977 | 3 |
| Tier 2: ring reader, read_segments_parallel (bytes) | cold | 3.28 ± 0.71 | 745 | 3 |
| Tier 2: ring reader, read_segments_into (buffer) | cold | 4.44 ± 0.43 | 493 | 3 |
| Tier 2: ring reader, read_segments_into, queue_depth=1 | cold | 4.37 ± 0.56 | 510 | 3 |
| Tier 2: ring reader, read_segments_into, queue_depth=8 | cold | 5.26 ± 0.41 | 413 | 3 |
| Tier 5: ring reader, file in page cache (warm) | warm | 51.29 ± 1.36 | 42 | 3 |
| Tier 3: .tgate4 compressed container | — | not measured | — | — |
| Tier 4: GPUDirect Storage + layer pipelining | — | not measured | — | — |

The earlier version of this table was not a cold measurement: it divided the whole file size by the time to read 10 tensors, derived Tier 3 as Tier 2 time × 0.25, hardcoded Tier 4 at 45.2 ms, and floored Tier 5 at 0.042 ms. Tier 3 has no loader that reads a `.tgate4` file end to end, and Tier 4 needs CUDA with cuFile, so neither is measured here. On this sequential layout queue depth 1 already reaches about 4.4 GB/s (kernel readahead helps sequential 1 MiB reads), which suggests most of the gain over the old reader below comes from capping requests at 1 MiB rather than from depth; random reads depend on depth far more (cift F1 on the same SSD: 1 MiB random reads went from 2.4 GB/s with 1 in flight to 6.0 GB/s with 64).

### Ring reader before and after (cold, interleaved A/B)

`NativeAsyncRingReader` used to give each of 8 threads a run of whole tensors to `pread`, and ignored `queue_depth`. It now splits reads into requests of at most 1 MiB and keeps up to `queue_depth` (default 64) in flight, which is what this SSD needs to reach its bandwidth. `scripts/benchmark_ring_reader_ab.py` alternates the two builds in child processes on the same cold 2048 MiB file, calling `read_segments_parallel`. Results: `results_ring_reader_ab_mps.json`.

| Build | GB/s (mean ± se) | GB/s (median) | ms | n |
| :--- | ---: | ---: | ---: | :---: |
| f936008 | 2.22 ± 0.35 | 2.33 | 1,225 | 8 |
| current | 5.78 ± 0.41 | 6.23 | 393 | 8 |

### Reproduce on your hardware
```bash
python scripts/benchmark_ingest_ladder_all_tiers.py --size-mib 2048 --repeats 3
git archive f936008 | tar -x -C /tmp/base && (cd /tmp/base && python setup.py build_ext --inplace)
python scripts/benchmark_ring_reader_ab.py --baseline /tmp/base --baseline-label f936008 --pairs 8
```

---

## 5. Removed Results

The [claims audit](claims-audit.md) has the full table (claim, producing script and line, classification). In summary, the removed sections fell into four groups:

* **Constants or formulas printed as results.** Speculation-gating throughput (`50 × c` plain, `50 × (1.82 − 0.3(c−1)) × c` speculative, so "1.82× at c=1" was an input); lineage residual norms (≈30.7 is the norm of the injected noise, and the naive collapse was a hard-coded `× 1.5` per turn); the 5,500 ms cold-start baseline and its 21.6× speedup; ingest Tier 3 (`Tier 2 × 0.25`), Tier 4 (45.2 ms) and the Tier 5 floor (0.042 ms) (§4 now has a cold measurement); the 7.85× cross-model transfer speedup (`baseline = measured × 7.85`); "75.0% VRAM savings" (a literal in the print string); the 7.42 GB multi-model footprint (a sum of declared estimates); "100% parity / parse success / OK" strings that were never checked. These scripts now print only what they compute.
* **Random tensors or stubs presented as model or system results.** The 131K-token NIAH scored one 512-row page of random keys; the "5-model pipeline" (Docling, GLiNER, BGE, reranker, LLM) loads no model (substring match, random embeddings, word overlap, no generation); serving, gateway and prefill/decode throughput used a random-initialised 4-layer toy model; MoE hit-rate and eviction counts come from a seeded synthetic Zipf trace; associative recall used toy models trained from scratch.
* **No committed result, or contradicted by the committed one.** Most values in the old §4–§9, §11–§18 and §20–§26 appear in no results file, and the committed runs of the same scripts disagree, sometimes in direction (for example semantic-anchor restore on MPS was slower than the baseline, the fused speculative verify ran at 0.28× on the L4, and the batched option select was 0.15×).
* **Comparisons that were not like for like.** A dense baseline without SDPA against a one-query hybrid layer; Python loops against vectorised kernels; cold page cache against warm; baselines whose first timed iteration paid kernel start-up.

**Not measured by this repository:** other engines (vLLM, SGLang, TensorRT-LLM, llama.cpp), MoE checkpoints of any size, and any model larger than SmolLM2-1.7B. Earlier versions also showed a serving-concurrency table against those engines; it was computed from formulas in `scripts/run_all_benchmarks.py` (`18.5 × c^0.72 × 2.35` for Turing Engine) and was removed in v1.6.0.
