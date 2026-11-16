# 🔍 Claims Audit

Every benchmark number in `docs/benchmarks.md` (except §2, §3 and §10, which were rebuilt from measurements in v1.6.0), `README.md`, `docs/index.md`, `docs/comparison.md` and `docs/models_and_hardware.md` was traced to the code that was supposed to produce it, as of commit `f936008`. Doc and script line numbers, and section numbers such as §19, refer to that commit; the storage ingestion ladder (old §19) has since been measured cold and is §4 of the current benchmarks page.

**Classes**

| | Meaning |
| :--- | :--- |
| **A** | Measured on a real pretrained model. |
| **B** | Measured, but on random tensors, a synthetic trace, a toy or random-initialised model, or a placeholder. |
| **C** | A constant or formula printed or written as a result. |
| **D** | No producing script, or the producing script's committed runs contradict the number and no committed result contains it. |
| **E** | A capability or competitor statement rather than a number. |

**Evidence used.** The scripts themselves; committed results (`results_*.json`, plus `exhaustive_benchmark_results_cuda.json` and `exhaustive_benchmark_results_mps.json`, which hold the recorded stdout of every documented script on an L4 and a Mac). No GPU was used for this audit; short Mac runs are noted where made.

**Outcome.** Of roughly 300 numbers outside §2, §3 and §10, none was measured on a real model with the Turing engine. The only real-model numbers were GPT-2 and SmolLM2 tok/s from HuggingFace `generate`. The page now keeps §2 and §3 (measured on real weights) and three random-weight FFN layer timings that the committed L4 run backs; everything else was removed or relabelled, and every script line that printed a constant or formula was changed to print what it computes.

---

## 1. Constants and formulas printed as results (class C)

| Claim | Where it appeared | Producing script:line | What the code did | Action |
| :--- | :--- | :--- | :--- | :--- |
| Spec gating 91 / 152 / 183 / 184 / 200 / 400 / 800 tok/s; "1.82× at c=1" | benchmarks §12C, §18; README; index | `scripts/benchmark_traffic_and_spec.py:97, 101-102` | `plain_tps = 50.0 * min(c, 16)`; `spec_tps = 50.0 * max(0.5, 1.82 - 0.3*(c-1)) * c`; no model runs | Removed; script now prints the gate decision only; `results_traffic_spec.json` regenerated |
| Byte-exact spec/plain parity "100%, 0 divergence" | §12C, §18; comparison | `benchmark_traffic_and_spec.py:148-151`; `run_gcp_master_validation_and_benchmarks_v8.py:350-357` | Compares two identical literal token lists; no decoding | Removed; scripts label it a verifier self-check |
| Lineage residual ‖ΔC_R‖ ≈ 30.70–30.73 (clean_base) | §12A, §17; README; index; comparison | `scripts/benchmark_lineage_strategies.py:80, 87` | `base + noise*1.5` with `noise = randn*0.01`: the norm is 0.015·√(1024·4096) = 30.72 by construction; `turing/core/lineage.py` imported, never called | Removed; script now drives the library strategies with a seeded synthetic translator and says so |
| Naive drift 1,024 → 21,285 | §12A, §17 | `benchmark_lineage_strategies.py:84` | `current * 1.5 + noise`: 2048·(1.5^t − 1) | Removed |
| append_only residual 0.00 | §12A, §17 | `benchmark_lineage_strategies.py:90-93` | First `seq_len` rows of the concatenation are the base itself | Removed |
| Lineage memory 105 MB (non-CUDA) | §12A | `benchmark_lineage_strategies.py:103` | `simulated_mem = seq_len*hidden*2*turn` | Script now reports the bytes of the cache it holds |
| 70B cold start 5,500 ms baseline, 21.63× | §11 | `scripts/benchmark_lora_and_speculation.py:178-179` | Literal `5,500.00 ms`; speedup `5500.0 / ttr` | Removed baseline and speedup |
| LoRA "0.00 ms bubble", "~600 MB pool", "6.2 MB/slot", "Async DMA" | §11 | `benchmark_lora_and_speculation.py:79-84` | Literal strings / `capacity * 6.2` (measured pool is 13.1 MB, slots 4.2 MB) | Script computes sizes from tensors; labels fixed |
| Speculative acceptance 100.0% | §11 | `benchmark_lora_and_speculation.py:133-137` | Target logits forced to match the draft (`= 50.0`); tree width 1 | Removed; script labels it a self-check |
| Cross-model W* transfer 54.96 vs 445.42 ms, 7.85× | comparison §7 | `scripts/benchmark_xkv_deliberation.py:102-104` | `text_baseline_ms = measured * 7.85` | Removed baseline and speedup |
| Ingest Tier 3 1,172.98 ms, 7.61× | §19; README; comparison | `scripts/benchmark_ingest_ladder_all_tiers.py:133-134` | `elapsed_t2_ms * 0.25` | Removed from docs; script owned by the offload-fixes session (see §7) |
| Ingest Tier 4 45.20 ms, 17–81 GB/s, 41.86× | §19; comparison | `benchmark_ingest_ladder_all_tiers.py:155` | `45.2 if torch.cuda.is_available() else elapsed_t2_ms * 0.15` | As above |
| Ingest Tier 5 0.042 ms, 87,487 GB/s | §19 | `benchmark_ingest_ladder_all_tiers.py:168` | `max(measured, 0.042)` around a `data_ptr()` loop | As above |
| "75.0% VRAM savings" (GPT-2, SmolLM2) | §22E | `scripts/benchmark_comprehensive_real_gpu.py:123` at `e3b18be^` | Literal in the print string (fixed in v1.6.0; it now computes 73.4% vs fp32 on random K) | Removed |
| NVFP4 micro-scaling "−75.0%" | §23D | `scripts/benchmark_nextgen_speedups.py:227` at `e3b18be^` | `compression_ratio = 75.0`; HEAD computes 46.9% | Removed; `results_nextgen_speedups_cuda.json` still holds the old 75.0 |
| "SVD INT8 KV −75%" | `benchmark_prefill_decode_duality.py` output | `scripts/benchmark_prefill_decode_duality.py:243-264` | Reduction equals the `svd_compression_ratio=0.75` it passes in | Benchmark removed from the script |
| Latent decode "99.6% memory traffic cut" | §7 | `scripts/benchmark_latent_decode.py:78-80` | Byte formula from shapes, constant across L | Script now sums the bytes of the tensors it times |
| Multi-model "7.42 GB peak VRAM" | §24; README; index; comparison; models_and_hardware | `scripts/benchmark_sie_pipeline_speedups.py:151-155, 207`; `run_gcp_master_validation_and_benchmarks_v8.py:298-299` | Sum of `estimated_vram_bytes` passed to `register_model` | Removed; scripts print "sum of declared estimates (not measured)" |
| Per-model ~1.1 / 1.3 / 0.6 / 1.2 / 3.2 GB | models_and_hardware §1B | none | Not even equal to the script's estimates | Removed (D) |
| "100% OK / Parse Success / Extraction Accuracy" | §14 | `scripts/benchmark_triple_gateway_and_structured.py:56, 63, 70, 129, 196` | Literal strings; status codes and parse results never checked | Script counts HTTP 200s and parse results |
| Gateway E2E tok/s | gateway script | `benchmark_triple_gateway_and_structured.py:85, 157, 214` | `32 * iters / time` assumed 32 tokens per request | Script sums `eval_count` / `completion_tokens` |
| "Numerical Tensor Parity: 100% Exact Match" | readahead output | `scripts/benchmark_cold_readahead_gcp.py:75` | Literal string | Script prints the count compared |
| C++ fallback timings | v5 suite output, README "2.47M tok/s", "1.18M req/s" | `scripts/benchmark_fused_kernels_and_simd_v5.py:329, 380, 414-415, 449, 488` | When the extension is missing: `baseline / 8.0`, `/ 6.0`, `/ 12.0`, `/ 7.5`, `/ 5.0` | Script now prints "not measured" (the committed v5 JSON was a real run, not the fallback) |
| Recall verdict "Softmax & Turing Hybrid maintain exact recall" | §25 | `scripts/benchmark_associative_recall_and_decay.py:307` | Hard-coded string (hybrid scores 25%) | Replaced with the sample size |
| "x.xx× FASTER (Eliminated GPU-CPU sync barrier)" | §20 | `scripts/benchmark_fused_kernels_and_simd.py:187-190` | Printed even at 0.28× | Neutral ratio wording |
| "Lineage invariant: Enforced" | v8 suite | `run_gcp_master_validation_and_benchmarks_v8.py:401` | Unconditional string | Script checks the strategy's return value |
| Prefix-cache bypass 94.1% | §23C; README | `benchmark_nextgen_speedups.py:186` | 2,048 / 2,176, fixed by the chosen inputs | Removed |
| Pruned share 57.1% / 58.6% / 55.0% / 50.0% | §1 | `turing/config.py:43` | `1 − d_sub/d_ffn` from the chosen config (arithmetic, harmless) | Kept as "channels removed" |
| KV 32K "int8 −48% (5.2 GB)" vs "FP8 (5.0 GB)" | README; comparison | arithmetic | Correct only for an 80-layer, 8-KV-head, head_dim-128 shape, in GiB; FP8 is smaller but was marked ⚠️ | Relabelled with the shape; FP8 marked ✅ |

## 2. Numbers with no producing script, or contradicted by the committed run (class D)

| Claim | Where | What exists | Action |
| :--- | :--- | :--- | :--- |
| §1 Gemma-4-31B (2.73 / 1.37 ms) and DeepSeek-V4-Flash (2.14 / 0.96 ms) rows | §1 | No config with those dims; the DeepSeek row matches the committed DeepSeek-V3 row relabelled | Removed |
| §4 Matryoshka 7.73× / 7.14× / 7.66× and all times | §4; README "7.8x" | `benchmark_freetoken_matryoshka.py` / `benchmark_native_fusions.py` (random weights); committed CUDA run: W=4096 37.8 ms (JIT in the timing), MPS 4.97 / 2.58 / 1.33 / 0.70 ms | Removed |
| §5 fused SVD INT8 27.28× / 2.38× / 2.48× | §5 | `benchmark_native_fusions.py:79-136` (randn); baseline's first loop pays launch start-up; no committed result | Removed |
| §6 semantic-anchor restore 96.5% / 89.9% / 91.9% | §6; README "96.5%" | `benchmark_freetoken_matryoshka.py:95-153`; "prefill" is 16 random matmuls; committed MPS run shows restore **slower** (13.74 → 14.89 ms) | Removed |
| §7 latent decode 1.67× … 35.37× | §7; index | `benchmark_latent_decode.py` (random int8, fp32 MHA baseline without GQA); no committed result matches | Removed |
| §8 hybrid prefill 0.56× / 2.33× / "∞ (prevents OOM)" | §8 | Hybrid layer attends with one query; dense baseline materialises L×L scores without SDPA; committed CUDA run 1.37× / 5.30× / 378 / 761 ms | Removed |
| §9 wire codec 2.10×, entropy 3.15×, hex BMU 5.44×, router 4.50×, budget "<0.05 µs" | §9 | `benchmark_native_fusions.py:344, 404, 430, 607` print the fused time only; no baseline is timed; nothing times the budget controller | Removed |
| §11 baselines "5+ s", "15–50 ms", "200–500 req/s", "12–25 ms", "55–70%" | §11 | None | Removed |
| §12C admission 41.61 µs | §12C, §18 | Script measures 1.70 µs (L4), 4.44 µs (Mac); 0.48 µs in this audit's Mac run | Removed |
| §12D serving load test 143.68 → 162.65 tok/s | §12D | Random-init toy model; committed L4 run 89.9 → 128.1, Mac run protected is **slower** (41.97 → 34.14); the spec gate only flips a flag | Removed |
| §13 gated zero-identity 2.4×, chunk filter 3.1×, Quadtree "63× .item() stalls" | §13 | Imported, never benchmarked | Removed |
| §14 gateway Mac column (identical to L4 for /api/tags, /show, /ps; 68.98 tok/s) | §14 | Committed Mac run: 1,966 / 2,368 / 2,976 µs, 4.60 tok/s | Removed |
| §15 prefill/decode Mac columns and jitter | §15 | `results_prefill_decode_duality.json` (MPS): 181.8 ms at 128 tokens, 0.72–31.8 tok/s decode | Removed |
| §16 "−60% DRAM traffic", option gather "sub-millisecond scoring" | §16 | Never measured; committed runs show the batched gather at 0.15× (L4 host) and 0.06× (Mac) | Removed |
| §19 Tier 0/1/2 times and 4.83× / 8.50× | §19; comparison "3.50 GB/s", "4.83×" | Both tiers use the same madvising reader; committed L4 runs 372.9 / 83.7 / 201.2 ms; 4.83× is cold vs warm page cache | Removed |
| §20 spec verify "No Device Stalls" | §20 | Fused verify measured 0.28× on the L4 | Removed |
| §22E vLLM 17.60 tok/s | §22E | Needs an external server; every recorded run printed "offline" | Removed |
| §24 padding tokens 26,976 / 8,696 | §24 | The deterministic script gives 23,952 / 7,735 | Removed |
| §26 "120 ns" routing baseline, "4.5×", "4.2× less DRAM", "2.1× faster draft", "zero GC allocations", "7 DRAM passes" | §26; index | No producing code; Matryoshka measured 0.98×, varlen packing 0.21× on the L4 | Removed |
| "84% associative recall collapse" | README; index | Nothing computes 84% | Removed |
| MoE 320B on 24 GB: 18–50 tok/s (and 1–5 tok/s for other engines) | README; comparison; models_and_hardware | No MoE checkpoint is decoded anywhere; HF loader has no MoE path; committed PILOT prefetch shows 1.0× | Removed |
| "Turing Engine VRAM" column (70B 15.7 GB; GLM-5.3 3.5 GB + 42 GB RAM; …) | models_and_hardware §1A | Docs-only commits; implies 0.7–1.8 bits/parameter; the repo's own formula gives 21.8 GiB for 70B | Removed; replaced with fp16 weight sizes |
| fp16 size 1,440 GB for a 2.4T model | models_and_hardware | 2.4T × 2 bytes ≈ 4,800 GB | Row removed |
| Mac "800 to 1,600 GB/s" | models_and_hardware | Hardware spec: M-series tops out near 800 GB/s | Corrected |
| LoRA "198 µs (84% hit)" | comparison | Another unseeded synthetic run; differs from §11 | Removed |
| Reasoning scanner "2.47M tok/s" | README; index | Committed L4 run: 2.45M chunks/s **at 0.49× the Python baseline**; Mac 0.54× | Removed |

## 3. Measured on synthetic data, toys or placeholders (class B)

These numbers are what the scripts measured, but not on a model, so they were removed from the user-facing pages; the scripts now say what they run.

| Claim | Where | Producing script:line | Data | Action |
| :--- | :--- | :--- | :--- | :--- |
| §1 Llama-3-8B / 3.1-70B / Qwen-2.5-72B layer 2.24× / 2.32× / 2.41× | §1; comparison | `benchmark_comprehensive_real_gpu.py:209-246` | One random-weight batch-1 FFN layer; cuBLAS on smaller matrices (no Triton); matches `results_gcp_master_benchmark_v4.json` | **Kept**, relabelled as one random-weight layer |
| §11 LoRA hit 184 µs, cold switch 0.95 ms, 3,094 req/s, 83.4% | §11 | `benchmark_lora_and_speculation.py:40-86` | Random adapters, unseeded 80/20 traffic; hit time includes a LoRA forward | Removed |
| §11 EAGLE3 draft 0.626 ms | §11 | `benchmark_lora_and_speculation.py:89-144` | Random-init draft head, randn input | Removed |
| §11 70B cold start 254 ms | §11 | `turing/models/streaming_loader.py:139-166` | `torch.randn` for 3 layers, `torch.empty` for 77; no file I/O | Removed; script labels it synthetic |
| §12B / §17 k-slot pooling 1.3×–3.1×; README / index "3.1×" | §12B, §17 | `benchmark_lineage_strategies.py:129-162` | Random-init pooler, randn K/V, one timed iteration; committed L4 shows 20× slower at N=512, Mac 0.2–0.7× | Removed; script now warms up and takes a median of 20 |
| §12C lane classification 0.328 µs | §12C, §18 | `benchmark_traffic_and_spec.py:83-88` | Times `classify_request`, not a sort | Removed; label fixed |
| §13 prefix hasher 28.76×, parity verifier 13.72×, lineage KV hashing 8.49×, Quadtree 3.31 ms | §13 | `benchmark_native_fusions_v2.py:29-126` | Fixed token lists, randn tensors; the "AVX2 SIMD" hasher is a scalar loop; the VRAM hash converts floats to int instead of bitcasting | Removed |
| §14 gateway L4 latencies and tok/s | §14; comparison | `benchmark_triple_gateway_and_structured.py` | In-process TestClient; tok/s on the random-init `test-tiny` model | Removed |
| §15 prefill / decode / jitter (L4) | §15 | `benchmark_prefill_decode_duality.py:27-206` | Random-init 4-layer fp32 toy; "8K burst" sends 1,024 tokens; GFLOP/s from a formula | Removed |
| §16 GGUF Q8_0 dequant 3.19×, fused RMSNorm+SwiGLU | §16 | `benchmark_fused_layers.py:23-94` | Random int8 blocks; the FFN ran on CPU tensors, so Triton never ran | Removed; script now uses CUDA when present |
| §20 prefix router 8.80×, k-slot gate 1.21×, header parser 1.04× | §20 | `benchmark_fused_kernels_and_simd.py:71-167` | Synthetic tokens, randn, a synthetic JSON header | Removed |
| §21 batch-union prefill up to 16.92×, spec-verify batch throughput | §21; README; comparison | `benchmark_vectorized_batch_union.py:78-153` | randn activations, random logits; "verification" is routing only | Removed |
| §21 hit rate 39.14% → 92.53%, LFRU 1,142 → 56 evictions, EMA 22 → 0 flips | §21; README; index; comparison | `benchmark_routing_history_and_pinning.py`, `benchmark_lfru_eviction_hysteresis.py`, `benchmark_bandwidth_adaptive_ema.py` | Seeded synthetic Zipf traces (identical output on Mac and L4); pins learned from the stream they are scored on | Removed |
| §22A–D paged decode GEMV 8.86×, linear recurrence 5.13×, tensor hash 23.00×, traffic manager 1.27× | §22; README; index | `benchmark_fused_kernels_and_simd_v5.py:116-492` | randn KV pools (baseline is a per-request Python gather loop), randn q/k/v, randn buffer, synthetic admit/release | Removed |
| §22E GPT-2 95.8 tok/s, SmolLM2 29.3 tok/s (class A, but HF `generate`) | §22E | `benchmark_comprehensive_real_gpu.py:104-110` | HuggingFace, not Turing; recorded L4 run 68.9 tok/s; SmolLM2 generated one token; cleanup was inside the timed window | Removed; script now forces 48 new tokens and times outside cleanup (Mac: GPT-2 58.8, SmolLM2-1.7B-Instruct 19.6 tok/s) |
| §22E "99.41% / 99.40% signal fidelity" | §22E | `benchmark_comprehensive_real_gpu.py:119-122` | INT8 rounding on `torch.randn` with rank = head_dim (no truncation) | Removed |
| §22E "131k NIAH 100% top-1" | §22E | `turing/serving/niah.py:40-63` | One 512-row page of random keys; context length only labels the row; the query is the needle | Removed; module and script relabelled |
| §23A split-K 1.36× / 13,966 tok/s; README "1.36×" | §23A; README | `benchmark_nextgen_speedups.py:81-117` | randn; 0.36× at N=1,024 | Removed |
| §23B KDA step 2.61 ms | §23B | `benchmark_nextgen_speedups.py:135-150` | Plain PyTorch, no baseline | Removed |
| §23C prefix match 152.42 µs; README "<155 µs", "O(1)" | §23C; README | `benchmark_nextgen_speedups.py:164-183` | Synthetic token lists; the matcher is O(n) | Removed |
| §23D cosine 99.999% | §23D | `benchmark_nextgen_speedups.py:226` | randn | Removed |
| §23E 2:4 sparse prefill 1.28× | §23E | `turing/kernels/triton_sparse_subspace_prefill.py:89-96` | `index_select` of 43% of channels; no 2:4 sparsity, no Triton launch | Removed |
| §24 segment pooling 65.40× / 135.35×, adaptive batching −67.43% | §24; README; index; comparison | `benchmark_sie_pipeline_speedups.py:31-126` | randn hidden states; exponential synthetic lengths (adaptive batching sorts by length, the baseline does not) | Removed |
| §24 five pipeline stages (Docling, GLiNER, BGE, reranker, LLM), total 167.34 ms | §24; README; index; comparison | `turing/serving/multi_model.py:136, 179, 222, 253, 261` | No model is loaded: substring match with a fixed 0.965 score, randn embeddings, word-overlap rerank, no generation | Removed; endpoints documented as placeholders |
| §25 recall table, prefill 1.61×, decode 48.84 µs; README "25.00%" | §25; README; index; comparison | `benchmark_associative_recall_and_decay.py:101-268` | Toy models trained from scratch for 300 steps, n=128 per cell; current MPS JSON shows chunkwise 0.95× | Removed |
| §26 hybrid anchor 61.14×, spec entropy 3.33×, routing history 444 µs | §26; index | `run_gcp_master_validation_and_benchmarks_v8.py:49-210` | randn; spec-entropy baseline forces `.cpu()` each iteration; 444 µs is ~925 ns per record, slower than the claimed baseline | Removed |
| Logits match HF "1.8e-7 / 2e-7" | README | `tests/test_hf_parity.py:11-37` | Random-init 4-layer Llama (hidden 64); the test asserts `atol=1e-3`; this audit's re-run gives 2.1e-7 | Relabelled, with the real-weight NLL result (§2A) alongside |
| int8 page pool "47% smaller than fp16" | README | `benchmark_nextgen_speedups.py:228-229` | Storage arithmetic, (32+2)/64 | Removed with the rest of the bullet |

## 4. Capability and competitor statements (class E)

| Statement | Where | Finding | Action |
| :--- | :--- | :--- | :--- |
| "Pruned into active subspaces (−57%)" at ingestion | comparison §1 | Default `--sparsity 0.0`; HF checkpoints load via transformers `from_pretrained` | Corrected |
| KV management "SVD INT8 Paged" | comparison | Serving uses `KVCache` (int8/int4); `use_svd_kv` is stored but never used | Corrected |
| Prefix caching via `SpectralRadixSVD` | comparison | Only used in `turing/demo/engine_wrapper.py` | Corrected: not in the serving path |
| Branch prefix sharing "Zero-Copy SVD" | comparison | `LocalExecutor.generate_branch` re-prefills the prefix | Corrected |
| Multi-node TP + PP "Native NCCL" | comparison; README | PP stages run in one process; TP layers are not applied to loaded models | Relabelled "building blocks; not validated" |
| `/v1/rerank`, `/v1/extract`, `/v1/embeddings` "Native" | comparison; index | Placeholders (see §24 above) | Relabelled |
| "Literally ANY open-weight model … out of the box"; 70B–320B on one 24 GB GPU; "Supported Frontier & MoE Models" table | models_and_hardware; index; README | Only SmolLM2 (Turing) and GPT-2 / SmolLM2-Instruct (HF) were run; Qwen2/Qwen3/Llama-3 RoPE scaling not mapped; fp16 70B weights are ~141 GB; several README short names are not CLI aliases | Replaced with what was run and fp16 sizes |
| AMD ROCm, Intel XPU, Vulkan support | README; index | Dispatch code exists; no recorded run | Noted as unrecorded |
| vLLM "OpenAI only", "no rerank (TEI needed)", "Apple experimental"; SGLang/Ollama/llama.cpp columns | README; comparison | Competitor claims, not checked here; some are likely out of date | Disclaimer added to both tables |

## 5. Script changes

Each script below now prints only what it computes, and labels synthetic inputs:

`benchmark_traffic_and_spec.py`, `benchmark_lineage_strategies.py`, `benchmark_lora_and_speculation.py`, `benchmark_xkv_deliberation.py`, `benchmark_latent_decode.py` (byte count from tensors, honest OOM/skip labels, MPS sync), `benchmark_triple_gateway_and_structured.py`, `benchmark_prefill_decode_duality.py`, `benchmark_fused_kernels_and_simd_v5.py`, `benchmark_fused_kernels_and_simd.py`, `benchmark_sie_pipeline_speedups.py`, `run_gcp_master_validation_and_benchmarks_v8.py`, `benchmark_associative_recall_and_decay.py`, `benchmark_cold_readahead_gcp.py`, `benchmark_comprehensive_real_gpu.py`, `benchmark_fused_layers.py`, and the `turing/serving/niah.py` docstring.

## 6. Mac re-runs made for this audit (Apple M4 Pro)

| Script | Result | Note |
| :--- | :--- | :--- |
| `benchmark_traffic_and_spec.py` | Admission 0.48 µs/decision; lane classification 0.08 µs | Pure Python; committed as `results_traffic_spec.json` |
| `benchmark_lineage_strategies.py --device mps` | Synthetic translator (gain 0.5): naive residual 512 → 3,686 over 6 turns, clean_base flat at 512 | Illustrates the algebra only |
| `benchmark_lora_and_speculation.py --device mps` | Pool 13.1 MB, 32 slots 4.2 MB, 82.2% hits, hit P50 218 µs | Synthetic traffic |
| `benchmark_comprehensive_real_gpu.py` | HF `generate`, batch 1, 48 tokens: GPT-2 58.8 tok/s, SmolLM2-1.7B-Instruct 19.6 tok/s | Not the Turing engine |
| `benchmark_sie_pipeline_speedups.py` | Placeholder pipeline 583 ms; declared estimates 7.42 GB | Placeholders |
| `benchmark_triple_gateway_and_structured.py` | /api/tags 323–398 µs (50/50 HTTP 200); random-init `test-tiny` 83–133 tok/s across two runs | Toy model |

None of these was added to the benchmarks page, because none measures the engine on a model.

## 7. Not done here

* **Ingest ladder** (`scripts/benchmark_ingest_ladder_all_tiers.py`, tiers 3–5 constants, GB/s over the full file): the doc claims are removed. The script fix belongs to the offload-fixes work (expert cache, async reader, synthetic ingest timings) and is made on that branch.
* **Library bugs found in passing** (they affect what scripts measure, not what the docs now say): TTFT in `turing/serving/engine.py:277, 310` is stamped before the forward pass; P99 falls back to the mean below 10 samples (`engine.py:403`); `turing/kernels/triton_vram_hash.py:33` converts floats to int instead of bitcasting; `triton_sparse_subspace_prefill.py` never launches its kernel.
* **Other pages that repeated removed numbers**: `docs/architecture.md`, `docs/serving.md` and the top-level `ARCHITECTURE.md` were cleaned in a follow-up commit (numbers removed or pointed at the measured sections, components marked experimental), except the ingest Tier 1–5 bullets in `docs/architecture.md` §19, which the offload-fixes branch rewrites with cold-measured numbers. Scripts not cited by these pages also print unverified claims, for example `benchmark_multi_gpu_mig.py` "partitions" an L4 into MIG slices (the L4 does not support MIG) and `benchmark_extended_matrix.py` computes VRAM as `fp16 × 0.166`.
* **GPU re-measurement**: nothing here was re-run on an L4; GPU runs need the owner's approval.
