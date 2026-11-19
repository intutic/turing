# 📊 Benchmarks

Every table on this page is generated from a committed `results_*.json` file by `scripts/make_doc_tables.py`
(`--check` fails when a table and its file disagree). GPU numbers are from one NVIDIA L4 (24 GB) on a Google Cloud
`g2-standard-16` Spot VM; Mac numbers from an Apple M4 Pro. ± is one standard error; n is given where it matters.
Comparisons with vLLM and llama.cpp are on the [comparison](comparison.md) page.

---

## 1. Summary

* **Turing is close to vLLM and ahead in a few settings** on the L4 (§5, [comparison](comparison.md)): vLLM is
  1.01–1.04× faster in fp16 and 1.06–1.12× at 4 bits; with the opt-in 4-bit LM head Turing is 1.15× vLLM at batch 1
  and 8; at 16K-token prompts Turing's fp16 is 1.0–1.22× faster per request, and vLLM's fp8 with layer 0 kept is
  1.3–1.4× faster than Turing's int8 at equal quality.
* **Its low-bit KV caches keep quality without settings** (§2): int8 is free, int4 costs about +0.02 nats, and both
  pass every pass-key retrieval to 32K; vLLM's int8/int4 and default fp8 caches lose 16 to 57 GSM8K points on the same
  model unless layer 0 is excluded by hand.
* **4-bit weights** decode 1.5–1.6× faster than fp16 in Turing under CUDA graphs (§3), with GPTQ costing 5 GSM8K
  points; TEAL activation sparsity adds 3–7% speed at batch 1 for 5–9 more points, which is rarely worth it.

---

## 2. KV cache quality and the layers it keeps

**How the cache stays correct.** When a pool stores KV in int8 or int4, Turing calibrates once at startup on 4 × 256
random tokens: each layer's K and V take the pool's round trip alone, and a layer whose next-token distribution moves
by more than 0.02 nats (KL divergence) is kept in the model's dtype, at most a quarter of the layers. For int4, key
channels at 16× or more of their head's median magnitude are also divided by a power of two (and the queries
multiplied by it), which leaves q·k unchanged bit for bit. Why these rules, from diagnostics on real models:

* On Qwen2.5-1.5B, int4 KV cost +1.93 nats, of which +1.77 came from layer 0's keys alone; every other layer cost
  under 0.005 and quantizing every value cost +0.002. Random-token calibration finds that layer (KL 0.13 against a
  median layer of 0.0006); attention-output error does not (Qwen2.5-0.5B's layer 0 has more error and costs +0.014).
* Key smoothing helps int4 where the model has real outlier channels (Qwen2's bias channels reach 64× the median):
  +0.040 → +0.013 nats on Qwen2.5-1.5B. Applied to SmolLM2's mild channels (at most 8×) it doubled int4's cost, and
  applied to int8 in bf16 it cost 16 GSM8K points (42.2% against 58.2%), so it is int4-only and outlier-only.

**NLL on WikiText-2** (fp16, 16 windows of 2,048 tokens, last 256 scored; paired against the exact model;
`results_quality_cuda.json`):

<!-- table:kv_quality -->
| Setting (ΔNLL vs exact) | SmolLM2-360M | SmolLM2-1.7B | Qwen2.5-1.5B |
| :--- | :---: | :---: | :---: |
| int8 KV | -0.0001 ± 0.0004 | -0.0001 ± 0.0003 | +0.0003 ± 0.0005 |
| int4 KV (smoothed, protected layers) | +0.0245 ± 0.0043 | +0.0189 ± 0.0029 | +0.0126 ± 0.0042 |
| int4 KV, no smoothing | +0.0245 ± 0.0043 | +0.0230 ± 0.0020 | +0.0402 ± 0.0081 |
| Quest, 25% of pages | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 |
| 4-bit weights, RTN | +0.1343 ± 0.0110 | +0.1279 ± 0.0118 | +0.0937 ± 0.0072 |
| 4-bit weights, GPTQ | +0.0676 ± 0.0069 | +0.0474 ± 0.0074 | +0.0344 ± 0.0054 |
| TEAL 25% (decode) | +0.0195 ± 0.0035 | +0.0270 ± 0.0057 | +0.0147 ± 0.0031 |
| TEAL 40% (decode) | +0.0942 ± 0.0091 | +0.0983 ± 0.0087 | +0.0805 ± 0.0092 |
| TEAL 50% (decode) | +0.2418 ± 0.0172 | +0.2323 ± 0.0139 | +0.1851 ± 0.0113 |
| token by token vs one pass (sanity) | -0.0001 ± 0.0001 | -0.0000 ± 0.0001 | +0.0001 ± 0.0001 |
<!-- /table -->

**Qwen2.5-1.5B-Instruct in bf16, against vLLM's own caches on the same checks** (GSM8K: first 500 test problems,
8-shot, greedy; pass-key: 20 needles at 2K–32K; ΔNLL: 8 WikiText-2 windows; time per request from §5;
`results_gsm8k_final_cuda.json`, `results_niah_final_cuda.json`, `results_quality_instruct_bf16_cuda.json`,
`results_quality_vllm_cuda.json`, vLLM 0.31):

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 2.05 |
| Turing int8 | — | 58.2% | 100.0% | +0.003 ± 0.002 | 2.32 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.95 |
| vLLM fp16 (reference) | — | 57.4% | 100.0% | — | 2.51 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.66 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
| vLLM fp8_per_token_head | — | 0.6% | 0.0% | +2.404 ± 0.364 | — |
| vLLM int8_per_token_head | — | 41.4% | 100.0% | +0.023 ± 0.007 | 4.72 |
| vLLM int4_per_token_head | — | 0.6% | 0.0% | +8.127 ± 0.495 | 4.41 |
| vLLM int4_per_token_head | layer 0 (manual) | 47.8% | 100.0% | +0.058 ± 0.010 | 4.30 |
| vLLM turboquant_k8v4 | — | 52.0% | 100.0% | +0.001 ± 0.001 | — |
<!-- /table -->

The vLLM caches that collapse do so silently: the server starts and answers. Skipping layer 0 (vLLM's
`kv_cache_dtype_skip_layers`) repairs fp8 and int4, which is what Turing's calibration does without being told.
Scope: measured on Qwen2.5 (0.5B, 1.5B, 1.5B-Instruct) and SmolLM2 (360M, 1.7B).

**GSM8K exact match** (Qwen2.5-1.5B-Instruct, first 500 test problems, 8-shot, greedy; `results_gsm8k_final_cuda.json`,
`results_gsm8k_teal_cuda.json` (200 problems, batch 1, where TEAL applies), `results_gsm8k_diag_cuda.json`):

<!-- table:gsm8k -->
| Setting | dtype | Batch | GSM8K exact match |
| :--- | :---: | :---: | :---: |
| native KV | bf16 | 32 | 57.2% ± 2.2% |
| int8 KV | bf16 | 32 | 58.2% ± 2.2% |
| int4 KV | bf16 | 32 | 52.8% ± 2.2% |
| native (TEAL baseline) | bf16 | 1 | 61.0% ± 3.5% (n=200) |
| TEAL 40% | bf16 | 1 | 52.5% ± 3.5% (n=200) |
| int8 KV, smoothing off | bf16 | 32 | 58.2% ± 2.2% |
| int8 KV | fp16 | 32 | 59.0% ± 2.2% |
| native KV | fp16 | 32 | 57.6% ± 2.2% |
<!-- /table -->

**Pass-key retrieval** (Qwen2.5-1.5B-Instruct, bf16, 5 depths per length; `results_niah_final_cuda.json`):

<!-- table:niah -->
| KV cache | 2K | 8K | 16K | 32K | All |
| :--- | :---: | :---: | :---: | :---: | :---: |
| native KV | 5/5 | 5/5 | 5/5 | 5/5 | 100.0% |
| int8 KV | 5/5 | 5/5 | 5/5 | 5/5 | 100.0% |
| int4 KV | 5/5 | 5/5 | 5/5 | 5/5 | 100.0% |
<!-- /table -->

**Six instruct models** (NVIDIA L4, vLLM 0.31, 2026-11-18). The calibration, run as
[`kv-layer-guard`](../packages/kv-layer-guard) for vLLM's fp8 cache and inside Turing's own int8/int4 pools. GSM8K
is 8-shot exact match on 500 problems unless noted, bf16 weights; pass-key is 5 depths per length, up to 32K for the
Qwen and Llama models and up to 16K (8K for Gemma 2) for the others. Results files: `results_quality_vllm_validation_cuda.json`,
`results_guard_*_cuda.json`, `results_gsm8k_validation_cuda.json`, `results_niah_validation_cuda.json`, and for
Phi-3.5's Turing pass-key `results_niah_phi35_longrope_fix_cuda.json`. Phi-3.5's first Turing run found the key 0% of
the time above 4K: a LongRoPE bug in chunked prefill (each chunk picked its own short/long frequencies), fixed in
2.0.0 and rerun.

<!-- table:validation -->
| Model | vLLM bf16 | vLLM fp8 (default) | Guard keeps in bf16 | vLLM fp8 + guard | Turing int8 KV | Turing int4 KV | Pass-key: fp8 / fp8 + guard / Turing int4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-1.5B-Instruct | 57.4% | 2.2% | 0 | 56.8% | 58.2% | 52.8% | 5.0% / 100.0% / 100.0% |
| Llama-3.2-3B-Instruct | 67.2% | 65.8% | none | = fp8 | 67.0% | 64.4% | 100.0% / 100.0% / 100.0% |
| gemma-2-2b-it | 40.8% | 40.4% | none | = fp8 | 40.2% | 38.0% | 100.0% / 100.0% / 100.0% |
| Phi-3.5-mini-instruct | 71.8% | 66.6% | none | = fp8 | 79.8% | 77.8% | 100.0% / 100.0% / 100.0% |
| mistral-7b-instruct-v0.3 | 47.8% | 47.0% | none | = fp8 | — | 46.0% (n=200) | 100.0% / 100.0% / 100.0% |
| Qwen2.5-7B-Instruct | 74.6% | 2.4% | 0, 1, 27 | 70.2% | — | 71.5% (n=200) | 0.0% / 100.0% / 100.0% |
<!-- /table -->

- The guard keeps layers only where vLLM's default fp8 cache breaks the model: layer 0 of Qwen2.5-1.5B and layers
  0, 1 and 27 of Qwen2.5-7B (GSM8K 2.2% and 2.4% without it). On the other four it keeps none, and fp8 there costs
  at most 1.4 points.
- **Limit:** Phi-3.5-mini loses 5.2 GSM8K points under fp8 (71.8% to 66.6%; each figure has a standard error of
  about 2 points) and the guard does not flag it: its largest per-layer KL is 0.006, under the 0.02 threshold. A loss
  spread thinly over many layers is not what a per-layer check catches.
- Phi-3.5 scores higher in Turing (79.8% with int8 KV) than in vLLM bf16 (71.8%). Both runs use the same evaluator
  and prompts; vLLM was started with a 16K maximum length, which may make it use LongRoPE's long frequencies for every
  request. A vLLM run capped at 4K is queued to test that; until then the gap is unexplained.

### Reproduce
```bash
python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B \
  --arms exact_forward,exact_decode,int8,int4,int4_nosmooth,quest25,rtn4,gptq4,teal25,teal40,teal50 --windows 16
turing eval-accuracy --model Qwen/Qwen2.5-1.5B-Instruct --samples 500 --dtype bf16 --kv-cache-dtype int4
turing eval-niah --model Qwen/Qwen2.5-1.5B-Instruct --context-len 2048 8192 16384 32768 --kv-cache-dtype int4
~/vllm-venv/bin/python scripts/eval_vllm_quality.py --kv-cache-dtype fp8 --kv-skip-layers 0
```

---

## 3. 4-bit weights and activation sparsity

**Decode speed at batch 1** (512-token prompt, 128 greedy steps; `results_weight_formats_cuda.json`,
`results_weight_formats_qwen_cuda.json`). Serving decodes through CUDA graphs; the eager column launches every kernel
from Python and shows why that matters for small Triton kernels:

<!-- table:formats -->
| Model | Weights | Decode tok/s (CUDA graphs) | Eager | Weight GiB |
| :--- | :---: | :---: | :---: | :---: |
| SmolLM2-1.7B | fp16 | 61.4 | 32.9 | 3.38 |
| SmolLM2-1.7B | rtn4 | 96.5 | 17.4 | 1.20 |
| SmolLM2-1.7B | gptq4 | 97.1 | 17.5 | 1.20 |
| SmolLM2-1.7B | teal25 | 68.7 | 16.1 | 3.38 |
| SmolLM2-1.7B | teal40 | 76.7 | 16.3 | 3.38 |
| SmolLM2-1.7B | teal50 | 82.7 | 16.2 | 3.38 |
| SmolLM2-1.7B | rtn4_teal40 | 123.4 | 15.7 | 1.20 |
| Qwen2.5-1.5B | fp16 | 68.7 | 27.4 | 3.31 |
| Qwen2.5-1.5B | rtn4 | 102.0 | 14.6 | 1.54 |
| Qwen2.5-1.5B | gptq4 | 100.9 | 14.4 | 1.54 |
| Qwen2.5-1.5B | teal25 | 73.1 | 13.6 | 3.31 |
| Qwen2.5-1.5B | teal40 | 82.1 | 13.5 | 3.31 |
| Qwen2.5-1.5B | teal50 | 88.7 | 13.4 | 3.31 |
| Qwen2.5-1.5B | rtn4_teal40 | 114.9 | 13.1 | 1.54 |
<!-- /table -->

**Where each int4 kernel wins** (microseconds, summed over seven layer shapes from 1536 × 1536 to the 1536 × 151,936
LM head of Qwen2.5-1.5B, timed as CUDA-graph replays; `results_w4a16_crossover_cuda.json`). The Triton kernel splits K
across programs so a narrow layer still fills the GPU, and serves up to `INT4_MAX_ROWS = 64` rows. Marlin (vendored from
vLLM, built only with `TURING_BUILD_MARLIN=1`) is faster at every row count here, so when it is built `QuantLinear`
sends it 1–1,024 rows:

<!-- table:crossover -->
| Rows (M) | int4 kernel | int4, no K split | dequantize + cuBLAS | fp16 cuBLAS | Marlin (optional) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 1,385 | 1,408 | 6,712 | 3,076 | 611 |
| 2 | 2,160 | 2,198 | 7,270 | 3,058 | 615 |
| 4 | 1,453 | 1,554 | 7,282 | 3,066 | 623 |
| 8 | 1,472 | 1,624 | 7,306 | 3,088 | 634 |
| 16 | 1,756 | 1,932 | 7,344 | 3,108 | 680 |
| 32 | 2,425 | 2,485 | 7,727 | 3,452 | 843 |
| 64 | 3,776 | 3,809 | 7,805 | 3,528 | 1,270 |
| 128 | 6,860 | 6,884 | 8,054 | 3,849 | 2,448 |
| 256 | 12,823 | 12,868 | 9,767 | 5,740 | 4,862 |
<!-- /table -->

TEAL sparsifies projection inputs only in one-row decode on CUDA, where the input-sparse kernel skips the weight rows
of zero inputs. Applied to a whole prefill it broke attention-sink tokens (SmolLM2-1.7B: +6.4 nats at 40%), so
prefill and batched decode stay dense.

---

## 4. Decode throughput against HuggingFace transformers

One WikiText-2 prompt is prefilled, its cache copied into B requests, and 32 batched decode steps are timed through
the serving path; the paged pool takes 85% of free GPU memory. 'pool full' = the pool cannot hold the batch, OOM =
CUDA out of memory, — = not run. Contexts above 8,192 exceed SmolLM2's trained length (speed only).

**SmolLM2-360M** (`results_kv_decode_cuda_smollm2_360m.json`), generated tokens per second:

<!-- table:decode_360m -->
| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 2K × B=64 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 8K × B=64 | 32K × B=1 | 32K × B=8 | 32K × B=32 | 32K × B=64 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 31 | 241 | 774 | 820 | 31 | 201 | 212 | OOM | 31 | 56 | OOM | — |
| Turing, native, no CUDA graph | 24 | 190 | 758 | 1,485 | 24 | 191 | 463 | pool full | OOM | — | — | — |
| Turing, native cache | 164 | 895 | 1,510 | 1,716 | 132 | 381 | 468 | pool full | OOM | — | — | — |
| Turing, int8 | 165 | 885 | 1,507 | 1,705 | 129 | 381 | 459 | 492 | OOM | — | — | — |
| Turing, int4 | 164 | 831 | 1,434 | 1,590 | 128 | 352 | 433 | 450 | OOM | — | — | — |
| Turing, int8 + Quest 25% | 164 | 876 | 1,488 | 1,690 | 104 | 260 | 256 | OOM | OOM | — | — | — |
<!-- /table -->

**SmolLM2-1.7B** (`results_kv_decode_cuda_smollm2_1p7b.json`):

<!-- table:decode_1p7b -->
| Setting | 2K × B=1 | 2K × B=8 | 2K × B=32 | 8K × B=1 | 8K × B=8 | 8K × B=32 | 32K × B=1 | 32K × B=8 | 32K × B=32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| HuggingFace transformers | 39 | 136 | 158 | 28 | 43 | OOM | 10 | OOM | — |
| Turing, native, no CUDA graph | 31 | 223 | 353 | 31 | 87 | pool full | 18 | pool full | — |
| Turing, native cache | 56 | 229 | 357 | 41 | 88 | pool full | 19 | pool full | — |
| Turing, int8 | 56 | 227 | 356 | 40 | 87 | pool full | 19 | pool full | — |
| Turing, int4 | 55 | 214 | 325 | 39 | 80 | pool full | 17 | pool full | — |
| Turing, int8 + Quest 25% | 56 | 226 | 355 | 37 | 50 | pool full | 19 | OOM | — |
<!-- /table -->

**Quest page selection at 25%** against reading every page, both int8, SmolLM2-360M, selection forced on at every
length (`results_quest_threshold_cuda_smollm2_360m.json`; > 1 means Quest is faster):

<!-- table:quest -->
| Batch | 2K | 4K | 8K | 16K | 32K |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.77× | 0.81× | 0.87× | OOM | OOM |
| 8 | 0.64× | 0.89× | 0.88× | — | — |
| 64 | 0.40× | 0.71× | 0.64× | — | — |
<!-- /table -->

Quest is slower at every length measured here, so it stays opt-in (`--kv-page-budget`) and starts only from
`kv_select_min_tokens` (8,192) when enabled.

### Reproduce
```bash
python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M \
  --settings hf_transformers,native_eager,native,int8,int4,int8_quest25 --contexts 2048,8192,32768 --batches 1,8,32,64
```

---

## 5. Long context at equal KV memory

Qwen2.5-1.5B-Instruct, batches of 16 to 128 requests, each a 16K-token prompt and 256 new tokens, on one L4. vLLM runs at `gpu_memory_utilization` 0.85 and records the KV memory it took (14.6 GiB); every
Turing setting gets the same bytes, so capacity differs only by bytes per token. Seconds per request
(`results_longctx_cuda.json`):

<!-- table:longctx -->
| Engine, KV cache | KV capacity (tokens) | B=16 | B=32 | B=64 | B=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing native | 547,840 | 2.25 | 2.12 | 2.07 | 2.05 |
| Turing int8 | 993,648 | 2.52 | 2.37 | 2.34 | 2.32 |
| Turing int4 | 1,305,488 | 3.14 | 3.03 | 2.98 | 2.95 |
| Turing int4, Quest 25% | 1,136,256 | 3.78 | 3.66 | 3.48 | 3.40 |
| vLLM auto | 547,856 | 2.26 | 2.27 | 2.43 | 2.51 |
| vLLM fp8 | 1,068,816 | 1.83 | 1.70 | 1.67 | 1.66 |
| vLLM fp8, skip layer 0 | 1,032,096 | 1.85 | 1.74 | 1.75 | 1.73 |
| vLLM int8_per_token_head | 1,062,096 | 4.90 | 4.79 | 4.74 | 4.72 |
| vLLM int4_per_token_head | 2,060,896 | 4.73 | 4.55 | 4.49 | 4.41 |
| vLLM int4_per_token_head, skip layer 0 | 1,876,928 | 4.51 | 4.42 | 4.36 | 4.30 |
<!-- /table -->

In fp16 Turing is 1.0–1.22× faster per request than vLLM; prefill attention over pages in the model dtype runs through
FlashInfer when it is installed. Turing's int4 cache holds 2.4× the tokens of fp16 but takes 1.4× its time per request
(2.95 against 2.05 s at B=128), and int8 1.1×: their prefill reads quantized pages through Turing's own kernel. Quest
at 25% of pages is slower still (3.40 s at B=128). vLLM's
fp8 cache with layer 0 kept is 1.3–1.4× faster than Turing's int8 at equal quality. Quality of each format is in §2;
vLLM fp8 without a skipped layer is fast because it is wrong on this model.

---

## 6. Cold start

Time from process start to the first generated token on the L4 (fresh process per run; cold = the model files
dropped from the page cache, Python and CUDA libraries cached; `results_cold_start_cuda.json`):

<!-- table:cold_start -->
| Model | Format | Page cache | GB | n | Process start → first token, s | Load, s | Imports, s | Load → first token, s |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-1.5B-Instruct | safetensors | cold | 3.10 | 5 | 25.43 ± 0.07 | 17.74 ± 0.06 | 5.67 | 1.99 |
| Qwen2.5-1.5B-Instruct | safetensors | warm | 3.10 | 5 | 10.04 ± 0.02 | 2.45 ± 0.01 | 5.58 | 1.98 |
| Qwen2.5-1.5B-Instruct | tgate4 | cold | 1.66 | 5 | 17.83 ± 0.03 | 10.17 ± 0.01 | 5.58 | 2.05 |
| Qwen2.5-1.5B-Instruct | tgate4 | warm | 1.66 | 5 | 9.83 ± 0.02 | 2.19 ± 0.01 | 5.56 | 2.05 |
| SmolLM2-1.7B | safetensors | cold | 3.43 | 5 | 27.21 ± 0.02 | 18.87 ± 0.00 | 4.26 | 4.05 |
| SmolLM2-1.7B | safetensors | warm | 3.43 | 5 | 10.23 ± 0.01 | 2.01 ± 0.00 | 4.15 | 4.05 |
| SmolLM2-1.7B | tgate4 | cold | 1.29 | 5 | 21.50 ± 0.02 | 7.45 ± 0.00 | 4.16 | 9.87 |
| SmolLM2-1.7B | tgate4 | warm | 1.29 | 5 | 15.32 ± 0.02 | 1.28 ± 0.00 | 4.16 | 9.86 |
<!-- /table -->

The 4-bit `.tgate4` directory loads 1.7–2.5× faster from a cold disk. On Qwen2.5-1.5B its first token now follows the
load by about 2 s in either format. The SmolLM2-1.7B rows come from an earlier version of the code and were not rerun;
their 4–10 s from load to first token may not reflect the current code. For comparison, vLLM takes 51.7 s cold
and 34.2 s warm to its first token on the same Qwen files and llama.cpp 7.1 s and 1.3 s with its 4-bit GGUF
(`results_cold_start_vs_vllm_cuda.json`, `results_cold_start_llamacpp_cuda.json`).

---

## 7. MoE expert offload

OLMoE-1B-7B (16 MoE layers × 64 experts, top-8) with every routed expert in host memory and a GPU expert cache of
64–512 experts, against all experts on the GPU (4 WikiText-2 windows, 256-token prompts, 32 teacher-forced decode
steps; `results_moe_offload_cuda.json`). "decider" picks copy-to-GPU or run-on-CPU per miss; "device" always copies:

<!-- table:moe -->
| Configuration | Decode tok/s | Expert-cache hit rate (decode) | Peak GPU GiB | Same greedy token as all-on-GPU |
| :--- | :---: | :---: | :---: | :---: |
| offload slots=64 lru decider | 4.90 | 17.8% | 2.7 | 98.4% |
| offload slots=64 lru device | 0.56 | 0.0% | 2.7 | 99.2% |
| offload slots=64 lfru decider | 5.11 | 17.7% | 2.7 | 98.4% |
| offload slots=64 lfru device | 4.71 | 19.5% | 2.7 | 99.2% |
| offload slots=128 lru decider | 5.80 | 34.0% | 3.4 | 99.2% |
| offload slots=128 lru device | 0.74 | 25.2% | 3.4 | 99.2% |
| offload slots=128 lfru decider | 5.87 | 34.3% | 3.4 | 99.2% |
| offload slots=128 lfru device | 5.07 | 30.4% | 3.4 | 99.2% |
| offload slots=256 lru decider | 6.10 | 36.2% | 4.9 | 99.2% |
| offload slots=256 lru device | 0.99 | 44.8% | 4.9 | 99.2% |
| offload slots=256 lfru decider | 5.94 | 36.2% | 4.9 | 99.2% |
| offload slots=256 lfru device | 5.45 | 47.2% | 4.9 | 98.4% |
| offload slots=512 lru decider | 6.00 | 36.2% | 7.9 | 99.2% |
| offload slots=512 lru device | 1.83 | 71.7% | 7.9 | 99.2% |
| offload slots=512 lfru decider | 5.96 | 36.2% | 7.9 | 99.2% |
| offload slots=512 lfru device | 7.03 | 72.4% | 7.9 | 98.4% |
| all experts on the GPU | 15.92 | — | 13.9 | — |
<!-- /table -->

---

## 8. Storage ingestion

`scripts/benchmark_ingest_ladder_all_tiers.py` reads a 2 GiB file of tensor-sized segments cold (written through
`F_NOCACHE` on macOS, dropped with `posix_fadvise` on Linux, `mincore` confirms nothing is cached).

**Apple M4 Pro SSD** (`results_ingest_ladder.json`; the machine was shared, so the spread is wide):

<!-- table:ingest_mac -->
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
<!-- /table -->

**L4 VM boot disk** (`results_ingest_ladder_linux_l4.json`; a network-attached persistent disk):

<!-- table:ingest_linux -->
| Row | Cache | GB/s | ms | n |
| :--- | :---: | ---: | ---: | :---: |
| Tier 0: mmap demand paging | cold | 0.19 ± 0.00 | 5,546 | 3 |
| Tier 1: mmap + MADV_WILLNEED | cold | 0.19 ± 0.00 | 5,699 | 3 |
| Tier 2: ring reader, read_segments_parallel (bytes) | cold | 0.19 ± 0.00 | 5,571 | 3 |
| Tier 2: ring reader, read_segments_into (buffer) | cold | 0.19 ± 0.00 | 5,518 | 3 |
| Tier 2: ring reader, read_segments_into, queue_depth=1 | cold | 0.19 ± 0.00 | 5,518 | 3 |
| Tier 2: ring reader, read_segments_into, queue_depth=8 | cold | 0.19 ± 0.00 | 5,518 | 3 |
| Tier 5: ring reader, file in page cache (warm) | warm | 4.28 ± 0.29 | 253 | 3 |
| Tier 3: .tgate4 compressed container | — | not measured | — | — |
| Tier 4: GPUDirect Storage + layer pipelining | — | not measured | — | — |
<!-- /table -->

**Ring reader before and after** the switch to ≤1 MiB requests with `queue_depth` in flight (cold, alternating
builds; `results_ring_reader_ab_mps.json`):

<!-- table:ring_ab -->
| Build | GB/s (mean ± se) | GB/s (median) | ms | n |
| :--- | ---: | ---: | ---: | :---: |
| f936008 | 2.22 ± 0.35 | 2.33 | 1,225 | 8 |
| current | 5.78 ± 0.41 | 6.23 | 393 | 8 |
<!-- /table -->

---

## 9. C++ engine

`turing-cli` decoding a synthetic 8-layer Llama-layout model, one thread (`results_cpp_engine_mps.json`,
`results_cpp_engine_x86.json`):

<!-- table:cpp -->
| Machine | Build | KV cache | n | tok/s |
| :--- | :---: | :---: | :---: | :---: |
| Apple M4 Pro (NEON) | c547190 | native | 5 | 16.9 ± 0.0 |
| Apple M4 Pro (NEON) | current | native | 5 | 124.6 ± 1.6 |
| Apple M4 Pro (NEON) | current | int8 | 5 | 117.2 ± 3.5 |
| L4 VM CPU (AVX2) | current | native | 5 | 25.3 ± 0.1 |
| L4 VM CPU (AVX2) | current | int8 | 5 | 25.9 ± 0.0 |
<!-- /table -->

---

## 10. Old serving defaults

Before v1.6 `turing serve` pruned 50% of FFN channels and applied a random-basis "recirculation" by default.
Measured on real weights (Apple M4 Pro, fp32; `results_serving_defaults_mps.json`), pruning without retraining is
destructive; both are off now:

<!-- table:serving_defaults -->
| Model | Exact NLL (= HuggingFace) | FFN pruning 50% (old default) | Recirculation (old default) |
| :--- | :---: | :---: | :---: |
| SmolLM2-135M | 2.606 | +15.775 ± 1.731 | +0.000 ± 0.000 |
| SmolLM2-360M | 2.309 | +8.920 ± 0.430 | +0.001 ± 0.001 |
| SmolLM2-1.7B | 1.921 | +6.848 ± 0.215 | +0.001 ± 0.001 |
<!-- /table -->

---

## 11. Removed results

The [claims audit](claims-audit.md) lists every number earlier versions published, with its producing script and
line. They fell into four groups:

* **Constants or formulas printed as results**: speculation-gating throughput, lineage residual norms, a hard-coded
  cold-start baseline, ingest tiers derived as `Tier 2 × 0.25` or hard-coded, a "7.85×" transfer speedup, "75.0%
  VRAM savings" written into a print string.
* **Random tensors or stubs presented as model results**: a 131K-token NIAH over one page of random keys, a
  "5-model pipeline" that loaded no model, throughput of a random 4-layer toy model, MoE hit rates from a synthetic
  Zipf trace.
* **No committed result, or contradicted by the committed one.**
* **Comparisons that were not like for like**: cold against warm caches, Python loops against vectorised kernels,
  baselines that paid kernel start-up in their first timed iteration.
