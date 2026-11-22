# Benchmarks

What Turing Engine was measured doing on real models: low-bit KV-cache and weight quality, decode throughput,
long-context serving, cold start, MoE expert offload, storage reads and the standalone C++ engine. Head-to-head
results against vLLM and llama.cpp are on the [comparison](comparison.md) page.

**Hardware.** GPU numbers are from one NVIDIA L4 (24 GB) on a Google Cloud `g2-standard-16`; Mac numbers are from an
Apple M4 Pro. ± is one standard error; n is given where it matters.

**Sustained load on the L4.** The card caps its power near 72 W: under sustained compute its core clock falls from
2,040 MHz to about 1,100–1,200 MHz within a tenth of a second (the memory clock does not move). One prefill graph
(two 1,024-token prompts, SmolLM2-360M) replays in 33.6 ms after a pause of 50 ms or more, 39.5 ms after 10 ms and
43.0 ms back to back; the eager forward takes 44.7 ms back to back (`results_l4_power_cap_cuda.json`). So a number
from a run with idle gaps (one prompt at a time) is not the number a burst gets, a CUDA graph buys nothing for
sustained GEMM-bound prefill, and engines are compared here under the same sustained load.

**Results and reproduction.** The raw results live in `benchmarks/results/`. Every table on this page is generated
from them by `scripts/make_doc_tables.py` (`--check` fails when a table and its file disagree), and
`benchmarks/README.md` has the commands to reproduce them. Each table names its results file.

---

## 1. Summary

* **Speed against vLLM** (§5, §7, §11, [comparison](comparison.md)): on Qwen2.5-1.5B Turing is level with vLLM in
  fp16 (101% of its throughput) and at 97–99% at 4 bits with its LM head in fp16. With the 4-bit LM head (the
  default) Turing is 1.05–1.27× vLLM at batch 1 to 64. At 16K-token prompts Turing's fp16 is 1.07–1.29× faster per
  request, and vLLM's fp8 cache with layer 0 kept is 1.06–1.09× faster than Turing's int8 at equal quality. With
  prompt lookup Turing is at 100–121% of vLLM's n-gram speculator, with an EAGLE-3 head at 94–99%, and on MoE
  models at 97–117% below the batch size where vLLM's KV budget runs out.
* **Low-bit KV caches keep quality without settings** (§2): int8 is free, int4 costs about +0.02 nats, and both pass
  every pass-key retrieval to 32K. On the same model, vLLM's int8/int4 and default fp8 caches lose 16 to 57 GSM8K
  points unless layer 0 is excluded by hand.
* **4-bit weights** decode 1.5–1.6× faster than fp16 under CUDA graphs (§3); GPTQ costs about 7 GSM8K points
  (57.6% → 50.4%). TEAL activation sparsity on top of GPTQ is not worth it end to end on the L4: 107–112 tok/s at
  batch 1 where plain GPTQ ran at 155 in the same session (167 now), and 4–8 more points lost
  ([comparison](comparison.md)).

---

## 2. KV cache quality and the layers it keeps

**How layers are chosen.** When a pool stores KV in int8 or int4, Turing calibrates once at startup on 4 × 256
random tokens: each layer's K and V take the pool's round trip alone, and a layer whose next-token distribution moves
by more than 0.02 nats (KL divergence) is kept in the model's dtype, at most a quarter of the layers. For int4, key
channels at 16× or more of their head's median magnitude are also divided by a power of two (and the queries
multiplied by it), which leaves q·k unchanged bit for bit.

The rules follow from diagnostic runs on real models (these numbers were not saved as results files):

* On Qwen2.5-1.5B, int4 KV cost +1.93 nats, of which +1.77 came from layer 0's keys alone; every other layer cost
  under 0.005 and quantizing every value cost +0.002. Random-token calibration finds that layer (KL 0.13 against a
  median layer of 0.0006); attention-output error does not (Qwen2.5-0.5B's layer 0 has more error and costs +0.014).
* Key smoothing helps int4 where the model has real outlier channels (Qwen2's bias channels reach 64× the median):
  +0.040 → +0.013 nats on Qwen2.5-1.5B. Applied to SmolLM2's mild channels (at most 8×) it doubled int4's cost, and
  applied to int8 in bf16 it cost 16 GSM8K points (42.2% against 58.2%), so it is int4-only and outlier-only.

**NLL on WikiText-2** (fp16, 16 windows of 2,048 tokens, last 256 scored, paired against the exact model;
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

**Against vLLM's caches** (Qwen2.5-1.5B-Instruct, bf16, vLLM 0.31). GSM8K: first 500 test problems, 8-shot, greedy;
pass-key: 20 needles at 2K–32K; ΔNLL: 8 WikiText-2 windows; time per request from §5. Files:
`results_gsm8k_final_cuda.json`, `results_niah_final_cuda.json`, `results_quality_instruct_bf16_cuda.json`,
`results_quality_vllm_cuda.json`.

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 1.95 |
| Turing int8 | — | 58.2% | 100.0% | +0.003 ± 0.002 | 1.83 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.09 |
| vLLM bf16 (reference) | — | 57.4% | 100.0% | — | 2.51 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.66 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
| vLLM fp8_per_token_head | — | 0.6% | 0.0% | +2.404 ± 0.364 | — |
| vLLM int8_per_token_head | — | 41.4% | 100.0% | +0.023 ± 0.007 | 4.72 |
| vLLM int4_per_token_head | — | 0.6% | 0.0% | +8.127 ± 0.495 | 4.41 |
| vLLM int4_per_token_head | layer 0 (manual) | 47.8% | 100.0% | +0.058 ± 0.010 | 4.30 |
| vLLM turboquant_k8v4 | — | 52.0% | 100.0% | +0.001 ± 0.001 | — |
<!-- /table -->

The vLLM caches that collapse do so silently: the server starts and answers. Skipping layer 0 (vLLM's
`kv_cache_dtype_skip_layers`) repairs fp8 and int4; Turing's calibration makes the same choice without being told.
The tables above cover Qwen2.5-1.5B(-Instruct) and SmolLM2 (360M, 1.7B); the six-model validation below adds
Qwen2.5-7B, Llama 3.2, Gemma 2, Phi-3.5 and Mistral.

**GSM8K exact match** (Qwen2.5-1.5B-Instruct, first 500 test problems, 8-shot, greedy; `results_gsm8k_final_cuda.json`,
`results_gsm8k_diag_cuda.json`, and `results_gsm8k_teal_cuda.json` for the TEAL rows: 200 problems at batch 1,
where TEAL applies):

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

**Six instruct models** (NVIDIA L4, vLLM 0.31, 2026-11-18). The same calibration runs as
[`kv-layer-guard`](https://github.com/intutic/turing/tree/master/packages/kv-layer-guard) on vLLM's fp8 cache and inside Turing's own int8/int4 pools. GSM8K is
8-shot exact match on 500 problems unless noted, bf16 weights; pass-key is 5 depths per length, up to 32K for the
Qwen and Llama models and up to 16K (8K for Gemma 2) for the others. Files: `results_quality_vllm_validation_cuda.json`,
`results_guard_*_cuda.json`, `results_gsm8k_validation_cuda.json`, `results_niah_validation_cuda.json`, and
`results_niah_phi35_longrope_fix_cuda.json` for Phi-3.5's Turing pass-key.

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

* The guard keeps layers only where vLLM's default fp8 cache breaks the model: layer 0 of Qwen2.5-1.5B and layers
  0, 1 and 27 of Qwen2.5-7B (GSM8K 2.2% and 2.4% without it). On the other four it keeps none; fp8 there costs
  at most 1.4 points on three of them, and Phi-3.5 loses 5.2 (next point).
* **Limit:** Phi-3.5-mini loses 5.2 GSM8K points under fp8 (71.8% to 66.6%; each figure has a standard error of
  about 2 points) and the guard does not flag it: its largest per-layer KL is 0.006, under the 0.02 threshold. A loss
  spread thinly over many layers is not what a per-layer check catches.
* Phi-3.5's vLLM column comes from a server started with a 16K maximum length (the evaluation script's default: the
  longest pass-key context + 32; the length is not recorded in that results file). vLLM then uses LongRoPE's long
  frequencies for every request, short ones included. With the maximum length at 4,096 the same evaluation scores
  78.8% (n=500, `results_quality_vllm_phi35_4k_cuda.json`), against 71.8% at 16K and Turing's 79.8% (int8 KV), which
  picks the frequencies per sequence as transformers does. The fp8 comparison is unaffected: both of its runs used 16K.

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

**Where each int4 kernel wins** (microseconds summed over seven layer shapes, timed as CUDA-graph replays;
`results_w4a16_crossover_cuda.json`). The shapes are five of Qwen2.5-1.5B, from 1536 × 1536 to its 1536 × 151,936 LM
head, and two 7B MLP shapes, 4096 × 14,336 and 14,336 × 4096. The Triton kernel splits K across programs so a narrow
layer still fills the GPU, and serves up to `INT4_MAX_ROWS = 64` rows. Marlin (vendored from vLLM, built only with
`TURING_BUILD_MARLIN=1`) is faster at every row count here, and at prefill sizes too
(`results_w4a16_prefill_rows_cuda.json`, Qwen2.5-1.5B's four layer shapes: 3,398 against 4,406 µs for dequantize +
cuBLAS at 2,048 rows, 6,846 against 7,682 at 4,096), so when it is built `QuantLinear` sends it 1–4,096 rows:

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
of zero inputs. Applied to a whole prefill it broke attention-sink tokens (SmolLM2-1.7B: +6.4 nats at 40%, from a
diagnostic run not saved as a results file), so prefill and batched decode stay dense.

---

## 4. Decode throughput against HuggingFace transformers

One WikiText-2 prompt is prefilled, its cache copied into B requests, and 32 batched decode steps are timed through
the serving path; the paged pool takes 85% of free GPU memory. 'pool full' = the pool cannot hold the batch, OOM =
CUDA out of memory, — = not run. Contexts above 8,192 exceed SmolLM2's trained length (speed only).

**SmolLM2-360M**, generated tokens per second (`results_kv_decode_cuda_smollm2_360m.json`):

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
length (> 1 means Quest is faster; `results_quest_threshold_cuda_smollm2_360m.json`):

<!-- table:quest -->
| Batch | 2K | 4K | 8K | 16K | 32K |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | 0.77× | 0.81× | 0.87× | OOM | OOM |
| 8 | 0.64× | 0.89× | 0.88× | — | — |
| 64 | 0.40× | 0.71× | 0.64× | — | — |
<!-- /table -->

Quest is slower at every length measured here, so it is opt-in (`--kv-page-budget`) and, when enabled, starts only
from `kv_select_min_tokens` (8,192).

### Reproduce
```bash
python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M \
  --settings hf_transformers,native_eager,native,int8,int4,int8_quest25 --contexts 2048,8192,32768 --batches 1,8,32,64
```

---

## 5. Long context at equal KV memory

Qwen2.5-1.5B-Instruct, batches of 16 to 128 requests, each a 16K-token prompt and 256 new tokens. vLLM runs at
`gpu_memory_utilization` 0.85 and records the KV memory it took (14.6 GiB); every Turing setting gets the same bytes
(Quest's row ran with less, 12.7 GiB), so capacity differs only by bytes per token. Seconds per request
(`results_longctx_cuda.json`; Turing's native, int8 and int4 rows are from the session of 2026-11-21, vLLM's rows and
the Quest row from the one before):

<!-- table:longctx -->
| Engine, KV cache | KV capacity (tokens) | B=16 | B=32 | B=64 | B=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing native | 547,840 | 2.11 | 2.01 | 1.97 | 1.95 |
| Turing int8 | 993,648 | 2.03 | 1.90 | 1.85 | 1.83 |
| Turing int4 | 1,305,488 | 2.30 | 2.20 | 2.12 | 2.09 |
| Turing int4, Quest 25% | 1,136,256 | 3.78 | 3.66 | 3.48 | 3.40 |
| vLLM auto | 547,856 | 2.26 | 2.27 | 2.43 | 2.51 |
| vLLM fp8 | 1,068,816 | 1.83 | 1.70 | 1.67 | 1.66 |
| vLLM fp8, skip layer 0 | 1,032,096 | 1.85 | 1.74 | 1.75 | 1.73 |
| vLLM int8_per_token_head | 1,062,096 | 4.90 | 4.79 | 4.74 | 4.72 |
| vLLM int4_per_token_head | 2,060,896 | 4.73 | 4.55 | 4.49 | 4.41 |
| vLLM int4_per_token_head, skip layer 0 | 1,876,928 | 4.51 | 4.42 | 4.36 | 4.30 |
<!-- /table -->

* In fp16 Turing is 1.07–1.29× faster per request than vLLM (1.95 against 2.51 s at B=128).
* Turing's int8 cache holds 1.8× the tokens of fp16 and takes 0.94–0.96× its time per request; int4 holds 2.4× and
  takes 1.07–1.09×. A prefill chunk is quantized by the decode step's append kernel, and the cached prefix is
  dequantized by one kernel into pages FlashInfer reads (`triton_kv_decode.py`): one 16K prompt alone prefills at
  11.2K (int8) and 11.1K (int4) tokens/s against 11.3K in fp16. Quest at 25% of pages was measured before these
  changes (3.40 s at B=128).
* vLLM's fp8 cache with layer 0 kept is 1.06–1.09× faster than Turing's int8 at equal quality. vLLM fp8 without a
  skipped layer is fast because it is wrong on this model; quality of each format is in §2.

---

## 6. Cold start

Time from process start to the first generated token on the L4, a fresh process per run. Cold = the model files
dropped from the page cache, with Python and CUDA libraries cached (`results_cold_start_cuda.json`):

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

The 4-bit `.tgate4` directory loads 1.7–2.5× faster from a cold disk. On Qwen2.5-1.5B the first token follows the
load by about 2 s in either format. The SmolLM2-1.7B rows were measured on an older build and not rerun; their 4–10 s
from load to first token may not reflect the current code.

For comparison, on the same Qwen files vLLM takes 51.7 s cold and 34.2 s warm to its first token, and llama.cpp
7.1 s and 1.3 s with its 4-bit GGUF (`results_cold_start_vs_vllm_cuda.json`, `results_cold_start_llamacpp_cuda.json`).

---

## 7. MoE decode and expert offload

### Grouped-GEMM decode

OLMoE-1B-7B-Instruct in bf16 on the L4, 512-token prompts, 256 new tokens, output tokens per second at batch size B.
Before, `SparseMoE` looped over the experts a batch routed to, which needs the expert list on the host each layer and
so ran without CUDA graphs. Now routing, the sort of token-expert pairs by expert, both expert products and the sum
over each token's experts are Triton kernels (`turing/kernels/triton_moe.py`), so decode steps are captured in CUDA
graphs. GSM8K is 8-shot exact match on the first 200 test problems (`results_moe_speed_base_cuda.json`,
`results_moe_speed_cuda.json`, `results_moe_gsm8k_base_cuda.json`, `results_moe_gsm8k_cuda.json`; the GSM8K run of
the grouped kernels predates the routing and sum kernels):

<!-- table:moe_decode -->
| Setting | B=1 | B=8 | B=32 | B=64 | TTFT, B=1 | GSM8K (n=200) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Turing, per-expert loop, no CUDA graph | 16 | 38 | 97 | 114 | 0.327 s | 33.5% ± 3.3 |
| Turing, grouped GEMM + CUDA graphs | 94 | 204 | 475 | 732 | 0.070 s | 35.0% ± 3.4 |
| vLLM | 94 | 205 | 490 | 582 | 0.078 s | — |
<!-- /table -->

The kernels match the per-expert loop at 1e-4 in fp32 (`tests/test_triton_moe.py`). Decode is 4.9–6.4× faster than
the loop. Against vLLM Turing is level at batch 1 and 8 (100.4% and 99.7%), 3% behind at batch 32 and 26% ahead at
batch 64, where vLLM's KV budget (4.6 GiB, 37,776 tokens) holds 49 of the 64 requests at a time and Turing's (6.8
GiB) all of them. At batch 1 the experts are read cold at 247 of the L4's 262 GB/s
(`results_moe_kernel_cold_cuda.json`, 16 layers a call so that no weight is in the GPU's cache).

### 4-bit experts

`--quantization` and `turing convert` pack an MoE layer's experts in 4 bits too. Qwen1.5-MoE-A2.7B-Chat (60 experts,
top-4, a shared expert), converted with GPTQ, against vLLM on Qwen's own GPTQ-Int4 checkpoint; 512-token prompts, 256
new tokens, every engine at its default KV budget (`results_moe_w4_speed_cuda.json`,
`results_moe_w4_speed_triton_cuda.json`, `results_moe_w4_gsm8k_cuda.json`, `results_moe_w4_quality_vllm_cuda.json`):

<!-- table:moe_w4 -->
| Setting | B=1 | B=8 | B=32 | B=64 | TTFT, B=64 | GSM8K (n=500) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 114 | 350 | 621 | 817 | 2.09 s | 40.6% ± 2.2 |
| Turing GPTQ (4-bit LM head, the default) | 133 | 367 | 630 | 819 | 1.97 s | 45.8% ± 2.2 |
| Turing GPTQ, fp16 LM head | 107 | 338 | 604 | 797 | 1.98 s | 46.0% ± 2.2 |
| Turing GPTQ, 4-bit LM head, Triton kernels (no Marlin) | 119 | 272 | 458 | 522 | 4.71 s | — |
<!-- /table -->

* With the Marlin MoE kernel (vendored from vLLM, built by `TURING_BUILD_MARLIN=1`) and the default 4-bit LM head,
  Turing is 1.17 / 1.05 / 1.01 / 1.00× vLLM at batch 1 / 8 / 32 / 64, and its first token comes sooner at every
  batch size. With the LM head in fp16, as in Qwen's checkpoint, Turing is at 94–98% of vLLM.
* A build without Marlin runs the Triton int4 kernels. They read whole quantization groups per step, which doubles
  the rate at which they stream cold weights, but a prefill is bound by unpacking the codes: that row's first token
  at batch 64 takes 4.7 s. Its numbers predate the routing, alignment and sum kernels.
* OLMoE-1B-7B, packed by round-to-nearest at load, decodes at 267 / 577 / 1,105 / 1,424 tok/s at batch 1 / 8 / 32 /
  64 (229 / 550 / 1,080 / 1,403 with the LM head in fp16), 1.9–2.8× its bf16 rate. Its WikiText-2 NLL moves by
  +0.059 ± 0.012 nats with round-to-nearest and +0.019 ± 0.008 with GPTQ (n = 8 windows;
  `results_moe_w4_quality_cuda.json`).

### Expert offload

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

### Expert offload in `turing serve`

`--moe-expert-slots N` keeps `N` experts per layer on the GPU and the rest in pinned host memory, copied into the
least recently used slot when a token is routed to them ([serving §5a](serving.md#5a-moe-expert-offload)). 512-token
prompts, 256 new tokens, output tokens per second (`results_moe_offload_serve_cuda.json`):

<!-- table:moe_offload_serve -->
| Model | Experts per layer on the GPU | B=1 | B=2 | B=8 | Slot hit rate (B=1) | Peak GPU GiB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| OLMoE-1B-7B-0924-Instruct | all | 89.4 | 110.1 | 199.7 | — | 18.7 |
| OLMoE-1B-7B-0924-Instruct | 48 of 64 | 25.5 | 40.5 | 53.8 | 88.6% | 17.3 |
| OLMoE-1B-7B-0924-Instruct | 32 of 64 | 14.4 | 23.2 | 24.1 | 73.1% | 16.1 |
<!-- /table -->

OLMoE routes every token to 8 of 64 experts, so with 48 or 32 slots most steps copy experts in, and throughput falls
to 29% and 16% of the all-on-GPU rate at batch 1. The outputs do not depend on the slots: GSM8K with 32 slots is
40.5% ± 3.5 against 37.5% ± 3.4 with every expert on the GPU (n = 200, `results_moe_offload_gsm8k_cuda.json`).

---

## 8. Storage ingestion

`scripts/benchmark_ingest_ladder_all_tiers.py` reads a file of tensor-sized segments cold: 2 GiB on the Mac, 1 GiB on
the L4 machine. The file is written through `F_NOCACHE` on macOS or dropped with `posix_fadvise` on Linux, and
`mincore` confirms nothing is cached.

**Apple M4 Pro SSD** (the machine was shared, so the spread is wide; `results_ingest_ladder.json`):

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

**L4 machine boot disk**, a network-attached persistent disk (`results_ingest_ladder_linux_l4.json`):

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

**Ring reader request size.** The first row is the build before reads were split into requests of at most 1 MiB
with `queue_depth` in flight; the second is the current reader (cold, builds alternated;
`results_ring_reader_ab_mps.json`):

<!-- table:ring_ab -->
| Build | GB/s (mean ± se) | GB/s (median) | ms | n |
| :--- | ---: | ---: | ---: | :---: |
| f936008 | 2.22 ± 0.35 | 2.33 | 1,225 | 8 |
| current | 5.78 ± 0.41 | 6.23 | 393 | 8 |
<!-- /table -->

---

## 9. C++ engine

`turing-cli` decoding a synthetic 8-layer Llama-layout model on one thread; the first row is an older build kept as
a baseline (`results_cpp_engine_mps.json`, `results_cpp_engine_x86.json`):

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

## 10. FFN channel pruning

`--sparsity` prunes FFN channels without retraining, and it is destructive: at 50% next-token NLL rises by +6.8 to
+15.8 nats on SmolLM2. The table also measures recirculation (`enable_recirculation`), a random-basis transform that
moves NLL by +0.000 to +0.001. Both were on by default before v1.6 and are off now (Apple M4 Pro, fp32;
`results_serving_defaults_mps.json`):

<!-- table:serving_defaults -->
| Model | Exact NLL (= HuggingFace) | FFN pruning 50% (old default) | Recirculation (old default) |
| :--- | :---: | :---: | :---: |
| SmolLM2-135M | 2.606 | +15.775 ± 1.731 | +0.000 ± 0.000 |
| SmolLM2-360M | 2.309 | +8.920 ± 0.430 | +0.001 ± 0.001 |
| SmolLM2-1.7B | 1.921 | +6.848 ± 0.215 | +0.001 ± 0.001 |
<!-- /table -->

---

## 11. Speculative decoding

**Prompt lookup** (`--draft-tokens`, 8 by default). Qwen2.5-1.5B-Instruct in fp16, 60 greedy requests per workload
of up to 256 new tokens, sent in waves of 1, 2 or 8 (the next wave starts when the last has finished). Output tokens
per second; vLLM runs its n-gram speculator with the same number of drafts (`results_speculation_cuda.json`):

<!-- table:spec_lookup -->
| Workload | Requests in flight | Turing, no drafts | Turing, 8 drafts | Drafts accepted | Tokens per target pass | vLLM, no drafts | vLLM, 8 drafts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| code | 1 | 76 | 435 | 74.0% | 6.62 | — | 411 |
| code | 2 | 136 | 708 | 73.9% | 6.63 | — | 654 |
| code | 8 | 439 | 1,626 | 73.8% | 6.63 | 459 | 1,629 |
| rewrite | 1 | 74 | 197 | 27.1% | 2.86 | — | 180 |
| rewrite | 2 | 133 | 326 | 27.1% | 2.86 | — | 292 |
| rewrite | 8 | 408 | 837 | 27.1% | 2.86 | 409 | 756 |
| chat | 1 | 76 | 120 | 14.2% | 1.67 | — | 103 |
| chat | 2 | 140 | 212 | 14.0% | 1.65 | — | 175 |
| chat | 8 | 511 | 647 | 14.1% | 1.65 | 507 | 546 |
<!-- /table -->

* Drafted tokens are verified as extra rows of the batched decode step, so a request's drafts cost no forward of
  their own. Over the nine cells, 8 drafts give 2.5× the undrafted rate (geometric mean; 2.2× with 4 and 2.8× with
  15, which loses 6% to 8 on chat at 8 requests), and never less than 1.27×.
* Greedy output with drafts equals the undrafted output in 521 of 540 requests. The others first differ at a token
  whose two candidates were the model's top two and at most 0.03 apart in fp16, as two batch sizes can differ
  (`results_spec_divergence_cuda.json`).

**EAGLE-3 draft head** (`--draft-head`). Qwen3-8B in bf16 with the published `RedHatAI/Qwen3-8B-speculator.eagle3`
head, 24 requests per workload in waves of 1, 2 or 4, both engines with chains of up to 5 drafts
(`results_eagle3_cuda.json`):

<!-- table:eagle3 -->
| Workload | Requests in flight | Turing, no drafts | Turing, 5 drafts | Drafts accepted | Tokens per target pass | vLLM, no drafts | vLLM, 5 drafts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| code | 1 | 15.7 | 48.0 | 60.6% | 3.96 | 16.5 | 50.8 |
| code | 2 | 27.5 | 83.8 | 60.5% | 3.96 | 27.0 | 88.8 |
| code | 4 | — | 141.1 | 60.4% | 3.95 | — | 144.5 |
| rewrite | 1 | 16.3 | 50.1 | 65.1% | 4.37 | 16.5 | 51.1 |
| rewrite | 2 | 28.2 | 88.3 | 65.4% | 4.39 | 28.6 | 89.4 |
| rewrite | 4 | — | 146.5 | 65.1% | 4.37 | — | 148.6 |
| chat | 1 | 16.5 | 33.5 | 37.4% | 2.64 | 16.7 | 33.8 |
| chat | 2 | 30.4 | 60.6 | 37.4% | 2.64 | 30.8 | 62.3 |
| chat | 4 | — | 105.1 | 37.5% | 2.64 | — | 109.3 |
<!-- /table -->

* The head runs once per draft position for every drafting request, from a paged pool of its own and under CUDA
  graphs. At equal chain length Turing is at 94–99% of vLLM.
* A request's chain is as long as its drafts are likely to be reached, so a higher limit costs little where drafts
  fail: with a limit of 7 Turing reaches 53.5 / 94.0 / 157.0 tok/s on rewrite and 49.8 / 85.5 / 141.6 on code at
  1 / 2 / 4 requests. Chat is fastest with 3 (114.7 tok/s at 4 requests). vLLM was measured at 3 and 5.
* 11 of 16 chat requests differ from undrafted greedy output, each first at a token whose two candidates were the
  model's top two and at most 0.25 apart (bf16; `results_spec_divergence_eagle3_cuda.json`).
