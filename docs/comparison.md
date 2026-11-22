# Comparison with vLLM, SGLang, TensorRT-LLM, llama.cpp and Ollama

**Setup.** One NVIDIA L4 (24 GB) running Turing Engine, vLLM 0.31 and llama.cpp (CUDA build) on the same models,
prompts and GPU. SGLang, TensorRT-LLM and Ollama were not run (Ollama serves through llama.cpp). The tables are
generated from the results files in `benchmarks/results/` by `scripts/make_doc_tables.py`; more measurements are on
the [benchmarks](benchmarks.md) page.

---

## Where Turing Engine stands

* **Ahead of vLLM in throughput in fp16 and with a 4-bit LM head, behind with a 1-byte KV cache at 16K tokens.**
  In fp16 Turing is at 100–103% of vLLM's throughput on Qwen2.5-1.5B and 101–111% on SmolLM2. At 4 bits with its
  LM head in 4 bits too (the default) Turing is 1.33 / 1.22 / 1.07× vLLM and 1.27 / 1.73 / 1.54× llama.cpp at
  batch 1 / 8 / 64; with the head kept in fp16 it is at 97–101% of vLLM. At 16K-token prompts in fp16 Turing is
  1.04–1.30× faster per request; with one byte per value vLLM's fp8 cache is 1.03–1.06× faster than Turing's fp8
  and 1.05–1.06× faster than its int8 when both take 2,048 tokens a step, and 1.05–1.09× at its own 8,192.
* **First token.** A prompt arriving alone gets its first token sooner than on vLLM on SmolLM2-360M (18 against
  29 ms) and Qwen2.5-1.5B (27 against 32 ms) and as soon on SmolLM2-1.7B (65 against 65–72 ms). In a burst the
  mean first token follows the tokens an engine takes per step ("Step budgets" below): with both engines at 8,192
  Turing's comes 8–41% sooner; with both at 2,048 it comes from 15% sooner to 7% later at batch 8 to 64.
  llama.cpp starts fastest from a cold process.
* **What is behind at equal budgets is 2–7%.** vLLM 0.31 schedules its next step while the current one runs
  (`async_scheduling`, its default), so the host's work between steps costs it no GPU time; Turing's engine
  overlaps only its decode steps.
* **Differences of a few percent are within the spread between runs.** One prompt-lookup cell measured four times
  back to back gave 448.8–449.6 tok/s without drafts and 1,573–1,585 with 8, and 439 and 1,626 an hour earlier
  (session of 2026-11-21; `results_speculation_repeat_cuda.json`). A Spot VM is also restarted now and then and
  may come back on another physical GPU, so the rows compared here were measured in one boot: the short-prompt,
  512-token, 16K and 4-bit MoE tables in the run of 2026-11-22 after 10:48 UTC, both engines.
* **Comparable quality.** Turing's bf16 matches vLLM's on GSM8K (57.2% and 57.4%), and its GPTQ matches vLLM's GPTQ
  checkpoint (50.4% and 50.2%; llama.cpp's Q4_K_M 52.4%, all ± 2.2%).
* **Low-bit KV that stays correct by default.** Turing picks the layers to keep in full precision at startup; vLLM's
  default fp8 and its per-token int4/fp8 caches break this model unless layer 0 is excluded by hand, and its per-token
  int8 cache loses 16 GSM8K points.
* **Not covered:** multi-GPU, SGLang, TensorRT-LLM, models larger than Qwen2.5-7B (7.6B), and model families other
  than the six in the KV-guard validation ([benchmarks §2](benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps)).

---

## Speed

**fp16, 1,024-token prompts, 256 new tokens, all requests submitted at once.** Output tokens per second, prefill
included (`results_e2e_generation_cuda.json`):

<!-- table:vs_short -->
| Model | Engine | B=1 | B=8 | B=32 | B=64 | TTFT, B=1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SmolLM2-1.7B | vLLM | 65 | 327 | 570 | 639 | 0.072 s |
| SmolLM2-1.7B | Turing | 66 | 337 | 609 | 710 | 0.065 s |
| SmolLM2-360M | vLLM | 235 | 1,287 | 2,424 | 2,789 | 0.029 s |
| SmolLM2-360M | Turing | 251 | 1,375 | 2,567 | 2,965 | 0.018 s |
<!-- /table -->

**Step budgets.** vLLM's offline `LLM` class takes 8,192 tokens per scheduler step on an L4, decode rows included
(its API server takes 2,048); Turing's engine takes 2,048 tokens per forward, the rows of the requests already
decoding included (a step replayed from a CUDA graph takes them on top of its whole prompts). Every other table on
this page has each engine at its own setting. Both engines at both settings
(Turing: `--prefill-chunk`, spent over forwards of 2,048 tokens; vLLM: `max_num_batched_tokens`), at the largest
batch; the 16K rows are from `results_longctx_cuda.json`:

<!-- table:step_budget -->
| Workload | Tokens per step | Engine | Requests at once | Output tok/s | Mean first token |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SmolLM2-360M, 1,024-token prompts | 2,048 | vLLM | 64 | 2,732 | 0.77 s |
| SmolLM2-360M, 1,024-token prompts | 2,048 | Turing | 64 | 2,965 | 0.77 s |
| SmolLM2-360M, 1,024-token prompts | 8,192 | vLLM | 64 | 2,789 | 0.75 s |
| SmolLM2-360M, 1,024-token prompts | 8,192 | Turing | 64 | 3,021 | 0.69 s |
| SmolLM2-1.7B, 1,024-token prompts | 2,048 | vLLM | 64 | 669 | 2.66 s |
| SmolLM2-1.7B, 1,024-token prompts | 2,048 | Turing | 64 | 710 | 2.75 s |
| SmolLM2-1.7B, 1,024-token prompts | 8,192 | vLLM | 64 | 639 | 2.84 s |
| SmolLM2-1.7B, 1,024-token prompts | 8,192 | Turing | 64 | 712 | 2.53 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | vLLM fp8 | 128 | 151 | 103.57 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | Turing int8 | 128 | 144 | 108.60 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | Turing fp8 | 128 | 147 | 106.76 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | vLLM fp8 | 128 | 154 | 97.55 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | Turing int8 | 128 | 143 | 105.59 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | Turing fp8 | 128 | 144 | 104.50 s |
<!-- /table -->

* On the short prompts Turing has the higher throughput at either budget (101–112% of vLLM over batch 1 to 64).
* A larger budget moves a burst's first tokens earlier and makes the requests already decoding wait for it. With
  both engines at 8,192 Turing's mean first token at batch 8 / 32 / 64 is 26 / 18 / 8% sooner on SmolLM2-360M and
  34 / 16 / 11% sooner on SmolLM2-1.7B; with both at 2,048 it is 15% sooner / 2.5% later / level on SmolLM2-360M
  and 2 / 5 / 3% later on SmolLM2-1.7B.
* At 16K tokens vLLM's fp8 cache is ahead at both budgets (3% and 7% in throughput at batch 128 against Turing's
  fp8, 5% and 7% against its int8).

**Batch sizes between the powers of two** (Qwen2.5-1.5B-Instruct in fp16, 512-token prompts, 256 new tokens;
`results_batch_sizes_cuda.json`). Turing's decode graphs come in 12 sizes up to 128 rows, so 12, 24 and 48 requests
are not padded to the next power of two:

<!-- table:batch_sizes -->
| Engine | B=12 | B=24 | B=48 |
| :--- | :---: | :---: | :---: |
| vLLM | 755 | 1,256 | 1,957 |
| Turing | 767 | 1,252 | 1,923 |
<!-- /table -->

**4 bits, Qwen2.5-1.5B-Instruct, 512-token prompts, 256 new tokens** (`results_4bit_speed_cuda.json`; GSM8K from
`results_4bit_quality_*_cuda.json`, all three engines scored by Turing's evaluator):

<!-- table:vs_4bit -->
| Setting | Batch 1, tok/s | Batch 8, tok/s | Batch 64, tok/s | GSM8K (n=500) |
| :--- | :---: | :---: | :---: | :---: |
| llama.cpp Q4_K_M | 182 | 782 | 2,176 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 173 | 1,109 | 3,150 | 50.2% ± 2.2% |
| vLLM AWQ (Qwen's checkpoint) | 171 | 1,102 | 3,131 | — |
| Turing GPTQ, fp16 LM head (`--no-quantize-lm-head`) | 174 | 1,092 | 3,064 | 50.4% ± 2.2% |
| Turing GPTQ (4-bit LM head, the default) | 231 | 1,356 | 3,359 | 51.8% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 25% | 107 | 953 | 2,600 | 46.2% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 40% | 112 | 960 | 2,614 | 42.0% ± 2.2% |
| vLLM fp16 (reference) | 75 | 538 | 2,236 | — |
| Turing fp16 (reference) | 77 | 540 | 2,246 | — |
<!-- /table -->

**16K-token prompts at equal KV memory** (Qwen2.5-1.5B-Instruct; seconds per request and each engine's KV capacity in
tokens; 14.6 GiB of KV, 12.7 GiB for the Quest row; `results_longctx_cuda.json`). Read it with the quality table
below: vLLM fp8 without a skipped layer is fast because it is wrong on this model. The rows are from the run of
2026-11-22 (one boot, both engines), except Turing's Quest row and vLLM's per-token caches, from 2026-11-17.

<!-- table:longctx -->
| Engine, KV cache | KV capacity (tokens) | B=16 | B=32 | B=64 | B=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing int8 | 969,264 | 1.96 | 1.86 | 1.80 | 1.77 |
| Turing fp8 | 997,536 | 1.95 | 1.84 | 1.78 | 1.75 |
| vLLM fp8 | 1,069,392 | 1.82 | 1.70 | 1.67 | 1.66 |
| vLLM fp8, skip layer 0 | 1,032,640 | 1.79 | 1.72 | 1.73 | 1.72 |
| Turing native | 534,384 | 2.14 | 2.02 | 1.96 | 1.93 |
| vLLM auto | 547,856 | 2.22 | 2.30 | 2.44 | 2.51 |
| Turing int4 | 1,273,456 | 2.28 | 2.16 | 2.12 | 2.10 |
| Turing int4, Quest 25% | 1,136,256 | 3.78 | 3.66 | 3.48 | 3.40 |
| vLLM int8_per_token_head | 1,062,096 | 4.90 | 4.79 | 4.74 | 4.72 |
| vLLM int4_per_token_head | 2,060,896 | 4.73 | 4.55 | 4.49 | 4.41 |
| vLLM int4_per_token_head, skip layer 0 | 1,876,928 | 4.51 | 4.42 | 4.36 | 4.30 |
<!-- /table -->

## Low-bit KV quality

Qwen2.5-1.5B-Instruct in bf16, the same GSM8K, pass-key and WikiText-2 checks for both engines
(`results_quality_vllm_cuda.json`, `results_gsm8k_final_cuda.json`, `results_niah_final_cuda.json`,
`results_quality_instruct_bf16_cuda.json`):

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 1.93 |
| Turing int8 | automatic | 55.2% | 100.0% | +0.001 ± 0.001 | 1.77 |
| Turing fp8 | automatic | 56.4% | 100.0% | +0.011 ± 0.005 | 1.75 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.10 |
| vLLM bf16 (reference) | — | 57.4% | 100.0% | — | 2.51 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.66 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.72 |
| vLLM fp8_per_token_head | — | 0.6% | 0.0% | +2.404 ± 0.364 | — |
| vLLM int8_per_token_head | — | 41.4% | 100.0% | +0.023 ± 0.007 | 4.72 |
| vLLM int4_per_token_head | — | 0.6% | 0.0% | +8.127 ± 0.495 | 4.41 |
| vLLM int4_per_token_head | layer 0 (manual) | 47.8% | 100.0% | +0.058 ± 0.010 | 4.30 |
| vLLM turboquant_k8v4 | — | 52.0% | 100.0% | +0.001 ± 0.001 | — |
<!-- /table -->

---

## Feature check

Each cell for another engine was checked against that project's official documentation on 2026-11-16 (links below
the table). "not checked" means the documentation page consulted did not settle it; it is not a claim that the
feature is missing. Turing cells describe this release.

| Feature | Turing Engine | vLLM | SGLang | TensorRT-LLM | llama.cpp server | Ollama |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| OpenAI chat / completions API | ✅ | ✅ [1] | ✅ [6] | ✅ `trtllm-serve` [9] | ✅ [10] | ✅ subset [12] |
| Anthropic Messages API (`/v1/messages`) | ✅ | ✅ [1] | not checked | not listed [9] | ✅ [10] | ✅ subset [13] |
| Token render endpoints (`/v1/chat/completions/render`) | ✅ | ✅ [1] | not checked | ✅ prototype [9] | not checked | not checked |
| JSON-schema constrained decoding | ✅ llguidance | ✅ xgrammar, guidance [2] | ✅ XGrammar, Outlines, llguidance [7] | not checked | ✅ grammars, JSON schema [10] | not checked |
| Several LoRA adapters, chosen per request | ✅ | ✅ by `model` name [3] | ✅ same batch, `base:adapter` [8] | not checked | ✅ per-request `lora` list [10] | not checked |
| n-gram / prompt-lookup speculative decoding | ✅ | ✅ [4] | not checked | not checked | speculative decoding ✅ [10] | not checked |
| Loads llama.cpp GGUF files | ✅ (no OPT) | ✅ "highly experimental" [5] | not checked | not checked | ✅ native format | ✅ import [11] |
| Continuous batching | ✅ | ✅ [14] | not checked | ✅ in-flight batching [16] | ✅ [10] | not checked |
| Multi-GPU parallelism | ⚠️ TP/PP tested with CPU processes only | ✅ tensor, pipeline, … [14] | ✅ [15] | ✅ tensor, pipeline, expert [16] | not checked | not checked |

Sources:
[1] [vLLM online serving](https://docs.vllm.ai/en/latest/serving/online_serving/) ·
[2] [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/) ·
[3] [vLLM LoRA](https://docs.vllm.ai/en/latest/features/lora/) ·
[4] [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/) ·
[5] [vLLM GGUF](https://docs.vllm.ai/en/latest/features/quantization/gguf.html) ·
[6] [SGLang OpenAI APIs](https://docs.sglang.io/basic_usage/openai_api_completions.html) ·
[7] [SGLang structured outputs](https://docs.sglang.io/advanced_features/structured_outputs.html) ·
[8] [SGLang LoRA](https://docs.sglang.io/advanced_features/lora.html) ·
[9] [trtllm-serve](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve/trtllm-serve.html) ·
[10] [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) ·
[11] [Ollama import](https://docs.ollama.com/import) ·
[12] [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) ·
[13] [Ollama Anthropic compatibility](https://docs.ollama.com/api/anthropic-compatibility) ·
[14] [vLLM documentation home](https://docs.vllm.ai/en/latest/) ·
[15] [SGLang documentation home](https://docs.sglang.io/) ·
[16] [TensorRT-LLM overview](https://nvidia.github.io/TensorRT-LLM/overview.html).

---

## Choosing

| If you need… | Use |
| :--- | :--- |
| Production serving: multi-GPU, models larger than 8B, a long track record | vLLM, SGLang or TensorRT-LLM (only vLLM was run here, on one GPU; SGLang and TensorRT-LLM were not run) |
| A C/C++ server with GGUF as its native format, or the fastest cold start measured here | llama.cpp |
| A desktop runtime that imports GGUF and Safetensors models [11] | Ollama |
| A low-bit KV cache that protects the layers it would break, HuggingFace-parity-tested models, and GSM8K, pass-key and NLL evals for trying KV-cache and quantization changes on real models | Turing Engine |
