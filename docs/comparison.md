# Comparison with vLLM, SGLang, TensorRT-LLM, llama.cpp and Ollama

**Setup.** One NVIDIA L4 (24 GB) running Turing Engine, vLLM 0.31 and llama.cpp (CUDA build) on the same models,
prompts and GPU. SGLang, TensorRT-LLM and Ollama were not run (Ollama serves through llama.cpp). The tables are
generated from the results files in `benchmarks/results/` by `scripts/make_doc_tables.py`; more measurements are on
the [benchmarks](benchmarks.md) page.

---

## Where Turing Engine stands

* **At or above vLLM's throughput in fp16 and at 4 bits; level with an fp8 KV cache at 16K tokens.**
  In fp16 Turing is at 101–103% of vLLM's throughput on Qwen2.5-1.5B (the median of three runs of each engine,
  taken in turn; the same at each engine's own step budget and with both at either one) and 101–114% on SmolLM2.
  At 4 bits with its LM head in 4 bits too (the default) Turing is 1.34 / 1.26 / 1.09× vLLM and 1.25 / 1.81 /
  1.54× llama.cpp at batch 1 / 8 / 64 (the three engines in turn within one boot); with the head kept in fp16 it
  is at 100–102% of vLLM. At 16K-token prompts in fp16 Turing is 1.06–1.40× faster per request; with one byte per
  value and both engines at the same step budget, Turing's fp8 makes 99.0–101.3% of vLLM fp8's output a second
  and its int8 97.0–99.0% (the median of 3 runs each). vLLM's fp8 keeps no layer in fp16, which costs this model
  its output ("Low-bit KV quality" below) but holds 3.6% more tokens; with layer 0 kept in fp16 as Turing keeps it, Turing's fp8 is
  at 103.5–106.2% and its int8 at 100.7–103.8%. With each engine at its own step budget (Turing 2,048, vLLM 8,192)
  Turing's fp8 takes 0.98–1.02× vLLM's time per request.
* **First token.** A prompt arriving alone gets its first token sooner than on vLLM: 18 against 28–29 ms on
  SmolLM2-360M, 27 against 32 ms on Qwen2.5-1.5B and 68 against 71–73 ms on SmolLM2-1.7B. In a burst the mean
  first token follows the tokens an engine takes per step ("Step budgets" below): with both engines at 8,192
  Turing's comes 10–27% sooner on SmolLM2 and 6–12% sooner on Qwen2.5-1.5B at batch 8 to 64; with both at 2,048
  it comes 5–12% sooner on SmolLM2-360M, 4% sooner on Qwen2.5-1.5B and from 3% later to 1% sooner on
  SmolLM2-1.7B. llama.cpp starts fastest from a cold process.
* **What is behind is at most 3% at equal step budgets.** The 1-byte KV caches at 16K tokens: Turing's int8 makes
  97.0–99.0% of vLLM fp8's output a second (an int8 page holds 4 bytes more per head and token, each token's scale
  and zero, and the decode kernel takes 5% longer over it), and its fp8 99.0% and 99.9% at 16 requests (within
  the 1.5% spread between runs of one engine) and 99.7% at 64 requests and 8,192 tokens a step, where vLLM's
  cache, with no layer in fp16, runs all 64 requests at once and Turing's 62. Qwen1.5-MoE in 4 bits with its LM
  head in fp16 at batch 8 to 64 is 0.4–0.9% behind (99.1–99.6%; 0.5–1.7% in five runs of 2026-11-23,
  `results_moe_w4_speed_repeat_cuda.json`), with first tokens 6–20% sooner: Turing's 4-bit weights carry a zero
  point per group, and Marlin's kernel for them takes 5–6% longer per routed-expert call than the one vLLM runs
  Qwen's symmetric GPTQ checkpoint through (in the 4-bit table below vLLM's own AWQ rows, with zero points, are
  1.2% under its GPTQ rows). Converted with `turing convert --symmetric` the model takes that kernel and is at
  100.0–102.6% (GSM8K 44.4% in bf16 and 42.8% in fp16 against 46.0% with zero points, each ± 2.2%; vLLM 40.6%).
  An EAGLE-3 head is at 101.0–102.4% in 13 of 15 cells; code with chains of 7 at 4 requests is 0.2–0.7% behind in
  two measurements of one boot, and chat at 4 requests 1.7% behind vLLM's fastest of four processes and 0.2% ahead
  of their mean (vLLM's outputs, and with them its speed, change between processes; Turing's do not). The mean first token of a burst of 8 on SmolLM2-1.7B with both engines at 2,048 tokens
  a step is within 3% either way from boot to boot. vLLM 0.31 schedules its next step while the current one runs
  (`async_scheduling`, its default), so the host's work between steps costs it no GPU time; Turing's engine
  launches its decode steps, its steps that prefill, a draft head's verify step and the cycle after it (with two or
  more requests drafting the whole cycle, for a lone request the head's next chain) ahead in the same way.
* **Differences of a few percent are within the spread between runs.** One prompt-lookup cell measured four times
  back to back gave 448.8–449.6 tok/s without drafts and 1,573–1,585 with 8, and 439 and 1,626 an hour earlier
  (session of 2026-11-21; `results_speculation_repeat_cuda.json`). A Spot VM is also restarted now and then and
  may come back on another physical GPU, so each comparison of the two engines here is within one boot: the
  512-token and short-prompt rows in blocks of the run of 2026-11-23 and 24, the 16K rows, the 4-bit MoE table's
  fp16-head rows and the EAGLE-3 table in blocks of the run of 2026-11-25 (a block holds the rows that are
  compared with each other, and every row carries its block and boot), with earlier repeats in the two
  `*_repeat_cuda.json` files. Both engines' first batch of a size is about 2% slower than the next, so the 16K rows
  are the median of 3 runs of each (`results_longctx_first_batch_cuda.json`).
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
| SmolLM2-1.7B | vLLM | 65 | 325 | 558 | 611 | 0.073 s |
| SmolLM2-1.7B | Turing | 66 | 336 | 602 | 700 | 0.068 s |
| SmolLM2-360M | vLLM | 231 | 1,268 | 2,402 | 2,747 | 0.028 s |
| SmolLM2-360M | Turing | 249 | 1,372 | 2,573 | 2,978 | 0.018 s |
<!-- /table -->

**Step budgets.** vLLM's offline `LLM` class takes 8,192 tokens per scheduler step on an L4, decode rows included
(its API server takes 2,048); Turing's engine takes 2,048 prompt tokens per forward, or 2,040 where the model's
forward measured cheaper per token at that row count, with the rows of the requests already decoding on top or
included, whichever measured faster for the model. Every other table on
this page has each engine at its own setting. Both engines at both settings
(Turing: `--prefill-chunk`, spent over forwards of 2,048 tokens; vLLM: `max_num_batched_tokens`), at the largest
batch; the 16K rows are from `results_longctx_cuda.json`:

<!-- table:step_budget -->
| Workload | Tokens per step | Engine | Requests at once | Output tok/s | Mean first token |
| :--- | :---: | :---: | :---: | :---: | :---: |
| SmolLM2-360M, 1,024-token prompts | 2,048 | vLLM | 64 | 2,700 | 0.79 s |
| SmolLM2-360M, 1,024-token prompts | 2,048 | Turing | 64 | 2,978 | 0.75 s |
| SmolLM2-360M, 1,024-token prompts | 8,192 | vLLM | 64 | 2,747 | 0.78 s |
| SmolLM2-360M, 1,024-token prompts | 8,192 | Turing | 64 | 3,005 | 0.70 s |
| SmolLM2-1.7B, 1,024-token prompts | 2,048 | vLLM | 64 | 633 | 2.95 s |
| SmolLM2-1.7B, 1,024-token prompts | 2,048 | Turing | 64 | 700 | 2.92 s |
| SmolLM2-1.7B, 1,024-token prompts | 8,192 | vLLM | 64 | 611 | 3.14 s |
| SmolLM2-1.7B, 1,024-token prompts | 8,192 | Turing | 64 | 699 | 2.76 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | vLLM fp8 | 128 | 150 | 104.53 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | Turing int8 | 128 | 148 | 105.60 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 2,048 | Turing fp8 | 128 | 152 | 103.17 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | vLLM fp8 | 128 | 148 | 101.87 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | Turing int8 | 128 | 145 | 104.45 s |
| Qwen2.5-1.5B, 16K-token prompts, 1-byte KV | 8,192 | Turing fp8 | 128 | 150 | 100.80 s |
<!-- /table -->

* On the short prompts Turing has the higher throughput at either budget (101–114% of vLLM over batch 1 to 64).
* A larger budget moves a burst's first tokens earlier and makes the requests already decoding wait for it. With
  both engines at 8,192 Turing's mean first token at batch 8 / 32 / 64 is 25 / 14 / 10% sooner on SmolLM2-360M and
  27 / 16 / 12% sooner on SmolLM2-1.7B; with both at 2,048 it is 12 / 5 / 5% sooner on SmolLM2-360M and, on
  SmolLM2-1.7B, 3% and 2% later at batch 8 and 32 and 1% sooner at 64. There the two engines run the same schedule
  (two prompts in the first forward, then 2,048 rows a forward with the decode rows among them), and from boot to
  boot the difference at batch 8 went from 2% sooner (the boot before the table's, `rows_of_the_boot_before`) to
  the table's 3% later to 1–2% sooner (five bursts of each engine, twice: 0.379–0.384 s against 0.384–0.394,
  `results_burst_first_token_cuda.json`).
* At 16K tokens Turing's fp8 cache is level with vLLM's in throughput at batch 128 when both take 2,048 tokens a
  step (155 against 154 tok/s) and 1.4% behind at 8,192 (154 against 156); its int8 cache is 1.5% and 2.8% behind.

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
| llama.cpp Q4_K_M | 183 | 764 | 2,171 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 171 | 1,096 | 3,082 | 50.2% ± 2.2% |
| vLLM AWQ (Qwen's checkpoint) | 169 | 1,081 | 3,046 | — |
| Turing GPTQ, fp16 LM head (`--no-quantize-lm-head`) | 174 | 1,117 | 3,092 | 50.4% ± 2.2% |
| Turing GPTQ (4-bit LM head, the default) | 229 | 1,385 | 3,350 | 51.8% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 25% | 107 | 953 | 2,600 | 46.2% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 40% | 112 | 960 | 2,614 | 42.0% ± 2.2% |
| vLLM fp16 (reference) | 75 | 538 | 2,236 | — |
| Turing fp16 (reference) | 77 | 547 | 2,265 | — |
<!-- /table -->

**16K-token prompts at equal KV memory** (Qwen2.5-1.5B-Instruct; seconds per request and each engine's KV capacity in
tokens; 14.3 to 14.6 GiB of KV, 12.7 GiB for the Quest row; `results_longctx_cuda.json`). Read it with the quality
table below: vLLM fp8 without a skipped layer is fast because it is wrong on this model. The rows are from the run
of 2026-11-23 and 24, in blocks of the rows that are compared with each other, each block within one boot of the VM
(the 1-byte rows all in one boot);
Turing's Quest row and vLLM's per-token caches are from 2026-11-17.

<!-- table:longctx -->
| Engine, KV cache | KV capacity (tokens) | B=16 | B=32 | B=64 | B=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing fp8 | 1,059,056 | 1.86 | 1.77 | 1.71 | 1.68 |
| Turing int8 | 1,029,104 | 1.90 | 1.80 | 1.75 | 1.72 |
| vLLM fp8 | 1,069,392 | 1.82 | 1.73 | 1.74 | 1.72 |
| vLLM fp8, skip layer 0 | 1,032,640 | 1.81 | 1.72 | 1.80 | 1.79 |
| vLLM auto | 547,568 | 2.22 | 2.23 | 2.58 | 2.65 |
| Turing native | 547,568 | 2.09 | 1.96 | 1.93 | 1.90 |
| Turing int4 | 1,304,864 | 2.25 | 2.12 | 2.09 | 2.06 |
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
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 1.90 |
| Turing int8 | automatic | 58.6% | 100.0% | -0.000 ± 0.002 | 1.72 |
| Turing fp8 | automatic | 56.6% | 100.0% | +0.009 ± 0.005 | 1.68 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.06 |
| vLLM bf16 (reference) | — | 57.4% | 100.0% | — | 2.65 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.72 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.79 |
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
