# Comparison with vLLM, SGLang, TensorRT-LLM, llama.cpp and Ollama

**Setup.** One NVIDIA L4 (24 GB) running Turing Engine, vLLM 0.31 and llama.cpp (CUDA build) on the same models,
prompts and GPU. SGLang, TensorRT-LLM and Ollama were not run (Ollama serves through llama.cpp). The tables are
generated from the results files in `benchmarks/results/` by `scripts/make_doc_tables.py`; more measurements are on
the [benchmarks](benchmarks.md) page.

---

## Where Turing Engine stands

* **vLLM is faster in most settings; Turing is ahead in a few.** In fp16 vLLM is 1.01–1.04× faster on Qwen2.5-1.5B,
  and Turing reaches 88–103% of vLLM on SmolLM2. At 4 bits vLLM is 1.06–1.12× faster; with Turing's opt-in 4-bit LM
  head Turing is 1.15–1.16× vLLM at batch 1 and 8 and 1.09× llama.cpp at batch 1 (vLLM 1.04× at batch 64). At
  16K-token prompts in fp16 Turing is 1.0–1.22× faster per request; vLLM's fp8 cache with layer 0 kept is 1.3–1.4×
  faster than Turing's int8. Turing's first token at batch 64 takes up to 2.3× vLLM's (SmolLM2-1.7B); llama.cpp
  starts fastest from a cold process.
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
| SmolLM2-1.7B | vLLM | 65 | 327 | 567 | 635 | 0.071 s |
| SmolLM2-1.7B | Turing | 64 | 327 | 585 | 586 | 0.073 s |
| SmolLM2-360M | vLLM | 231 | 1,264 | 2,391 | 2,752 | 0.029 s |
| SmolLM2-360M | Turing | 204 | 1,253 | 2,342 | 2,694 | 0.042 s |
<!-- /table -->

**4 bits, Qwen2.5-1.5B-Instruct, 512-token prompts, 256 new tokens** (`results_4bit_speed_cuda.json`; GSM8K from
`results_4bit_quality_*_cuda.json`, all three engines scored by Turing's evaluator):

<!-- table:vs_4bit -->
| Setting | Batch 1, tok/s | Batch 8, tok/s | Batch 64, tok/s | GSM8K (n=500) |
| :--- | :---: | :---: | :---: | :---: |
| llama.cpp Q4_K_M | 180 | 752 | 2,157 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 171 | 1,093 | 3,065 | 50.2% ± 2.2% |
| Turing GPTQ | 155 | 1,035 | 2,739 | 50.4% ± 2.2% |
| Turing GPTQ + 4-bit LM head (opt-in) | 197 | 1,266 | 2,961 | 51.8% ± 2.2% |
| Turing GPTQ + TEAL 25% | 107 | 953 | 2,600 | 46.2% ± 2.2% |
| Turing GPTQ + TEAL 40% | 112 | 960 | 2,614 | 42.0% ± 2.2% |
| vLLM fp16 (reference) | 75 | 537 | 2,235 | — |
| Turing fp16 (reference) | 73 | 530 | 2,152 | — |
<!-- /table -->

**16K-token prompts at equal KV memory** (Qwen2.5-1.5B-Instruct; seconds per request and each engine's KV capacity in
tokens; 14.6 GiB of KV, 12.7 GiB for the Quest row; `results_longctx_cuda.json`). Read it with the quality table
below: vLLM fp8 without a skipped layer is fast because it is wrong on this model.

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

## Low-bit KV quality

Qwen2.5-1.5B-Instruct in bf16, the same GSM8K, pass-key and WikiText-2 checks for both engines
(`results_quality_vllm_cuda.json`, `results_gsm8k_final_cuda.json`, `results_niah_final_cuda.json`,
`results_quality_instruct_bf16_cuda.json`):

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 2.05 |
| Turing int8 | — | 58.2% | 100.0% | +0.003 ± 0.002 | 2.32 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.95 |
| vLLM bf16 (reference) | — | 57.4% | 100.0% | — | 2.51 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.66 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
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
| The fastest serving on NVIDIA GPUs, multi-GPU | vLLM, SGLang or TensorRT-LLM (vLLM was faster in most measurements above; SGLang and TensorRT-LLM were not run) |
| A C/C++ server with GGUF as its native format, or the fastest batch-1 4-bit decode with default settings measured here | llama.cpp |
| A desktop runtime that imports GGUF and Safetensors models [11] | Ollama |
| A low-bit KV cache that protects the layers it would break, HuggingFace-parity-tested models, and GSM8K, pass-key and NLL evals for trying KV-cache and quantization changes on real models | Turing Engine |
