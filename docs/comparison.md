# 🌐 Comparison with vLLM, SGLang, TensorRT-LLM, llama.cpp and Ollama

**Measured on one NVIDIA L4, same model, same prompts, same GPU:** Turing Engine against vLLM 0.31 and llama.cpp
(CUDA build). SGLang, TensorRT-LLM and Ollama were not run (Ollama serves through llama.cpp). Tables are generated from
the committed results by `scripts/make_doc_tables.py`.

---

## Speed

**fp16, 1,024-token prompts, 256 new tokens, all requests submitted at once** (`results_e2e_generation_cuda.json`;
output tokens per second, prefill included):

<!-- table:vs_short -->
| Model | Engine | B=1 | B=8 | B=32 | B=64 | TTFT, B=1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SmolLM2-1.7B | vLLM | 65 | 324 | 563 | 622 | 0.073 s |
| SmolLM2-1.7B | Turing | 58 | 271 | 454 | 443 | 0.102 s |
| SmolLM2-360M | vLLM | 230 | 1,258 | 2,375 | 2,708 | 0.029 s |
| SmolLM2-360M | Turing | 150 | 855 | 1,555 | 1,781 | 0.175 s |
<!-- /table -->

**4 bits, Qwen2.5-1.5B-Instruct, 512-token prompts, 256 new tokens** (`results_4bit_speed_cuda.json`, GSM8K from
`results_4bit_quality_*_cuda.json`, all three through Turing's evaluator):

<!-- table:vs_4bit -->
| Setting | Batch 1, tok/s | Batch 8, tok/s | GSM8K (n=500) |
| :--- | :---: | :---: | :---: |
| llama.cpp Q4_K_M | 186 | 790 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 169 | 1,078 | 50.2% ± 2.2% |
| Turing GPTQ | 87 | 557 | 52.6% ± 2.2% |
| Turing GPTQ + 4-bit LM head (opt-in) | — | — | — |
| Turing GPTQ + TEAL 25% | 90 | 531 | 47.2% ± 2.2% |
| Turing GPTQ + TEAL 40% | 93 | 530 | 43.6% ± 2.2% |
| vLLM fp16 (reference) | 75 | 533 | — |
| Turing fp16 (reference) | 65 | 440 | — |
<!-- /table -->

**16K-token prompts at equal KV memory** (Qwen2.5-1.5B-Instruct; seconds per request; each engine's KV capacity in
tokens; `results_longctx_cuda.json`). Read it with the quality table below: vLLM fp8 without a skipped layer is fast
because it is wrong on this model.

<!-- table:longctx -->
| Engine, KV cache | KV capacity (tokens) | B=16 | B=32 | B=64 | B=128 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing native | 547,568 | 3.67 | 3.59 | 3.53 | 3.51 |
| Turing int8 | 993,184 | 3.74 | 3.64 | 3.59 | 3.56 |
| Turing int4 | 1,304,864 | 3.80 | 3.61 | 3.57 | 3.54 |
| Turing int4, Quest 25% | 1,135,712 | 4.58 | 4.24 | 3.98 | 3.87 |
| vLLM auto | 547,856 | 2.37 | 2.37 | 2.53 | 2.60 |
| vLLM fp8 | 1,069,392 | 1.90 | 1.75 | 1.72 | 1.71 |
| vLLM fp8, skip layer 0 | 1,032,096 | 1.85 | 1.74 | 1.75 | 1.73 |
| vLLM int8_per_token_head | 1,062,096 | 4.90 | 4.79 | 4.74 | 4.72 |
| vLLM int4_per_token_head | 2,060,896 | 4.73 | 4.55 | 4.49 | 4.41 |
| vLLM int4_per_token_head, skip layer 0 | 1,876,928 | 4.51 | 4.42 | 4.36 | 4.30 |
<!-- /table -->

## Low-bit KV quality

Qwen2.5-1.5B-Instruct in bf16; the same GSM8K, pass-key and WikiText-2 checks for both engines
(`results_quality_vllm_cuda.json` and Turing's final results):

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 3.51 |
| Turing int8 | — | 58.2% | 100.0% | +0.003 ± 0.002 | 3.56 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 3.54 |
| vLLM fp16 (reference) | — | 57.4% | 100.0% | — | 2.60 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.71 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
| vLLM fp8_per_token_head | — | 0.6% | 0.0% | +2.404 ± 0.364 | — |
| vLLM int8_per_token_head | — | 41.4% | 100.0% | +0.023 ± 0.007 | 4.72 |
| vLLM int4_per_token_head | — | 0.6% | 0.0% | +8.127 ± 0.495 | 4.41 |
| vLLM int4_per_token_head | layer 0 (manual) | 47.8% | 100.0% | +0.058 ± 0.010 | 4.30 |
| vLLM turboquant_k8v4 | — | 52.0% | 100.0% | +0.001 ± 0.001 | — |
<!-- /table -->

---

## Where Turing Engine stands

* **Slower.** vLLM is 1.1–1.5× faster in fp16 and about 2× faster at 4 bits; llama.cpp leads at batch 1 in 4 bits.
  Turing's time to first token is 1.4–10× vLLM's.
* **Comparable quality.** Turing's fp16 matches vLLM's on GSM8K (57.2% and 57.4%), and its GPTQ matches llama.cpp's
  Q4_K_M (52.6% and 52.4%; vLLM's GPTQ checkpoint 50.2%).
* **Low-bit KV that stays correct by default.** Turing picks the layers to keep in full precision at startup; vLLM's
  default fp8 and its per-token int4/fp8 caches break this model unless layer 0 is excluded by hand, and its per-token
  int8 cache loses 16 GSM8K points.
* **Not measured:** multi-GPU, SGLang, TensorRT-LLM, models larger than OLMoE-1B-7B, other model families for the KV
  comparison.

---

## Feature check

Each cell for another engine was checked against that project's official documentation on 2026-11-16 (links below
the table). "not checked" means the documentation page consulted did not settle it; it is not a claim that the
feature is missing. Turing cells describe the code at this commit.

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
| The fastest serving on NVIDIA GPUs, multi-GPU | vLLM, SGLang or TensorRT-LLM (faster than Turing in every measurement above) |
| A C/C++ server with GGUF as its native format, or the fastest batch-1 4-bit decode measured here | llama.cpp |
| A desktop runtime that imports GGUF and Safetensors models [11] | Ollama |
| A small, readable engine to test KV-cache and quantization ideas against real models, with exact HuggingFace logits and measured evals | Turing Engine |
