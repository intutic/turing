# 🌐 Comparison with vLLM, SGLang, TensorRT-LLM, llama.cpp and Ollama

**No throughput comparison has been measured.** `scripts/benchmark_e2e_generation.py` runs the same batch of
requests through Turing Engine and through vLLM on one GPU (time to finish the batch, output tokens per second, time
to first token); its GPU run is pending. The [benchmarks](benchmarks.md) compare Turing only with HuggingFace
transformers (decode throughput) and llama.cpp (GGUF logit parity). Until the end-to-end run exists, assume the mature
engines are faster.

---

## Where Turing Engine stands

* **Exactness by default.** Logits match HuggingFace for 16 model types in the tests, and every lossy setting is
  opt-in. Real-weight parity is measured for SmolLM2 against HuggingFace and for 13 GGUF files against llama.cpp.
* **One process, three APIs.** OpenAI, Anthropic Messages and Ollama endpoints share one engine and one generation
  core.
* **Small and readable.** The engine, KV pool and kernels are a few Python modules and Triton kernels, with a
  standalone C++ engine beside them.
* **Not there yet.** No multi-GPU run, no throughput comparison with other engines, no measured speed for 4-bit
  weights or activation sparsity, and no model larger than OLMoE-1B-7B has been run.

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
| Production multi-GPU serving | vLLM, SGLang or TensorRT-LLM, which document multi-GPU parallelism [14–16]; Turing has no multi-GPU run |
| A C/C++ server with GGUF as its native format | llama.cpp |
| A desktop runtime that imports GGUF and Safetensors models [11] | Ollama |
| Exact HuggingFace logits in a small, inspectable server with OpenAI, Anthropic and Ollama APIs in one process, to study or extend | Turing Engine |
