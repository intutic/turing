# Turing Engine

**HuggingFace-exact inference for Llama-family models, with an int8 KV cache that halves cache memory at no measured quality cost.**

[![Release](https://img.shields.io/badge/Release-v1.7.0-blue.svg)](https://github.com/intutic/turing/releases/tag/v1.7.0)





[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-383%20%2818%20need%20CUDA%29-brightgreen.svg)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ What is Turing Engine?

Turing Engine is an inference and serving runtime for NVIDIA GPUs and Apple Silicon Macs. By default it reproduces the HuggingFace model's logits exactly; every lossy option (KV quantization, page selection, FFN pruning) is opt-in, and the measured ones are in [benchmarks](benchmarks.md). Real-weight runs so far cover SmolLM2 135M–1.7B (Llama architecture); no MoE checkpoint and no model larger than 1.7B has been run end to end, and the serving path keeps fp16 weights, so a 70B model does not fit on one 24 GB GPU.

---

## 🚀 Core Technologies

**Measured on real weights** ([benchmarks](benchmarks.md)):

* **Exact HuggingFace parity** for Llama-architecture checkpoints (mean NLL difference below 0.00001 on SmolLM2 135M–1.7B in fp32).
* **int8 / int4 KV cache with fused batched decode**: int8 halves the cache (−48% at head_dim 128) with no measurable quality cost; on CUDA one fused append + attention kernel per layer serves every running request (1,274 tok/s vs 833 for HuggingFace transformers, SmolLM2-360M, batch 64, 2K context, NVIDIA L4).
* **Opt-in FFN channel pruning**: smaller matrix multiplies, but +6.8 to +15.8 nats NLL at 50% on SmolLM2, so it is off by default.

**Implemented, not measured on a model** (unit tests and random-data micro-benchmarks only; see the [claims audit](claims-audit.md) for why earlier numbers were removed):

* **Serving**: OpenAI (`/v1/chat/completions`), Anthropic (`/v1/messages`) and Ollama (`/api/*`) endpoints in one server; JSON-schema prompting, truncated-JSON repair and tool-call parsing; token-budget admission control with three priority lanes; a gate that turns speculation off at 4+ active streams. `/v1/embeddings`, `/v1/rerank` and `/v1/extract` are placeholders (random embeddings, word-overlap rerank, substring extract).
* **Loading**: native GGUF reader (`Q4_0`, `Q4_K_M`, `Q8_0`, `FP16`), safetensors mmap with readahead, `io_uring` and GPUDirect Storage paths.
* **Turing DSL**: `@turing.chain`, `fork()`, `join()`, `select()`, `constrain()` (branches currently re-prefill the shared prefix).
* **Standalone C++20 binary (`turing-cli`)** with an embedded HTTP server.
* **Experimental kernels and components**: paged and split-K decode GEMV, linear-recurrence and hybrid anchor attention, latent (SVD INT8) decode, k-slot KV pooling, Matryoshka draft heads, multi-turn `clean_base` lineage, cross-model KV transfer, MoE expert cache (LFRU, routing history, PILOT prefetch), tensor hashing, reasoning-tag scanner. Some are slower than the baseline they replace.
* **Tensor and pipeline parallel building blocks** (multi-GPU serving not validated).
* **Hardware dispatch**: CUDA (Triton), Apple Silicon (MPS) and CPU (AVX2/NEON). AMD ROCm, Intel XPU and Vulkan have dispatch code but no recorded run.

---

## ⏱️ Quickstart in 30 Seconds

```bash
# 1. Install Turing Engine
pip install turing-engine

# 2. Chat in your terminal with real weights
turing chat --model deepseek-r1-1.5b

# 3. Launch an OpenAI-compatible serving server on port 8000
turing serve --model deepseek-r1-7b --port 8000
```

---

## 📖 Navigation

* [🌐 Technical Comparison: vs vLLM, SGLang, Ollama & llama.cpp](comparison.md)
* [⚡ Quickstart Guide](quickstart.md)
* [🌐 Supported Models & Hardware Setup](models_and_hardware.md)
* [🚀 Serving & API Reference](serving.md)
* [☸️ Kubernetes & llm-d Integration](llmd-integration.md)
* [🧠 Architecture Deep-Dive](architecture.md)
* [🔌 Ecosystem Integrations](integrations.md)
* [📊 Empirical Benchmarks](benchmarks.md)
* [🔍 Claims Audit](claims-audit.md)
* [📜 Licensing Terms](licensing.md)

