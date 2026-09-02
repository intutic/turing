# Turing Engine

**Serve 70B+ LLMs on a Single 24GB GPU or Mac with 75% Less Memory.**

[![Release](https://img.shields.io/badge/Release-v1.4.0-blue.svg)](https://github.com/intutic/turing/releases/tag/v1.4.0)





[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-327%2F327%20Passing-brightgreen.svg)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ What is Turing Engine?

Turing Engine is an open-source inference and serving runtime that allows you to run frontier models (**LLaMA-3.3-70B**, **DeepSeek-R1 Distill**, **Qwen-2.5-72B**, **GLM-5.3-Flash-320B**) on **single consumer 24GB GPUs** (RTX 3090/4090, L4) and **Apple Silicon Macs** with zero loss in reasoning accuracy.

---

## 🚀 Core Technologies

* **Native GGUF Binary Loading**: Direct memory-mapped execution of quantized `.gguf` binaries (`Q4_0`, `Q4_K_M`, `Q8_0`, `FP16`) with zero offline conversion.
* **Turing Programmatic DSL**: Python prompt programming with `@turing.chain`, parallel `fork()` with prefix KV sharing, majority `join()`, `select()`, and schema `constrain()`.
* **Standalone C++20 Binary (`turing-cli`)**: Bare-metal C++20 runtime with zero Python runtime dependency, 64-byte aligned SIMD memory, and embedded HTTP server.
* **Distributed Multi-GPU & Pipeline Parallelism**: Scalable Tensor Parallelism (TP) and Pipeline Parallelism (PP) with 1F1B micro-batch scheduling across multi-node GPU clusters.
* **Triple API Gateway**: Unified drop-in endpoints for **OpenAI** (`/v1/chat/completions`, `/v1/embeddings`), **Anthropic** (`/v1/messages`), **Ollama** (`/api/*`), plus cross-encoder `/v1/rerank` and structured `/v1/extract`.
* **Structured Outputs & Tool Calling**: Native JSON Schema validation, JSON Mode auto-repair, and standardized OpenAI/Anthropic tool calling.
* **57% Subspace Channel Pruning**: Skips dead intermediate neurons during token generation for a **2.32× per-layer speedup**.
* **Batched Continuous Paged Decode GEMV**: Fused Triton PagedAttention GEMV over 2D block tables reaching **8.86× speedup (30,057 tok/s at $B=32$)**.
* **Single-Step Linear Recurrent Attention**: Fuses decay and output projection directly in SRAM (**5.13× speedup**, $57.76\,\mu\text{s}$).
* **Direct Pointer Memory Tensor Hashing**: AVX2 256-bit SIMD scanner hashing raw memory buffers at **8.18 GB/s (23.00× faster)**.
* **Streaming Zero-Copy Reasoning Scanner**: State-machine tag filter running at **2.47M tok/s**.
* **Latent Flash-Decode (SPECTRA Mode-B)**: Direct attention in rank-64 latent subspace against INT8 cached singular coordinates (**35.37× speedup on 32K context**).
* **Multi-Turn Clean-Base Lineage**: Cryptographic BLAKE2b ledger preserving bounded representation fidelity ($\|\Delta C_R\|_2 \approx 30.70$) across multi-agent deliberations with zero drift.
* **$k$-Slot Symmetric Pooling**: Compresses $N$-token KV caches into $k=4$ learned summary slots per head/layer via fused Triton kernels (**3.1× transfer speedup at $N=8,192$**).
* **AI Traffic Management & 3-Lane QoS**: Sub-50µs VRAM admission control (HTTP 429 queuing with `Retry-After: 2.0s`), 64-bit FNV-1a prefix routing, and `Interactive`/`Batch`/`Background` priority scheduling (**1.18M requests/sec**).
* **Concurrency-Adaptive Speculation Gating**: Dynamically throttles speculation tree width under load ($1.82\times$ at $c=1$, plain decode at $c \ge 4$), with byte-exact greedy decode parity.
* **Elastic Single-GPU Multi-Model Manager**: Orchestrates 5+ models (Docling OCR $\to$ GLiNER NER $\to$ BGE Embeddings $\to$ BGE Reranker $\to$ Qwen LLM) on a single 24GB GPU in **167.34 ms total** with fused segment pooling (**65.40× faster**, $179\text{M tok/s}$) and compute-cost adaptive batching (**-67.43% padding waste reduction**).
* **Hybrid Gated Softmax Anchoring**: Eliminates the 84% associative recall collapse of linear attention/SSMs on RAG tasks via dynamic factual token anchoring (**25.00% exact recall**, $\sim 16\times$ above chance floor) with chunkwise in-SRAM learned decay (**1.61× faster prefill**, $48.84\,\mu\text{s}$ decode).
* **SVD INT8 KV Cache Paging**: Compresses 32K context memory from **10 GB down to 2.5 GB (-75%)**.
* **Learned MoE Pinning, LFRU Hysteresis & Vectorized Batch-Union**: Persistent activation tracking (`RoutingHistoryTracker`, **$92.53\%$ warm hit rate**), $25\%+4$ hysteresis promotion gate (**$95.10\%$ thrash reduction**), PILOT one-layer-ahead PCIe streaming, and vectorized PyTorch `index_add_` batch-union accumulation (**up to $16.92\times$ prefill acceleration**, $113\text{k tok/s}$).
* **Heterogeneous MoE Memory Management**: Runs 320B Mixture-of-Experts models on a single 24GB card by caching active experts in VRAM and streaming inactive experts from Host DRAM.
* **Universal Hardware Support**: Native auto-discovery for **NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), Intel (XPU), Vulkan, and CPU (AVX2 SIMD)**.


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
* [📜 Licensing Terms](licensing.md)

