# Turing Engine

**Serve 70B+ LLMs on a Single 24GB GPU or Mac with 75% Less Memory.**

[![Release](https://img.shields.io/badge/Release-v0.6.0-blue.svg)](https://github.com/intutic/turing/releases/tag/v0.6.0)





[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-230%2F230%20Passing-brightgreen.svg)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ What is Turing Engine?

Turing Engine is an open-source inference and serving runtime that allows you to run frontier models (**LLaMA-3.3-70B**, **DeepSeek-R1 Distill**, **Qwen-2.5-72B**, **GLM-5.3-Flash-320B**) on **single consumer 24GB GPUs** (RTX 3090/4090, L4) and **Apple Silicon Macs** with zero loss in reasoning accuracy.

---

## 🚀 Core Technologies

* **Triple API Gateway**: Unified drop-in endpoints for **OpenAI** (`/v1/chat/completions`), **Anthropic** (`/v1/messages`), and **Ollama** (`/api/*`).
* **Structured Outputs & Tool Calling**: Native JSON Schema validation, JSON Mode auto-repair, and standardized OpenAI/Anthropic tool calling.
* **57% Subspace Channel Pruning**: Skips dead intermediate neurons during token generation for a **2.32× per-layer speedup**.
* **Latent Flash-Decode (SPECTRA Mode-B)**: Direct attention in rank-64 latent subspace against INT8 cached singular coordinates (**35.37× speedup on 32K context**).
* **Multi-Turn Clean-Base Lineage**: Cryptographic BLAKE2b ledger preserving bounded representation fidelity ($\|\Delta C_R\|_2 \approx 30.70$) across multi-agent deliberations with zero drift.
* **$k$-Slot Symmetric Pooling**: Compresses $N$-token KV caches into $k=4$ learned summary slots per head/layer via fused Triton kernels (**3.1× transfer speedup at $N=8,192$**).
* **AI Traffic Management & 3-Lane QoS**: Sub-50µs VRAM admission control (HTTP 429 queuing with `Retry-After: 2.0s`), 64-bit FNV-1a prefix routing, and `Interactive`/`Batch`/`Background` priority scheduling.
* **Concurrency-Adaptive Speculation Gating**: Dynamically throttles speculation tree width under load ($1.82\times$ at $c=1$, plain decode at $c \ge 4$), with byte-exact greedy decode parity.
* **3:1 Hybrid Linear-Full Recurrence**: 75% linear recurrence layers + 25% chunk-scoring full attention, eliminating long-context OOM crashes up to 65K+ tokens.
* **SVD INT8 KV Cache Paging**: Compresses 32K context memory from **10 GB down to 2.5 GB (-75%)**.
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

