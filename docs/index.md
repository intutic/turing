# Turing Engine

**Serve 70B+ LLMs on a Single 24GB GPU or Mac with 75% Less Memory.**

[![Release](https://img.shields.io/badge/Release-v0.3.1-blue.svg)](https://github.com/intutic/turing/releases/tag/v0.3.1)

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests: 131/131 Passing](https://img.shields.io/badge/Tests-131%2F131%20Passing-brightgreen.svg)]()


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ What is Turing Engine?

Turing Engine is an open-source inference and serving runtime that allows you to run frontier models (**LLaMA-3.3-70B**, **DeepSeek-R1 Distill**, **Qwen-2.5-72B**, **GLM-5.3-Flash-320B**) on **single consumer 24GB GPUs** (RTX 3090/4090, L4) and **Apple Silicon Macs** with zero loss in reasoning accuracy.

---

## 🚀 Core Technologies

* **57% Subspace Channel Pruning**: Skips dead intermediate neurons during token generation for a **2.32× per-layer speedup**.
* **SVD INT8 KV Cache Paging**: Compresses 32K context memory from **10 GB down to 2.5 GB (-75%)**.
* **Heterogeneous MoE Memory Management**: Runs 320B Mixture-of-Experts models on a single 24GB card by caching active experts in VRAM and streaming inactive experts from Host DRAM.
* **Universal Hardware Support**: Native auto-discovery for **NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), Intel (XPU), Vulkan, and CPU (AVX2 SIMD)**.
* **OpenAI & Anthropic Compatible**: Drop-in serving runtime for `/v1/chat/completions`, `/v1/completions`, and `/v1/messages`.

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

* [⚡ Quickstart Guide](quickstart.md)
* [🌐 Supported Models & Hardware Setup](models_and_hardware.md)
* [🚀 Serving & API Reference](serving.md)
* [☸️ Kubernetes & llm-d Integration](llmd-integration.md)
* [🧠 Architecture Deep-Dive](architecture.md)
* [🔌 Ecosystem Integrations](integrations.md)
* [📊 Empirical Benchmarks](benchmarks.md)
* [📜 Licensing Terms](licensing.md)

