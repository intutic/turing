# Turing Engine

**Serve 70B+ LLMs on a Single 24GB GPU or Mac with 75% Less Memory.**

[![Release](https://img.shields.io/badge/Release-v0.1.8-blue.svg)](https://github.com/intutic/turing/releases/tag/v0.1.8)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests: 103/103 Passing](https://img.shields.io/badge/Tests-103%2F103%20Passing-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ What is Turing Engine?

Turing Engine is an open-source LLM serving runtime that makes large models (**LLaMA-3.3-70B**, **DeepSeek-R1 Distill**, **Qwen-2.5-72B**, **GLM-5.3-Flash-320B**) run fast on **consumer 24GB GPUs** (RTX 3090/4090, L4) and **Apple Silicon Macs** without destroying accuracy.

---

## 🚀 Key Features

* **57% Subspace Channel Pruning**: Skips inactive neurons during token generation for a **2.32× layer speedup**.
* **SVD INT8 KV Cache Paging**: Compresses 32K context memory from **10 GB down to 2.5 GB (-75%)**.
* **Heterogeneous MoE Offload**: Runs massive Mixture-of-Experts models (e.g. GLM-5.3-Flash, DeepSeek-V4) by keeping active experts in VRAM and inactive experts in Host RAM.
* **Universal Hardware Support**: Auto-discovers and accelerates on **NVIDIA (CUDA), AMD (ROCm), Apple Silicon (Metal), Intel (XPU), Vulkan, and CPU (AVX2 SIMD)**.
* **Zero-Token Latent Agents (XKV)**: Lets multi-agent systems communicate directly in latent KV space for **7.85× faster deliberation** with real-time auditability.
* **OpenAI & Anthropic Compatible**: Drop-in replacement for `/v1/chat/completions` and `/v1/messages`.

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
* [🌐 Hardware & Accelerators](hardware/universal_backends.md)
* [🦜️🔗 LangChain & LangGraph Integration](integrations/langchain.md)
* [📊 Hardware Benchmarks & Sizing](benchmarks/gpu_benchmarks.md)
* [🧠 Architecture Deep-Dive](architecture/index.md)
* [💻 CLI Commands Reference](serving/cli.md)
