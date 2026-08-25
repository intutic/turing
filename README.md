# ⚡ Turing Engine

**Serve Frontier 70B+ Models on a Single Consumer GPU (24GB) or Mac Workstation with 75% Less Memory.**

[![Docs: Live](https://img.shields.io/badge/Documentation-intutic.github.io%2Fturing-blue.svg)](https://intutic.github.io/turing/)
[![Release: v0.1.4](https://img.shields.io/badge/Release-v0.1.4-blue.svg)](https://github.com/intutic/turing/releases/tag/v0.1.4)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)
[![Tests: 94/94 Passing](https://img.shields.io/badge/Tests-94%2F94%20Passing-brightgreen.svg)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ Quickstart in 30 Seconds

### 1. Install Turing Engine
```bash
pip install turing-engine
```
*(Or install locally with C++ SIMD optimizations: `git clone https://github.com/intutic/turing.git && cd turing && pip install -e .`)*

### 2. Instant Terminal Chat (No Setup Needed!)
Chat with real pretrained weights directly in your terminal:
```bash
# Chat with SmolLM2 or DeepSeek-R1:
turing chat --model smollm2
turing chat --model deepseek-r1-1.5b
```

### 3. Launch an OpenAI & Anthropic-Compatible Serving Server
```bash
# Serve models locally on port 8000:
turing serve --model smollm2 --port 8000
turing serve --model deepseek-r1-7b --port 8000
```
Point any chat UI (Open WebUI, LibreChat, Chatbox, Cursor) to `http://localhost:8000/v1` and you're chatting!

---

## 🌐 Supported Frontier Models

Turing Engine natively ingests, compresses, and serves open weights directly from Hugging Face:

| Lab / Family | Models Supported | Single-GPU Hardware Target |
| :--- | :--- | :--- |
| **DeepSeek** | `deepseek-r1-1.5b` to `70b`, `deepseek-v3` | 1x 6GB–24GB GPU / Mac |
| **Alibaba Qwen** | `qwen-coder-32b`, `qwen-coder-7b`, `qwen-72b`, `qwen-14b`, `qwen-7b` | 1x 12GB–24GB GPU / Mac |
| **Meta AI** | `llama-3.3-70b`, `llama-3.1-70b`, `llama-3.1-8b`, `llama-3.2-1b/3b` | 1x 8GB–24GB GPU / Mac |
| **Mistral AI** | `mistral-small-24b`, `mistral-7b`, `ministral-8b`, `mistral-large-123b` | 1x 16GB–24GB GPU / Mac |
| **Google** | `gemma-2-27b`, `gemma-2-9b`, `gemma-2-2b` | 1x 8GB–24GB GPU / Mac |
| **Zhipu / OpenBMB** | `glm-4-9b`, `internlm3-8b`, `minicpm3-4b`, `yi-1.5-34b` | 1x 8GB–24GB GPU / Mac |
| **Microsoft** | `phi-4` (14B), `phi-4-mini` | 1x 8GB–16GB GPU / Mac |

---

## 🧠 Why Turing Engine? (3 Core Advantages)

```
       +-----------------------------------------------------------------+
       |                  TURING ENGINE INFERENCE STACK                  |
       +-----------------------------------------------------------------+
          |                                                             |
          v                                                             v
+-----------------------------+                               +-----------------------------+
|   57% Subspace Channel      |                               |    SVD INT8 KV Cache        |
|   Activation Pruning        |                               |    Memory Compression       |
|                             |                               |                             |
|  * Bypasses inactive FFN    |                               |  * 32K context memory drops |
|    channels during decode   |                               |    from 10.0 GB -> 2.5 GB   |
|  * 2.32x CUDA Layer Speedup |                               |  * -75% VRAM KV footprint   |
+-----------------------------+                               +-----------------------------+
```

1. **⚡ 57% Subspace Activation Pruning (2.32× Layer Speedup)**:
   Dynamically slices out inactive feed-forward channels during generation, delivering a measured **2.32× CUDA layer speedup** with zero loss in reasoning accuracy.

2. **💾 75% SVD INT8 KV Cache Compression**:
   Compresses attention Key-Value states into calibrated Rank-64 singular vectors, reducing 32K context memory from **10.0 GB down to 2.5 GB (-75%)**.

3. **🍏 Universal Hardware Support (Apple Silicon & NVIDIA CUDA)**:
   Automatically auto-dispatches between NVIDIA CUDA (Triton GPU kernels), Apple Silicon Metal (`mps`), and bare-metal C++20 AVX2 SIMD CPU routines.

---

## 💻 CLI Quick Reference

```bash
# 1. Start interactive terminal chat:
turing chat --model smollm2

# 2. Serve OpenAI & Anthropic compatible API:
turing serve --model deepseek-r1-7b --port 8000

# 3. Generate text from a single prompt:
turing generate --model gpt2 --prompt "Artificial intelligence is"

# 4. Run live mathematical reasoning (GSM8K) evaluation:
turing eval-accuracy --model gpt2 --samples 5

# 5. Run physical hardware micro-benchmarks:
python scripts/benchmark_comprehensive_matrix.py
```

---

## 🔌 Using with Python OpenAI SDK

```python
from openai import OpenAI

# Connect to local Turing Engine server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-live")

response = client.chat.completions.create(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Explain quantum computing in two sentences."}],
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## 📚 In-Depth Documentation

For researchers and infrastructure engineers looking for deep architectural specs and raw data:

* 📖 **[Quickstart Guide](docs/quickstart.md)** — Detailed setup, Docker, and multi-GPU configurations.
* 🧠 **[Architecture Deep-Dive](docs/architecture/index.md)** — Mathematical formulations, SVD rank-64 projections, and Triton kernel designs.
* 📊 **[Full Hardware Benchmark Matrix](docs/benchmarks/gpu_benchmarks.md)** — Layer latency comparisons, memory geometry, and physical silicon data across 10+ frontier models.
* 💰 **[Cloud TCO & Cost Comparison](docs/benchmarks/tco_comparison.md)** — Analysis of 96.7% annual hosting savings ($210k ➔ $7k/yr).
* 📜 **[Licensing & Community Tier](docs/licensing.md)** — Business Source License 1.1 terms (Free for all single-node and consumer GPU deployments).

---

## 📦 Pre-Built Binary Wheels (v0.1.4)

Pre-compiled binary wheels with native C++20 AVX2 SIMD optimizations are published on [GitHub Releases v0.1.4](https://github.com/intutic/turing/releases/tag/v0.1.4):

```bash
# Install directly from release wheel:
pip install https://github.com/intutic/turing/releases/download/v0.1.4/turing_engine-0.1.4-cp311-cp311-macosx_15_0_arm64.whl
```

---

## 📄 License

Turing Engine is licensed under the **Business Source License 1.1 (BSL 1.1)**. 
* **Free & Open for Community Use**: Completely free for single-node development, testing, research, and non-commercial deployments.
* Converts automatically to **Apache 2.0** on **March 1, 2030**.
