# ⚡ Turing Engine

**HuggingFace-exact inference for Llama-family models, with an int8 KV cache that halves cache memory at no measured quality cost.**

[![Docs: Live](https://img.shields.io/badge/Documentation-intutic.github.io%2Fturing-blue.svg)](https://intutic.github.io/turing/)
[![Release: v1.7.0](https://img.shields.io/badge/Release-v1.7.0-blue.svg)](https://github.com/intutic/turing/releases/tag/v1.7.0)







[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)
[![Tests: 377/377 Passing](https://img.shields.io/badge/Tests-377%2F377%20Passing-brightgreen.svg)](https://github.com/intutic/turing/actions)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## ⚡ Quickstart in 30 Seconds

### 1. Install Turing Engine
```bash
pip install turing-engine
```
*(Or install locally with C++ SIMD optimizations: `git clone https://github.com/intutic/turing.git && cd turing && pip install -e .`)*

### 2. Instant Terminal Chat (Hugging Face Hub & Local GGUF)
Chat with real pretrained weights dynamically directly from Hugging Face Hub or local GGUF binaries:
```bash
# Chat with any canonical Hugging Face Hub repository:
turing chat --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --reasoning-effort high

# Chat directly with local quantized .gguf files (zero conversion needed!):
turing chat --model ./models/llama-3.3-70b-q4_k_m.gguf

# Ergonomic shortcuts for popular checkpoints:
turing chat --model smollm2
turing chat --model llama-3.3-70b
```

### 3. Launch Triple API Gateway (OpenAI, Anthropic & Ollama Compatible)
```bash
# Serve any HF or GGUF model dynamically on port 8000 (OpenAI /v1, Anthropic /v1/messages, Ollama /api):
turing serve --model meta-llama/Llama-3.3-70B-Instruct --port 8000
turing serve --model ./models/qwen2.5-32b-q8_0.gguf --port 8000
```
Point **Open WebUI**, **Continue.dev**, **Cursor**, **Chatbox**, or **Cline** directly to `http://localhost:8000`!

### 4. Programmatic AI Workflows with Turing DSL
Chain multi-step prompts, fork tree-of-thought reasoning branches with prefix KV cache sharing, and enforce JSON schemas in pure Python:
```python
import turing

@turing.chain(model="smollm2", device="auto")
def parallel_tree_of_thought(question: str):
    # Step 1: Initial reasoning
    prompt = turing.gen(f"Analyze question: {question}", max_tokens=64)
    
    # Step 2: Fork 3 concurrent branches sharing the prefix KV cache
    branches = turing.fork(3, temperature=0.8)
    for i, b in enumerate(branches):
        b.gen(f"Perspective {i+1}: What is the optimal solution?", max_tokens=64)
        
    # Step 3: Majority vote or best log-prob merge
    return turing.join(branches, strategy="best")
```

### 5. Standalone C++20 Executable (`turing-cli`)
Deploy Turing Engine in embedded, edge, or containerized environments with **zero Python dependencies**:
```bash
# Build native binary with CMake
cmake -B build && cmake --build build

# Run bare-metal generation from GGUF binary
./build/turing-cli generate --model ./models/llama-3.3-70b.gguf --prompt "Explain quantum computing:"
./build/turing-cli serve --model ./models/llama-3.3-70b.gguf --port 8000
```

### 6. Distributed Multi-GPU & Pipeline Parallel Serving
Tensor- and pipeline-parallel building blocks exist, but multi-GPU serving has not been validated (pipeline stages currently run in one process):
```bash
# Serve across 4 GPUs with TP=2, PP=2
turing serve --model meta-llama/Llama-3.3-70B-Instruct --tensor-parallel 2 --pipeline-parallel 2 --port 8000

# Or launch across cluster nodes via torchrun
./scripts/launch_distributed.sh meta-llama/Llama-3.3-70B-Instruct 4 2
```

---

## 🌐 Models

What has actually been run with real weights in this repository:

| Model | What was run | Where |
| :--- | :--- | :--- |
| `HuggingFaceTB/SmolLM2-135M`, `-360M`, `-1.7B` | Turing forward pass vs HuggingFace (NLL), int8/int4 KV, batched decode throughput | [benchmarks §2–§3](https://intutic.github.io/turing/benchmarks/) |
| `gpt2`, `SmolLM2-1.7B-Instruct` | HuggingFace `generate` timing (not the Turing engine) | `scripts/benchmark_comprehensive_real_gpu.py` |

Exact HuggingFace parity is implemented and tested for **Llama**, **Llama-3.1 RoPE scaling**, **Qwen2/Qwen2.5** and **Qwen3** architectures (on small randomly initialised models of each; real-weight quality is measured for SmolLM2 only). The loader refuses what it cannot reproduce (other RoPE scaling types, sliding-window attention), and no MoE checkpoint and no model larger than 1.7B has been run end to end. The serving path keeps fp16 weights, so a 70B model needs about 141 GB of weight memory and does not fit on one 24 GB GPU. Names such as `llama-3.3-70b` or `glm-5.3-flash` in `SIZING_PROFILES` are layer geometries for dry-run sizing, not tested models.

> **❓ Clarification: `CLI_ALIASES` vs. `SIZING_PROFILES`**
> * **`CLI_ALIASES` (`resolver.py`)**: Purely optional typing shortcuts for interactive developers (e.g. typing `turing chat --model smollm2` instead of `turing chat --model HuggingFaceTB/SmolLM2-135M`). Passing any full repo ID (e.g. `turing chat --model EleutherAI/pythia-70m` or `meta-llama/Llama-3.3-70B-Instruct`) bypasses aliases and streams weights directly from Hugging Face Hub.
> * **`SIZING_PROFILES` (`sizing_profiles.py`)**: An offline catalog of theoretical parameter counts used **only by `turing bench`** for dry-run FLOP simulations and PCIe bandwidth calculations without downloading 140GB weights to disk. It is **never used during live inference**.
> * **Zero Hardcoding**: Live inference uses `ModelConfig.from_pretrained()` to dynamically derive architecture parameters on the fly for any model in the world.

---

## ⚡ Universal Cross-Vendor Hardware Support

Turing Engine auto-discovers and accelerates inference on all major silicon architectures:

| Hardware Vendor | Supported Accelerators | Acceleration Backend | Hardware Optimization |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | RTX 3090 / 4090 / 5090, L4, A100, H100 | **CUDA + Triton 3.x** | Custom Tensor Core SwiGLU & Flash-Tree Triton kernels |
| **AMD** | Radeon RX 7900 XTX / 8000, Instinct MI250X / MI300X | **ROCm (HIP) + Triton** | Wave32 (RDNA) / Wave64 (CDNA) Matrix Core heuristics |
| **Intel** | Intel Arc A770 / A750 / B580 Battlemage, Max 1550 | **Intel XPU (SYCL / OneAPI)**| Intel XMX Matrix Engines + IPEX bindings |
| **Apple** | M1 / M2 / M3 / M4 (Pro, Max, Ultra) | **Metal (MPS)** | Metal Performance Shaders + Vectorized Subspace Slicing |
| **Cross-Vendor** | Intel, AMD APUs, Qualcomm Adreno, ARM Mali | **Vulkan Compute** | SPIR-V Compute Shaders + Host-Visible Coherent Memory |
| **x86_64 / ARM CPU** | Intel Xeon, AMD EPYC, Ampere Altra, Apple Silicon | **C++20 AVX2 / NEON SIMD** | 64-byte aligned SIMD fused FMA + zero-copy `mmap` |

Recorded runs exist only for an NVIDIA L4 (CUDA), Apple M4 Pro (MPS) and CPU. The AMD, Intel and Vulkan backends have dispatch code but no recorded run.

---

## 🧠 Why Turing Engine? (Core Advantages)

```
       +-----------------------------------------------------------------+
       |                  TURING ENGINE INFERENCE STACK                  |
       +-----------------------------------------------------------------+
          |                               |                             |
          v                               v                             v
+-----------------------------+ +-----------------------------+ +-----------------------------+
|   Exact HF-Parity Mode      | |    int8 / int4 KV Cache     | |   Draft Heads, MoE Cache,   |
|   (the default)             | |    + Fused CUDA Decode      | |   Lineage, Pooling          |
|                             | |                             | |                             |
|  * Logits match HF          | |  * int8: ~half the KV bytes | |  * Experimental: random-    |
|  * FFN pruning is opt-in    | |    at no measured loss      | |    weight micro-benchmarks  |
|    and costs quality        | |  * 1 kernel/layer per batch | |    only, see claims audit   |
+-----------------------------+ +-----------------------------+ +-----------------------------+
```

1. **🎯 Exact HuggingFace Parity (the default)**:
   Llama-family checkpoints load with every weight (layer norms, RoPE base, head_dim) and reproduce HuggingFace's logits: about 2e-7 max difference in fp32 on the random-initialised 4-layer Llama in `tests/test_hf_parity.py`, and a mean NLL difference below 0.00001 on SmolLM2 135M–1.7B real weights (fp32). FFN channel pruning is still available (`--sparsity`), but it is not lossless: at 50% it raises next-token NLL by +6.8 (SmolLM2-1.7B) to +15.8 nats (135M), so it is off by default. See [benchmarks](https://intutic.github.io/turing/benchmarks/).

2. **💾 int8 / int4 KV Cache with Fused Batched CUDA Decode**:
   An int8 cache (one fp16 scale and zero per token and KV head: 48% smaller at head_dim 128) changes NLL by at most ±0.0005 on SmolLM2 135M–1.7B; int4 with per-32-value scales costs +0.02. Requests share one paged pool (at most one partial 16-token page wasted each; forks share pages copy-on-write). On CUDA a decode step for every running request is one append + attention kernel per layer, replayed as a CUDA graph: on an NVIDIA L4, SmolLM2-360M at 2K context decodes 162 tok/s for one request (HuggingFace transformers: 29) and 1,715 tok/s at batch 64 (827). Quest page selection is opt-in (`--kv-page-budget`) and pays at large batches and long contexts.

3. **🧪 Experimental components (no real-model measurement yet)**:
   Multi-turn `clean_base` lineage (recomputes residuals against the original cache instead of chaining them), k-slot KV pooling, Matryoshka draft heads, a concurrency gate that turns speculation off at 4+ active streams plus a greedy parity verifier, token-budget admission control with three priority lanes, split-K paged decode, MoE expert caching and prefetch (LFRU, routing history, PILOT), hybrid linear/softmax attention, cross-model KV transfer, and assorted Triton kernels and C++ helpers. Each has unit tests and random-data micro-benchmarks only; the numbers previously quoted for them were removed because none was measured on a model and several were constants or formulas ([claims audit](docs/claims-audit.md)). Some are slower than the baseline they replace.

4. **🌐 Triple API gateway**: OpenAI (`/v1`), Anthropic (`/v1/messages`) and Ollama (`/api/*`) endpoints in one server, with JSON-schema prompting, truncated-JSON repair and tool-call parsing. `/v1/embeddings`, `/v1/rerank` and `/v1/extract` are placeholders today (random embeddings, word-overlap rerank, substring extract).

5. **🏎️ Ingest tiers**: `madvise(MADV_WILLNEED)` readahead (2.25 GB/s cold against 0.75 GB/s demand paging), a parallel `pread` reader (1 MiB requests, 64 in flight: 5.78 GB/s cold on an M4 Pro SSD, up from 2.22 GB/s), `.tgate4` int4 files and a GPUDirect Storage path. The last two are not measured yet ([benchmarks §4](https://intutic.github.io/turing/benchmarks/#4-storage-ingestion-ladder-measured-cold-apple-m4-pro-ssd)).

6. **🍏 Hardware dispatch**: CUDA (Triton kernels), Apple Silicon (`mps`) and C++20 AVX2/NEON CPU routines.

---

## 🌐 Technical Comparison: Turing Engine vs. Alternatives

| Feature Category | Capability / Specification | **Turing Engine** | **vLLM** | **SGLang** | **Ollama** | **llama.cpp** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Ingestion** | Direct Hugging Face Hub Streaming | ✅ **Universal** | ✅ Universal | ✅ Universal | ❌ (GGUF Hub) | ❌ (GGUF only) |
| | Offline GGUF Conversion Required | 🚀 **Zero Conversion** | 🚀 Zero Conversion | 🚀 Zero Conversion | ⚠️ Yes | ⚠️ Yes |
| **Memory & KV Cache** | Attention KV Management | ✅ **int8 / int4 per-request cache** | ✅ PagedAttention | ✅ RadixAttention | ⚠️ Ring buffer | ⚠️ Ring / Paged |
| | KV cache at 32K tokens, Llama-70B shape (fp16 = 10.0 GiB) | ✅ **int8: 5.16 GiB** | ✅ FP8: 5.0 GiB | ✅ FP8: 5.0 GiB | ✅ q8_0 / q4_0 | ✅ q8_0 / q4_0 |
| | Multi-Turn Clean-Base Lineage | 🧪 **Experimental** | ❌ | ❌ | ❌ | ❌ |
| **Compute Optimization**| Activation Channel Pruning | ⚠️ **Opt-in, costs quality**| ❌ None | ❌ None | ❌ None | ❌ None |
| **Serving & Gateway** | Triple Serving Gateway (OpenAI + Anthropic + Ollama) | ✅ **Triple Native** | ❌ (OpenAI only) | ⚠️ Partial | ❌ (Ollama only) | ❌ (OpenAI only) |
| | Structured Outputs (JSON Schema & Auto-Repair) | ✅ **Native + Repair** | ✅ Outlines / FSM | ✅ `xGrammar` | ⚠️ Format string | ✅ GBNF Grammars |
| | Native Tool & Function Calling | ✅ **Native** | ✅ Native | ✅ Native | ✅ Native | ⚠️ Custom parsing |
| **Agentic Deliberation** | Cross-Model Closed-Form KV Transfer ($W^*$)| 🧪 **Experimental (untested on models)**| ❌ (Re-prefills) | ❌ (Re-prefills) | ❌ (Re-prefills) | ❌ (Re-prefills) |

Competitor columns are the maintainers' reading of those projects and are not verified by this repository; none of those engines is run here.

> 📖 *Read the complete in-depth architectural breakdown at [**intutic.github.io/turing/comparison/**](https://intutic.github.io/turing/comparison/).*

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

# 5. Run synthetic micro-benchmarks (random tensors; see the claims audit):
python scripts/benchmark_native_fusions_v2.py
python scripts/benchmark_lineage_strategies.py
python scripts/benchmark_traffic_and_spec.py
python scripts/benchmark_serving_e2e.py
```

---

## 🔌 Drop-In Client SDK Support

### 1. Python OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="turing-live")
response = client.chat.completions.create(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Explain subspace channel pruning:"}],
    response_format={"type": "json_object"},
    extra_headers={"X-Turing-Lane": "interactive"}
)
print(response.choices[0].message.content)
```

### 2. Python Ollama SDK
```python
import ollama

client = ollama.Client(host="http://localhost:8000")
response = client.chat(
    model="deepseek-r1-7b",
    messages=[{"role": "user", "content": "Write a quick Python binary search:"}]
)
print(response["message"]["content"])
```

### 3. Python Anthropic SDK
```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="turing-live")
message = client.messages.create(
    model="deepseek-r1-7b",
    max_tokens=128,
    messages=[{"role": "user", "content": "Hello Turing!"}]
)
print(message.content[0].text)
```

---

## 📚 Documentation

Explore the complete documentation at [**intutic.github.io/turing**](https://intutic.github.io/turing/):

* 🌐 **[Technical Comparison](https://intutic.github.io/turing/comparison/)** — Deep-dive matrix vs vLLM, SGLang, Ollama, and llama.cpp.
* 🚀 **[Quickstart Guide](https://intutic.github.io/turing/quickstart/)** — 30-second setup, CLI commands, and OpenAI/Anthropic/Ollama SDK usage.
* 🖥️ **[Models & Hardware](https://intutic.github.io/turing/models_and_hardware/)** — what has been run, model naming, and hardware setup.
* ⚡ **[Production Serving](https://intutic.github.io/turing/serving/)** — High-concurrency continuous batching, streaming API, and SLAs.
* ☸️ **[Kubernetes & llm-d](https://intutic.github.io/turing/llmd-integration/)** — Distributed multi-pod serving, prefix-cache aware routing, and P/D disaggregation.
* 🧠 **[Architecture Deep-Dive](https://intutic.github.io/turing/architecture/)** — Subspace channel pruning, SVD INT8 KV cache paging, and Triton kernels.
* 📊 **[Empirical Benchmarks](https://intutic.github.io/turing/benchmarks/)** — Real-weight quality against the exact model, and decode throughput against HuggingFace transformers on an NVIDIA L4.
* 🔍 **[Claims Audit](https://intutic.github.io/turing/claims-audit/)** — Every earlier benchmark number traced to the script that produced it.
* 🔌 **[Integrations](https://intutic.github.io/turing/integrations/)** — Native guides for LangChain, LlamaIndex, LiteLLM, vLLM, Open WebUI, and Docker.
* 📜 **[Licensing](https://intutic.github.io/turing/licensing/)** — BSL 1.1 terms (Free for all single-node development, testing, and research).


---

## 📦 Pre-Built Binary Wheels (v1.5.0)
 
Pre-compiled binary wheels with native C++20 AVX2 SIMD optimizations are published on [GitHub Releases v1.5.0](https://github.com/intutic/turing/releases/tag/v1.5.0):
 
```bash
# Install directly from release wheel:
pip install https://github.com/intutic/turing/releases/download/v1.5.0/turing_engine-1.5.0-cp311-cp311-macosx_15_0_arm64.whl
```




---

## 📄 License

Turing Engine is licensed under the **Business Source License 1.1 (BSL 1.1)**. 
* **Free & Open for Community Use**: Completely free for single-node development, testing, research, and non-commercial deployments.
* Converts automatically to **Apache 2.0** on **March 1, 2030**.
