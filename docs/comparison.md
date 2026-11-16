# 🌐 Technical Comparison: Turing Engine vs. vLLM, SGLang, Ollama & llama.cpp

This document provides a comprehensive technical and architectural comparison between **Turing Engine**, **vLLM**, **SGLang**, **Ollama**, and **llama.cpp** across model ingestion, KV memory compression, compute optimization, serving protocols, and agentic workflows.

---

## 🏛️ Executive Summary & Core Design Philosophies

* **vLLM**: The industry-standard **cloud/datacenter production workhorse** for high-throughput batching across multi-GPU clusters (`PagedAttention`).
* **SGLang**: The **agentic multi-turn specialist** optimized for dynamic prefix caching (`RadixAttention`) and FSM-guided decoding (`xGrammar`).
* **Ollama**: The **local developer desktop champion** built on `llama.cpp` for single-user zero-config CLI execution with GGUF format.
* **llama.cpp**: The **bare-metal C/C++ foundational runtime** for edge CPU/GPU execution using quantized GGUF weights.
* **Turing Engine**: A research-stage inference and serving runtime with **HuggingFace-exact model execution** (Llama architecture), an **int8 KV cache (about half the memory, no measured quality cost)**, a **Triple API Gateway (OpenAI + Anthropic + Ollama)**, and a set of experimental components (cross-model KV transfer, lineage, k-slot pooling, MoE expert caching) that have not been measured on a model.

> **How to read this page.** The Turing column describes what the code implements; ✅ does not mean measured. Measured results are only those in [benchmarks](benchmarks.md) (SmolLM2 135M–1.7B). The vLLM, SGLang, Ollama and llama.cpp columns are the maintainers' reading of those projects, not verified here; none of those engines is run by this repository, and some entries may be out of date.

---

## 📊 Master 5-Way Architectural & Feature Matrix

| Feature Category | Capability / Specification | **Turing Engine** | **vLLM** | **SGLang** | **Ollama** | **llama.cpp** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Ingestion** | Direct Hugging Face Hub ID Streaming | ✅ **Universal** | ✅ Universal | ✅ Universal | ❌ (GGUF Hub) | ❌ (GGUF only) |
| | Native Quantized GGUF File Loader | ✅ **Native (Q4/Q8/FP16)**| ❌ (Safetensors) | ❌ (Safetensors) | ✅ Native | ✅ Native |
| | Offline GGUF Conversion Required | 🚀 **Zero Conversion** | 🚀 Zero Conversion | 🚀 Zero Conversion | ⚠️ Yes | ⚠️ Yes |
| | Tri-Part Namespace (`provider/model/effort`) | ✅ **Native** | ❌ | ❌ | ❌ | ❌ |
| | Zero Hardcoding (Dynamic AutoConfig) | ✅ **Yes** | ✅ Yes | ✅ Yes | ❌ | ❌ |
| **Memory & KV Cache** | Attention KV Management | ✅ **int8 / int4 per-request cache** | ✅ PagedAttention | ✅ RadixAttention | ⚠️ Ring buffer | ⚠️ Ring / Paged |
| | KV cache at 32K tokens, Llama-70B shape (fp16 = 10.0 GiB) | ✅ **int8: 5.16 GiB** | ✅ FP8: 5.0 GiB | ✅ FP8: 5.0 GiB | ✅ q8_0 / q4_0 | ✅ q8_0 / q4_0 |
| | Multi-Turn Clean-Base Lineage | 🧪 **Experimental** | ❌ | ❌ | ❌ | ❌ |
| | $k$-Slot Cache Pooling | 🧪 **Experimental** | ❌ | ❌ | ❌ | ❌ |
| | Prefix Caching Across Requests | ❌ Not in the serving path | ⚠️ Hash-based | ✅ `RadixAttention` | ⚠️ Prompt cache | ⚠️ Prompt cache |
| **Compute Optimization**| Activation Channel Pruning | ⚠️ **Opt-in, costs quality**| ❌ None | ❌ None | ❌ None | ❌ None |
| | MoE Expert Offload | 🧪 **Components only; no MoE checkpoint run** | ✅ | ✅ | ✅ | ✅ |
| | Fused C++20 AVX2 SIMD Micro-Kernels | ✅ **Native** | ❌ | ❌ | ✅ Native | ✅ Native |
| | Triton 3.x Tensor Core GPU Kernels | ✅ **Native** | ✅ Native | ✅ Native | ❌ | ❌ |
| **Programmatic Workflows**| Python DSL with `fork()` + `join()` | ✅ **@turing.chain** | ❌ | ✅ SGLang DSL | ❌ | ❌ |
| | Prefix Cache Sharing across Branches | ❌ **Branches re-prefill** | ❌ | ✅ Radix | ❌ | ❌ |
| **Multi-Model Pipelines** | Elastic Single-GPU Multi-Model Manager | 🧪 **Placeholder models only** | ❌ (1 server/GPU) | ❌ (1 server/GPU) | ❌ (1 active) | ❌ (1 model) |
| | Fused Segment Pooling | 🧪 **Experimental kernel** | ❌ | ❌ | ❌ | ❌ |
| | Compute-Cost Adaptive Batching | 🧪 **Experimental** | ⚠️ Fixed padding | ⚠️ Fixed padding | ❌ | ❌ |
| **Serving & Gateway** | Continuous Batching Scheduler | ✅ **3-Lane QoS** | ✅ Iteration-level | ✅ Iteration-level | ❌ (FIFO) | ⚠️ Basic slot |
| | Multi-Node Distributed (TP + PP) | ⚠️ **Building blocks; not validated** | ✅ Native NCCL | ✅ Native NCCL | ❌ | ⚠️ RPC only |
| | OpenAI API (`/v1/chat`, `/v1/embeddings`) | ✅ **Native** | ✅ Native | ✅ Native | ✅ Native | ✅ Native |
| | Cross-Encoder Rerank (`/v1/rerank`) | ⚠️ **Placeholder (word overlap)** | ❌ (TEI needed) | ❌ (TEI needed) | ❌ | ❌ |
| | Structured Extract (`/v1/extract`) | ⚠️ **Placeholder (substring match)** | ❌ | ❌ | ❌ | ❌ |
| | Anthropic API (`/v1/messages` with SSE) | ✅ **Native** | ❌ (Proxy needed) | ⚠️ Partial | ❌ | ❌ |
| | Ollama REST API (`/api/*` Endpoints) | ✅ **Native** | ❌ | ❌ | ✅ Native | ❌ |
| | Kubernetes `llm-d` Router Token Render | ✅ `/render` + ZMQ | ❌ (Patch needed) | ❌ | ❌ | ❌ |
| | AI Traffic Management & Memory Watermarks | ✅ **Token-budget admission** | ⚠️ Queue limits | ⚠️ Queue limits | ❌ | ❌ |
| **Deployment & Binary** | Zero-Python Standalone Executable | ✅ **`turing-cli`** | ❌ (Python base) | ❌ (Python base) | ✅ Go/C++ daemon| ✅ `llama-cli` |
| **Structured Output** | JSON Schema & JSON Mode Enforcement | ✅ **Native + Repair** | ✅ Outlines / FSM | ✅ `xGrammar` | ⚠️ Format string | ✅ GBNF Grammars |
| | Native Tool & Function Calling | ✅ **Native** | ✅ Native | ✅ Native | ✅ Native | ⚠️ Custom parsing |
| **Agentic & Memory** | Cross-Model Representation Transfer ($W^*$)| 🧪 **Experimental**| ❌ (Re-prefills) | ❌ (Re-prefills) | ❌ (Re-prefills) | ❌ (Re-prefills) |
| | Hybrid Gated Softmax Anchoring | 🧪 **Toy recall probe only** | ⚠️ O(N) Softmax only| ⚠️ O(N) Softmax only| ⚠️ O(N) Softmax only| ⚠️ O(N) Softmax only|
| | Multi-Tenant LoRA Hot-Swap | 🧪 **Random adapters only**| ⚠️ Merge required | ⚠️ S-LoRA | ❌ | ❌ |
| | Speculative Decoding Parity Gate | 🧪 **Verifier only; not run on decoding** | ⚠️ PyTorch | ⚠️ PyTorch | ❌ | ⚠️ Basic |
| **Hardware Targets** | NVIDIA CUDA | ✅ Primary | ✅ Primary | ✅ Primary | ✅ Native | ✅ Native |
| | Apple Silicon Metal (MPS / Unified Memory) | ✅ Native | ⚠️ Experimental | ❌ | ✅ Native | ✅ Native |
| | CPU AVX2 / NEON Bare-Metal SIMD | ✅ Native (64-byte) | ⚠️ Slow CPU | ❌ | ✅ Native | ✅ Native |

---

## 🔍 Deep-Dive Systems Analysis

### 1. Ingestion: Direct Safetensors `mmap` vs. GGUF Conversion Pipeline
* **Ollama & llama.cpp**: Require offline conversion from PyTorch/Safetensors to GGUF format via `convert_hf_to_gguf.py` and quantization steps. New model architectures cannot run until conversion scripts are explicitly authored.
* **vLLM & SGLang**: Ingest Hugging Face Safetensors directly into GPU VRAM in full dense precision (FP16/BF16/FP8).
* **Turing Engine**: Dynamically resolves Hugging Face Hub repositories via `ModelResolver`. HuggingFace checkpoints load through transformers' `from_pretrained`; local safetensors and GGUF files are memory-mapped (`madvise(MADV_WILLNEED)`), with no offline conversion. FFN pruning is off by default.

---

### 2. Memory: int8 / int4 KV Cache
* **vLLM & SGLang**: Store full-dimensional KV states. Even with FP8 KV cache, a 32K context stream consumes ~5.0 GB VRAM.
* **Ollama & llama.cpp**: Offer fp16, q8_0 and q4_0 KV cache types.
* **Turing Engine**: Stores K and V as int8 per token and KV head (or int4 per 32 values). A 32K-context stream of a Llama-70B-shaped model (80 layers, 8 KV heads, head_dim 128) needs 5.16 GiB instead of 10.0 GiB, comparable to an FP8 cache, with no measurable quality cost on SmolLM2 135M–1.7B ([benchmarks](benchmarks.md#2-quality-on-real-weights)).

---

### 3. Compute: Subspace Channel Pruning vs. Dense Execution
* **vLLM, SGLang, Ollama & llama.cpp**: Compute all feed-forward network (FFN) channels across all layers regardless of token activation magnitude.
* **Turing Engine**: Can prune FFN channel tiles by weight norm (`--sparsity`). The pruned layer is faster because its matrices are smaller (2.32× at 57% for one random-weight layer of a 70B shape), but without retraining it is not lossless: at 50% next-token NLL rises by +6.8 to +15.8 nats on SmolLM2. It is off by default.

---

### 4. MoE Offloading

* **Turing Engine** has the pieces of an expert-offload path: an on-GPU LFRU expert slot cache with a hysteresis gate (`GPULRUExpertCache`), routing-history pinning (`RoutingHistoryTracker`) and a one-layer-ahead prefetcher (`OneLayerAheadExpertPrefetcher`). They are tested on synthetic routing traces only. The HuggingFace loader has no MoE path, no MoE checkpoint has been decoded, and the recorded prefetch run shows no latency hidden (1.0×). Earlier versions of this page quoted 18–50 tok/s for 320B MoE models on a 24 GB GPU and 1–5 tok/s for other engines; no script produced either figure.

---

### 5. Serving Gateway: Triple API Gateway vs. Single-Protocol Servers
* **vLLM & SGLang**: Native OpenAI API (`/v1/*`); Anthropic or Ollama endpoints require external reverse proxies.
* **llama.cpp**: Exposes a basic OpenAI-compatible `/v1` server.
* **Ollama**: Exposes its proprietary `/api/*` endpoints with an OpenAI compatibility wrapper.
* **Turing Engine**: Features a **Triple API Gateway in a single server instance**:
  - **OpenAI**: `/v1/chat/completions`, `/v1/completions` (with reasoning effort and streaming SSE).
  - **Anthropic**: `/v1/messages` (with native `thinking` blocks and token streaming).
  - **Ollama**: `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/embed`, `/api/pull`.
  - **Kubernetes llm-d**: `/render` prefix token hashing and ZeroMQ PUB port 5556.

---

### 6. Structured Outputs & Tool Calling
* **SGLang & vLLM**: Use FSM-based regex state machines (`xGrammar` / Outlines) to constrain logits token-by-token.
* **llama.cpp**: Uses GBNF grammars to enforce context-free grammar parsing.
* **Turing Engine**:
  - **Structured Outputs**: JSON-schema instructions injected into the prompt, JSON extraction from model output, and bracket/quote repair to recover JSON truncated by `max_tokens`. Output is not constrained token by token.
  - **Tool Calling**: OpenAI/Anthropic tool schemas injected into the prompt and tool calls parsed from the output.

---

### 7. Agentic Deliberation: Zero-Token $W^*$ Transfer vs. Re-Prefill
* **vLLM, SGLang, Ollama & llama.cpp**: In multi-agent pipelines (draft $\to$ target model handoff), the receiving model must re-tokenize and re-prefill the entire conversation history from scratch.
* **Turing Engine**:
  - Experimental: a closed-form ridge mapping ($W^* = (X^T X + \lambda I)^{-1} X^T Y$) translates key-value states from one model to another instead of re-prefilling text. It has been timed only on random tensors; transfer quality between real models has not been measured. (An earlier "7.85× faster" figure was a hard-coded multiplier.)
  - **Multi-Turn Clean-Base Lineage** (`CleanBaseLineageBuffer`) recomputes each turn's residual against the original receiver cache instead of chaining translations. Its effect on a real model has not been measured.

---

### 8. Cold Ingestion

* **Turing Engine** has a `madvise(MADV_WILLNEED)` readahead reader, an `io_uring` reader, `.tgate4` int4 files and a GPUDirect Storage (`cuFile`) path. None has a valid measurement yet: the ingest ladder's tier 3–5 timings were constants (`Tier 2 × 0.25`, 45.2 ms, a 0.042 ms floor), its GB/s divided the whole file by the time to read 10 tensors, and its readahead "4.83×" compared a cold page cache with a warm one.

---

## 🏆 Selection Guide: When to Choose Which

| If Your Use Case Is... | Recommended Engine | Rationale |
| :--- | :---: | :--- |
| **Multi-GPU Datacenter Cluster Serving** | **vLLM** | Mature multi-node tensor/pipeline parallelism across H100/A100 clusters. |
| **Complex Multi-Turn Prompt-Chaining Pipelines** | **SGLang** | Advanced RadixAttention tree-sharing and specialized SGLang DSL primitives. |
| **Minimal CLI Desktop Chat via GGUF** | **Ollama** | Single-command installation with pre-packaged local GGUF catalog. |
| **Standalone C/C++ Embedded Binary (Zero Python)** | **llama.cpp** | Minimal single-binary executable with GBNF grammar support for pure edge hardware. |
| **Research on exact-parity serving with a quantized KV cache (Llama-architecture, small models)** | **Turing Engine** | **HF-exact logits**, **int8 KV (−48%)**, **fused batched CUDA decode** measured on SmolLM2; Triple Gateway (OpenAI + Anthropic + Ollama), structured-output helpers, and experimental components not yet measured on models. |

