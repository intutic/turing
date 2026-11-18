# Agent Guidelines for Turing Engine

This document provides instructions, constraints, and architecture invariants for AI coding assistants (Claude Code, Cursor, Copilot, Antigravity, Codex, Aider) working on the **Turing Engine** codebase.

---

## 🛠️ Build & Test Commands

### 1. Environment & Setup
```bash
# Editable install with dev dependencies
pip install -e ".[dev]"

# Build C++20 AVX2 SIMD PyBind11 native extension in-place
python setup.py build_ext --inplace
```

### 2. Running Tests
```bash
# Run the entire test suite with strict warning escalation
python -W error -m pytest -v

# Run a specific test file
python -W error -m pytest tests/test_cross_model_kv_transfer.py -v

# Run quick quiet validation
python -W error -m pytest -q
```
> **Invariant**: All 705 automated tests MUST pass with 0 warnings on CUDA (on a Mac the CUDA-only ones skip). Any newly introduced warning or test regression must be resolved immediately.




---

## 🏗️ Architecture & Codebase Invariants

### 1. Core Directories & Responsibilities
- **`csrc/`**: C++20 SIMD headers (AVX2/FMA on x86_64, NEON on arm64; `turing_simd.hpp`), PyBind11 bindings (`pybind_bindings.cpp`), GGUF reader and exact dequantizer (`turing_gguf_cpp.hpp`), `ParallelReader` (`turing_parallel_reader.hpp`, io_uring or `pread`), cuFile loader (`turing_cufile.hpp`), and the standalone engine: BPE tokenizer (`turing_tokenizer_cpp.hpp`), forward pass (`turing_model_cpp.hpp`, Llama-layout GGUF, optional int8 KV), HTTP server (`turing_http_server.hpp`) and `turing-cli` (`turing_main.cpp`). The engine's weights, KV cache and SIMD scratch buffers are 64-byte aligned. Native code must compile with `-fvisibility=hidden` and LTO.
- **`turing/core/`**: `kv_cache.py` (serving KV cache: one paged pool per model (`KVPool`), a page table per request (`KVCache`, copy-on-write forks), `native`/`int8`/`int4` storage, Quest page bounds) and `rope.py` (default, linear, dynamic, YaRN, LongRoPE, Llama 3, proportional). Library code outside the serving path, none measured on a model:
  - `cross_model_kv.py`: Closed-form $W^*$ ridge KV transfer with RoPE content decoupling and null-space projection.
  - `hybrid_memory.py`, `hybrid_attention.py`: Linear recurrence with anchor attention ($S_t = b_t S_{t-1} + K_t^T V_t$); needs a model trained with hybrid layers.
  - `lineage.py`: Multi-turn `clean_base` lineage controller.
  - `kslot_pooling.py`: $k$-slot attention summary pooling and gated zero-identity heads.
  - `heterogeneous_moe.py`, `expert_cache.py`, `expert_prefetch.py`: MoE expert offload (`HeterogeneousMoERunner` runs a model's own experts from host memory, hot experts in `GPULRUExpertCache`).
  - `mhc.py`: Sinkhorn-projected (doubly stochastic) hyper-connections.
  - `speculation.py`: Untrained draft heads (Matryoshka, quadtree) and tree pruning; serving speculates by prompt lookup instead.
  - `paging.py`, `elastic_memory.py`: Older paged pools and an expert-cache/KV budget balancer.
- **`turing/dsl/`**: Workflow primitives (`@chain`, `gen`, `fork`, `join`, `select`, `constrain`); `fork()` prefills the shared prefix once and branches continue it on the engine.
- **`turing/kernels/`**: Triton GPU kernels: `triton_kv_decode.py` (fused batched KV append + split-K paged decode, the serving path), `triton_w4a16.py` (int4 GEMM), `triton_sparse_gemv.py` (activation sparsity), and library kernels (`triton_swiglu.py`, `triton_flash_tree.py`, `triton_recirculation.py`, `triton_hybrid_anchor_fuse.py`, `triton_spec_entropy.py`, `triton_kslot_pool.py`, …) with CUDA dispatch wrappers.
- **`turing/models/`**: `causal_lm.py` (`SubspaceCausalLM`, config-driven for 16 HF model types, CUDA-graph decode), `hf_loader.py` (`FAMILIES` allow-list, strict meta-device loading), `gguf_loader.py` / `gguf_tokenizer.py` (llama.cpp GGUF), `quant.py` (GPTQ/RTN, `QuantLinear`, `.tgate4`), `streaming_loader.py` (layer-by-layer `turing convert`), `activation_sparsity.py` (TEAL), `adapters.py` (PEFT multi-LoRA), `tensor_parallel.py` (TP sharding and PP stages), `architecture_registry.py`, `resolver.py`, `safetensors_mmap.py`, `converter.py`, `vllm_adapter.py`.
- **`turing/serving/`**: `server.py` (OpenAI `/v1`, Anthropic `/v1/messages` and Ollama `/api/*` on one app), `generation.py` (shared generation core: chat templates, tools, reasoning, constrained decoding, stop strings, usage), `engine.py` (continuous batching, prefix cache via `prefix_cache.py`, prompt-lookup speculation, multi-LoRA), `traffic.py` (admission control and 3-lane QoS), `spec_gate.py`, `structured.py` (llguidance), `tools.py`, `reasoning.py`, `embeddings.py`, `extract.py`, `pull.py`, `kv_events.py` (llm-d), `distributed.py` (TP/PP driver), `kv_transfer.py` / `disagg_sidecar.py` (prefill/decode over TCP), `accuracy_eval.py` (GSM8K) and `niah.py` (pass-key retrieval).
- **`turing/demo/`**: Multi-agent deliberation coordinator (`agent_system.py`), dynamic environment model (`world_model.py`), epistemic uncertainty gate (`epistemic_gate.py`), and interactive CLI runner.

### 2. Coding Principles & Guidelines
1. **Surgical Modifications**: Touch only what is strictly necessary. Never perform broad cosmetic reformats across unrelated files.
2. **Deterministic Parity**: Any new SIMD intrinsic or Triton GPU kernel must have a reference Python/PyTorch numerical parity test verifying output tensor equivalence.
3. **No Hallucinated Jargon or Phantom Aliases**: Always use canonical names:
   - `MultiAgentCoordinator` (not `AstraAgentSystem`)
   - `DynamicEnvironmentModel` (not `ActiveInferenceWorldModel`)
   - `EntropyConfidenceTreePruner` (not `BAPHConfidenceTreePruner`)
   - `CleanBaseLineageBuffer` (not `PristineCacheRegistry`)
   - `KSlotCachePooler` (not `SlotSummaryCompressor`)
   - `constraint_penalty` (not `free_energy`)
4. **Platform Compatibility**: Code must gracefully handle and auto-dispatch between NVIDIA CUDA (Triton), Apple Silicon Metal (MPS), and x86_64/ARM CPU (AVX2/NEON SIMD fallback).
5. **License Compliance**: Respect Business Source License 1.1 terms in [`LICENSE`](LICENSE).
