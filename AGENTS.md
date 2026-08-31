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
> **Invariant**: All 154 automated tests MUST pass with 0 warnings. Any newly introduced warning or test regression must be resolved immediately.



---

## 🏗️ Architecture & Codebase Invariants

### 1. Core Directories & Responsibilities
- **`csrc/`**: Bare-metal C++20 AVX2 SIMD headers and PyBind11 bindings (`pybind_bindings.cpp`). All allocations must be 64-byte aligned (`posix_memalign` / `aligned_alloc`). Native code must compile with `-fvisibility=hidden` and LTO.
- **`turing/core/`**: Algorithmic & mathematical primitives (closed-form $W^*$ KV transfer, Birkhoff hyper-connections, heterogeneous MoE routing, SVD KV paging, quadtree MRP speculative decoding).
- **`turing/kernels/`**: Triton GPU kernels (`triton_swiglu.py`, `triton_flash_tree.py`, `triton_w4a16.py`, `triton_recirculation.py`) and CUDA dispatch wrappers.
- **`turing/models/`**: Subspace model architectures (`causal_lm.py`), weight converters (`converter.py`), safetensors mmap loaders, and vLLM integration adapters.
- **`turing/serving/`**: Production continuous batching engine (`engine.py`), dual OpenAI + Anthropic `/v1/messages` server (`server.py`), and NIAH evaluator (`niah.py`).
- **`turing/demo/`**: Multi-agent deliberation coordinator (`agent_system.py`), dynamic environment model (`world_model.py`), epistemic uncertainty gate (`epistemic_gate.py`), and interactive CLI runner.

### 2. Coding Principles & Guidelines
1. **Surgical Modifications**: Touch only what is strictly necessary. Never perform broad cosmetic reformats across unrelated files.
2. **Deterministic Parity**: Any new SIMD intrinsic or Triton GPU kernel must have a reference Python/PyTorch numerical parity test verifying output tensor equivalence.
3. **No Hallucinated Jargon or Phantom Aliases**: Always use canonical names:
   - `MultiAgentCoordinator` (not `AstraAgentSystem`)
   - `DynamicEnvironmentModel` (not `ActiveInferenceWorldModel`)
   - `EntropyConfidenceTreePruner` (not `BAPHConfidenceTreePruner`)
   - `constraint_penalty` (not `free_energy`)
4. **Platform Compatibility**: Code must gracefully handle and auto-dispatch between NVIDIA CUDA (Triton), Apple Silicon Metal (MPS), and x86_64/ARM CPU (AVX2 SIMD fallback).
5. **License Compliance**: Respect Business Source License 1.1 terms in [`LICENSE`](LICENSE).
