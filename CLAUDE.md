# Claude Code Guide — Turing Engine

## Primary Commands
- **Build Extension**: `python setup.py build_ext --inplace`
- **Run All Tests**: `python -W error -m pytest -v` (Strict warning escalation, 82/82 passing)
- **Run Single Test**: `python -W error -m pytest tests/<test_file>.py -v`
- **Start Serving Server**: `turing serve --model llama-3.1-70b --port 8000`
- **Run Benchmarks**: `python scripts/run_all_benchmarks.py`

## Architecture Quick Reference
- `csrc/`: C++20 AVX2 SIMD headers (64-byte aligned, zero-copy mmap, pybind11)
- `turing/core/`: Algorithmic primitives ($W^*$ Ridge KV transfer, mHC Birkhoff, MoE dispatcher, SVD paging)
- `turing/kernels/`: Triton GPU kernels (SwiGLU, W4A16 GEMM, FlashTree) & CUDA dispatch
- `turing/models/`: `SubspaceCausalLM`, `.tgate` converters, HuggingFace loaders
- `turing/serving/`: Continuous batching engine, OpenAI + Anthropic `/v1/messages` server
- `turing/demo/`: `MultiAgentCoordinator`, `DynamicEnvironmentModel`, `EpistemicUncertaintyGate`

## Guidelines & Constraints
- **Test Integrity**: Never lower warning escalation or disable strict checks.
- **Canonical Naming**: Use `MultiAgentCoordinator`, `DynamicEnvironmentModel`, `EntropyConfidenceTreePruner`, `constraint_penalty`.
- **Zero-Copy Invariants**: Preserve memory mapping (`madvise(MADV_WILLNEED)`) and avoid unnecessary host-device DRAM copies.
- **Surgical Edits**: Keep changes minimal, focused, and directly tied to the requested task.
