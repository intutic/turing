# Turing Engine (jflux) — Claude Code Directive

## 🛠️ Build, Test & Verification Commands
- **Install Dev Dependencies**: `pip install -e ".[dev]"`
- **Build C++20 AVX2 SIMD Extension**: `python setup.py build_ext --inplace`
- **Run Full Test Suite (Strict 0-Warning Invariant)**: `python -W error -m pytest -v` (Must pass all 327/327 tests with 0 warnings)
- **Run Single Test**: `python -W error -m pytest tests/<test_file>.py -v`
- **Verify Docs Build**: `mkdocs build --strict`
- **Run Exhaustive Master Benchmarks**: `python scripts/run_all_documented_benchmarks_exhaustive.py`

## ☁️ Remote GCP GPU Execution (NVIDIA L4 24GB, CUDA 13.0)
- **Host**: `ishangupta@34.171.37.217` (VM: `neollm-l4`, zone: `us-central1-a`)
- **SSH Command**: `ssh -i ~/.ssh/google_compute_engine ishangupta@34.171.37.217`
- **Code Sync (Rsync)**:
  `rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'build' --exclude '*.so' -e "ssh -i ~/.ssh/google_compute_engine" ./ ishangupta@34.171.37.217:~/jflux/`
- **Remote Benchmark Run**:
  `ssh -i ~/.ssh/google_compute_engine ishangupta@34.171.37.217 "cd ~/jflux && python3 scripts/run_all_documented_benchmarks_exhaustive.py"`

## 🏗️ Architecture & Component Boundaries
- `csrc/`: Bare-metal C++20 AVX2 SIMD headers, PyBind11 native bindings, 64-byte aligned allocs (`posix_memalign`), zero-copy mmap, and standalone GGUF parser.
- `turing/core/`: Algorithmic & mathematical primitives:
  - `cross_model_kv.py`: Closed-form W* KV ridge projection.
  - `hybrid_memory.py`: Linear recurrent state blending with anchor attention.
  - `lineage.py`: Multi-turn clean-base lineage controller (prevents drift collapse).
  - `kslot_pooling.py`: k-Slot symmetric attention summary pooling.
  - `expert_cache.py`: Hot-expert LFRU cache and atomic routing frequency matrices.
  - `speculation.py`: Dynamic tree speculative decoding and Matryoshka heads.
- `turing/kernels/`: Triton GPU kernels (`triton_swiglu.py`, `triton_flash_tree.py`, `triton_hybrid_anchor_fuse.py`, `triton_spec_entropy.py`, `triton_kslot_pool.py`).
- `turing/models/`: Subspace causal LM architectures, binary GGUF reader, safetensors mmap loaders.
- `turing/serving/`: Continuous batching engine (`engine.py`), elastic multi-model serving (`multi_model.py`), 3-lane QoS traffic management (`traffic.py`), spec gating (`spec_gate.py`).
- `turing/dsl/`: Workflow programmatic primitives (`@chain`, `gen`, `fork`, `join`, `select`, `constrain`).

## 🚨 Critical Engineering Rules & Invariants
1. **Zero Mocks**: Never fabricate benchmark figures or use mock objects for hardware validation. All telemetry must be physically measured.
2. **Numerical Parity**: Any new Triton GPU kernel or SIMD intrinsic must have an equivalent PyTorch test validating numerical parity (`atol <= 1e-3`).
3. **Canonical Naming**: Never invent aliases. Use:
   - `MultiAgentCoordinator` (not `AstraAgentSystem`)
   - `DynamicEnvironmentModel` (not `ActiveInferenceWorldModel`)
   - `EntropyConfidenceTreePruner` (not `BAPHConfidenceTreePruner`)
   - `CleanBaseLineageBuffer` (not `PristineCacheRegistry`)
   - `KSlotCachePooler` (not `SlotSummaryCompressor`)
   - `constraint_penalty` (not `free_energy`)
4. **Platform Auto-Dispatch**: Auto-dispatch cleanly across NVIDIA CUDA (Triton), Apple Silicon Metal (MPS), and x86_64/ARM CPU (AVX2/NEON SIMD).
5. **Surgical Modifications**: Never execute cosmetic reformats across unrelated files; preserve existing comments, docstrings, and contracts.
6. **Git Sign-Off**: Always append `-m "Signed-off-by: Ishan Gupta <ishangupta.ds@gmail.com>"` to git commits.
