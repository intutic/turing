# Turing Engine (jflux) — Claude Code Guide & Directives

This document provides instructions, constraints, and architecture invariants for AI coding assistants (Claude Code, Cursor, Copilot, Antigravity, Codex, Aider) working on the **Turing Engine** codebase.

---

## 🛠️ Build, Test & Serving Commands

### 1. Environment & Setup
```bash
# Editable install with dev dependencies
pip install -e ".[dev]"

# Build C++20 AVX2 SIMD PyBind11 native extension in-place
python setup.py build_ext --inplace
```

### 2. Testing & Quality Verification (Strict 0-Warning Invariant)
```bash
# Run the entire test suite with strict warning escalation (All 371 tests MUST pass with 0 warnings on CUDA; 18 of them need CUDA and skip on Mac/CPU)
python -W error -m pytest -v

# Run a specific test file
python -W error -m pytest tests/test_cross_model_kv_transfer.py -v

# Run quick quiet validation
python -W error -m pytest -q

# Verify documentation build with zero warnings and zero broken links
mkdocs build --strict
```

### 3. Serving & Benchmarks
```bash
# Start local serving server (Dual OpenAI + Anthropic /v1/messages)
turing serve --model llama-3.1-70b --port 8000

# Run standard benchmarks
python scripts/run_all_benchmarks.py

# Run exhaustive 20-suite master benchmark matrix across all 26 documented sections
python scripts/run_all_documented_benchmarks_exhaustive.py
```

---

## ☁️ Remote GCP GPU Execution (NVIDIA L4 24GB, temporary Spot VM)
- **No standing GPU VM.** `neollm-l4` is now a CPU-only `e2-small` running the user's `liberty` bots; never stop or resize it. `intutic-judge-l4` was deleted on 2026-11-14. Quota: `GPUS_ALL_REGIONS = 1`, on-demand L4 `0`, Spot L4 `3` in `us-central1`.
- **Workflow (needs the user's OK each time; it costs money):** create a temporary Spot L4 VM, run, copy results back, delete it.
  ```bash
  gcloud compute instances create turing-bench-l4 --zone us-central1-a --machine-type g2-standard-4 \
    --provisioning-model SPOT --instance-termination-action STOP --max-run-duration 6h \
    --image-family pytorch-2-9-cu129-ubuntu-2204-nvidia-580 --image-project deeplearning-platform-release \
    --boot-disk-size 150GB --maintenance-policy TERMINATE --metadata install-nvidia-driver=True \
    --labels purpose=turing-bench,owner=ishangupta,temporary=true
  # first boot: sudo apt-get install -y build-essential python3.10-dev; sudo pip uninstall -y torchaudio
  # (the image's torchaudio fails to load and transformers 5 imports it); then pip install -e '.[dev]'
  gcloud compute instances delete turing-bench-l4 --zone us-central1-a --quiet   # when done
  ```
  Spot L4 capacity stocks out; try zones `-a`, `-b`, `-c`.
- **Access is IAP-only**: the firewall allows tcp:22 only from IAP (`35.235.240.0/20`), so direct `ssh <ip>` times out. Never open port 22 to `0.0.0.0/0`.
- **SSH Command**:
  ```bash
  gcloud compute ssh ishangupta@turing-bench-l4 --zone us-central1-a --tunnel-through-iap --ssh-key-file ~/.ssh/google_compute_engine
  ```
- **Code Sync (Rsync over IAP)**:
  ```bash
  rsync -avz --exclude '.git' --exclude '__pycache__' --exclude 'build' --exclude '*.so' -e "ssh -i ~/.ssh/google_compute_engine -o HostKeyAlias=turing-bench-l4 -o ProxyCommand='gcloud compute start-iap-tunnel turing-bench-l4 22 --listen-on-stdin --zone us-central1-a --verbosity=warning'" ./ ishangupta@turing-bench-l4:~/jflux/
  ```
- **Long runs**: start them with `nohup` on the VM and poll a status file; IAP SSH sessions drop.

---

## 🏗️ Architecture & Core Responsibilities

- **`csrc/`**: Bare-metal C++20 AVX2 SIMD headers, PyBind11 native bindings (`pybind_bindings.cpp`), standalone GGUF parser (`turing_gguf_cpp.hpp`), BPE tokenizer (`turing_tokenizer_cpp.hpp`), forward pass engine (`turing_model_cpp.hpp`), embedded socket server (`turing_http_server.hpp`), and `turing-cli` binary. All allocations must be 64-byte aligned (`posix_memalign` / `aligned_alloc`). Native code must compile with `-fvisibility=hidden` and LTO.
- **`turing/core/`**: Algorithmic & mathematical primitives:
  - `cross_model_kv.py`: Closed-form $W^*$ Ridge KV transfer, RoPE content decoupling, and SVD null-space projection.
  - `hybrid_memory.py`: Linear recurrent state blending with anchor attention ($S_t = b_t S_{t-1} + K_t^T V_t$).
  - `lineage.py`: Multi-turn `clean_base` lineage controller (prevents multi-turn representation drift collapse).
  - `kslot_pooling.py`: $k$-Slot symmetric attention summary pooling and gated zero-identity heads.
  - `expert_cache.py`: Hot-expert LFRU cache, routing heat trackers, and atomic routing frequency matrices.
  - `birkhoff.py`: Birkhoff hyper-connections and doubly stochastic routing matrices.
  - `moe.py` & `heterogeneous_moe.py`: Heterogeneous MoE routing and dynamic dispatch.
  - `paging.py` & `radix_svd.py`: SVD INT8 KV paging and Spectral Radix prefix cache trees.
  - `speculation.py`: Quadtree MRP speculative decoding and Matryoshka sliced draft heads.
  - `kv_cache.py`: Serving KV cache: one paged pool per model (`KVPool`), a page table per request (`KVCache`, copy-on-write forks), `native`/`int8`/`int4` storage, Quest page bounds.
- **`turing/dsl/`**: High-level programmatic workflow primitives (`@chain`, `gen`, `fork`, `join`, `select`, `constrain`) with prefix KV cache sharing and structured parsing executors.
- **`turing/kernels/`**: Triton GPU kernels (`triton_swiglu.py`, `triton_flash_tree.py`, `triton_w4a16.py`, `triton_recirculation.py`, `triton_hybrid_anchor_fuse.py`, `triton_spec_entropy.py`, `triton_kslot_pool.py`, `triton_kv_decode.py` fused batched KV append + decode) and CUDA dispatch wrappers.
- **`turing/models/`**: Subspace model architectures (`causal_lm.py`), native binary GGUF reader & dequantizer (`gguf_loader.py`, `gguf_tokenizer.py`), `.tgate` weight converters (`converter.py`), safetensors mmap loaders, HuggingFace loaders, tensor/pipeline parallelism (`tensor_parallel.py`), and vLLM integration adapters.
- **`turing/serving/`**: Continuous batching engine (`engine.py`), elastic multi-model serving (`multi_model.py`), distributed multi-GPU & multi-node driver (`distributed.py`), dual OpenAI + Anthropic `/v1/messages` server (`server.py`), AI traffic management & 3-lane QoS scheduling (`traffic.py`), concurrency-adaptive speculation gating (`spec_gate.py`), and NIAH evaluator (`niah.py`).
- **`turing/demo/`**: Multi-agent deliberation coordinator (`agent_system.py`), dynamic environment model (`world_model.py`), epistemic uncertainty gate (`epistemic_gate.py`), and interactive CLI runner.

---

## 🚨 Critical Engineering Rules & Invariants

1. **Zero Mocks & Empirical Truth**: Never fabricate benchmark figures or use mock objects for hardware validation. All telemetry must be physically measured on real silicon.
2. **Deterministic Parity**: Any new Triton GPU kernel or SIMD intrinsic must have a reference Python/PyTorch numerical parity test verifying output tensor equivalence (`atol <= 1e-3`).
3. **Zero-Copy Invariants**: Preserve memory mapping (`madvise(MADV_WILLNEED)`) and avoid unnecessary host-device DRAM copies.
4. **Canonical Naming (No Hallucinated Jargon or Phantom Aliases)**:
   - `MultiAgentCoordinator` (not `AstraAgentSystem`)
   - `DynamicEnvironmentModel` (not `ActiveInferenceWorldModel`)
   - `EpistemicUncertaintyGate` (canonical epistemic gate)
   - `EntropyConfidenceTreePruner` (not `BAPHConfidenceTreePruner`)
   - `CleanBaseLineageBuffer` (not `PristineCacheRegistry`)
   - `KSlotCachePooler` (not `SlotSummaryCompressor`)
   - `constraint_penalty` (not `free_energy`)
5. **Platform Compatibility & Auto-Dispatch**: Code must gracefully handle and auto-dispatch between NVIDIA CUDA (Triton), Apple Silicon Metal (MPS), and x86_64/ARM CPU (AVX2/NEON SIMD fallback).
6. **Surgical Modifications**: Touch only what is strictly necessary. Never perform broad cosmetic reformats across unrelated files; preserve existing comments, docstrings, and contracts.
7. **License Compliance**: Respect Business Source License 1.1 terms in [`LICENSE`](LICENSE).
8. **Git Commit Sign-Off**: Always append `-m "Signed-off-by: Ishan Gupta <ishangupta.ds@gmail.com>"` to git commits.

---

## 🧠 Hard-Won Lessons & Low-Level Gotchas

1. **Zero-Sync Drafter Rule (Never call `.item()` on GPU Tensors)**:
   - In speculative drafting and entropy gating (`turing/core/speculation.py`), calling `.item()` (e.g., `entropy = -(probs * log_probs).sum().item()`) forces a synchronous device-to-host PCIe round-trip.
   - This creates a 15–25µs CPU pipeline bubble on every draft step. Speculative entropy and argmax must remain inside the fused Triton kernel (`triton_spec_entropy.py`) or in on-device tensor registers without CPU sync.

2. **Multi-Turn `clean_base` Invariant (Lineage Drift Collapse)**:
   - In multi-turn deliberation loops (`turing/core/lineage.py`), naive re-injection of translated KV residuals onto an already-drifted receiver cache causes residual norms ($\|\Delta C_R\|_2$) to explode (>20,000 by turn 6), destroying associative recall.
   - Always recompute residuals against the frozen, original receiver cache (`clean_base`). Never chain residual translations on mutated caches.

3. **Strict PyTorch Deprecation Sensitivity (`-W error`)**:
   - Because we enforce `python -W error -m pytest -v`, any modern PyTorch warning (e.g., omitting `indexing='ij'` in `torch.meshgrid`, using legacy `torch.cuda.amp.autocast()`, or NumPy scalar conversions) is treated as a fatal test failure.
   - Always use modern, explicit PyTorch APIs (`torch.amp.autocast('cuda')`, explicit meshgrid indexing) to prevent warning escalation failures.

4. **Remote GCP GPU VRAM Leakage & Orphaned PIDs**:
   - When terminating or interrupting remote benchmark scripts on the temporary GCP L4 VM, Python or PyTorch background worker processes occasionally hold onto VRAM allocations.
   - Before running memory-intensive benchmarks or elastic multi-model serving, always verify free VRAM via `ssh ... "nvidia-smi"` and terminate any orphaned Python processes (`kill -9 <PID>`).

5. **Triple-File Version Alignment Invariant**:
   - The release version string (e.g., `1.5.0`) must always be updated simultaneously across:
     1. `setup.py` (`version="..."`)
     2. `pyproject.toml` (`version = "..."`)
     3. `turing/__init__.py` (`__version__ = "..."`)

