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
# Run the entire test suite with strict warning escalation (all 667 tests MUST pass with 0 warnings on CUDA; 49 of them need CUDA and skip on Mac/CPU)
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
# Start the server (OpenAI /v1, Anthropic /v1/messages and Ollama /api/* on one port)
turing serve --model Qwen/Qwen2.5-7B-Instruct --port 8000

# Measurements that produce the committed results_*.json (scripts/l4_session.sh lists the GPU runs)
python scripts/eval_kv_quality.py --help
python scripts/benchmark_kv_decode.py --help
python scripts/eval_gguf_parity.py --help
```

---

## ☁️ Remote GCP GPU Execution (NVIDIA L4 24GB, temporary Spot VM)
- **No standing GPU VM.** `neollm-l4` is now a CPU-only `e2-small` running the user's `liberty` bots; never stop or resize it. `intutic-judge-l4` was deleted on 2026-11-14. Quota: `GPUS_ALL_REGIONS = 1`, on-demand L4 `0`, Spot L4 `3` in `us-central1`.
- **Workflow (needs the user's OK each time; it costs money):** create a temporary Spot L4 VM, run, copy results back, delete it.
  ```bash
  gcloud compute instances create turing-bench-l4 --zone us-central1-a --machine-type g2-standard-16 \
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
   - Each such sync stalls the CPU pipeline until the GPU catches up (unmeasured here; estimated at tens of microseconds per call). Speculative entropy and argmax must remain inside the fused Triton kernel (`triton_spec_entropy.py`) or in on-device tensor registers without CPU sync.

2. **Multi-Turn `clean_base` Invariant (Lineage Drift Collapse)**:
   - In multi-turn deliberation loops (`turing/core/lineage.py`), naive re-injection of translated KV residuals onto an already-drifted receiver cache makes the residual norms ($\|\Delta C_R\|_2$) compound turn after turn. Measured (`results_lineage_mps.json`, Qwen2.5-0.5B → 1.5B, α=1): the residual norm grows about 26× per turn (1,595 at turn 1, 5.7e9 at turn 6) and the receiver's WikiText NLL goes from 2.43 to 9.14 at turn 2; `clean_base` stays at 2.43 every turn.
   - Always recompute residuals against the frozen, original receiver cache (`clean_base`). Never chain residual translations on mutated caches.

3. **Strict PyTorch Deprecation Sensitivity (`-W error`)**:
   - Because we enforce `python -W error -m pytest -v`, any modern PyTorch warning (e.g., omitting `indexing='ij'` in `torch.meshgrid`, using legacy `torch.cuda.amp.autocast()`, or NumPy scalar conversions) is treated as a fatal test failure.
   - Always use modern, explicit PyTorch APIs (`torch.amp.autocast('cuda')`, explicit meshgrid indexing) to prevent warning escalation failures.

4. **Remote GCP GPU VRAM Leakage & Orphaned PIDs**:
   - When terminating or interrupting remote benchmark scripts on the temporary GCP L4 VM, Python or PyTorch background worker processes occasionally hold onto VRAM allocations.
   - Before running memory-intensive benchmarks or serving, always verify free VRAM via `ssh ... "nvidia-smi"` and terminate any orphaned Python processes (`kill -9 <PID>`).

5. **Triple-File Version Alignment Invariant**:
   - The release version string (e.g., `1.5.0`) must always be updated simultaneously across:
     1. `setup.py` (`version="..."`)
     2. `pyproject.toml` (`version = "..."`)
     3. `turing/__init__.py` (`__version__ = "..."`)

