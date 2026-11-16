# Turing Engine: System Architecture & Technical Specification

This document provides an in-depth architectural blueprint of the **Turing Engine** serving runtime, detailing its 4-layer compute stack, memory hierarchy, mathematical foundations, and hardware kernel mechanics.

> **What is measured.** Real-weight quality, int8/int4 KV-cache decode throughput and single-layer pruning timings are in [docs/benchmarks.md](docs/benchmarks.md). Everything else on this page is experimental: the components have unit tests and, at most, micro-benchmarks on random data. The speedups and rates earlier versions of this page listed were removed because none was measured on a model and several were constants or formulas ([claims audit](docs/claims-audit.md)).

---

## 🏛️ High-Level System Overview

```
 ┌───────────────────────────────────────────────────────────────────────────────────┐
 │                                PRODUCTION SERVING LAYER                           │
 │  • Continuous Batching Scheduler (16–64 streams)                                  │
 │  • Triple API Gateway: OpenAI (/v1), Anthropic (/v1/messages), Ollama (/api/*)    │
 │  • Kubernetes llm-d Prefix-Cache Aware Router Integration & ZMQ Cache Events       │
 │  • Multi-Agent Deliberation Coordinator & Dynamic Environment Model               │
 └────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────▼──────────────────────────────────────────┐
 │                               CORE ALGORITHMIC LAYER                              │
 │  • Closed-Form Cross-Model KV Cache Transfer (W* Ridge Normal Equations)          │
 │  • Manifold-Constrained Hyper-Connections (mHC Doubly-Stochastic Birkhoff Mixing) │
 │  • Bandwidth-Adaptive CPU–GPU MoE Dispatcher & GPU LRU Slot Cache                 │
 │  • Hierarchical SVD INT8 Sequence Pager (Huge 512 / Med 64 / Small 16 Pages)      │
 │  • Frontier Speculative Decoding (Subspace-EAGLE3, DFlash O(1), Dynamic Tree)     │
 └────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────▼──────────────────────────────────────────┐
 │                            HARDWARE COMPUTE & KERNEL LAYER                        │
 │  • SRAM-Fused SwiGLU Triton Kernel                                                │
 │  • Packed INT4 W4A16 Tensor Core GEMM (Zero DRAM Write-Back)                      │
 │  • Flash-Tree-Attention Triton DAG Verification                                   │
 │  • 2D-Tiled Subspace Recirculation Tensor Core Engine                             │
 └────────────────────────────────────────┬──────────────────────────────────────────┘
                                          │
 ┌────────────────────────────────────────▼──────────────────────────────────────────┐
 │                           BARE-METAL C++20 SIMD LAYER                             │
 │  • 64-Byte Aligned AVX2 Vector Intrinsics (_mm256_fmadd_ps, _mm256_load_ps)       │
 │  • Zero-Copy Memory-Mapped SafeTensors & .tgate Loader (madvise MADV_WILLNEED)    │
 │  • C++ Block-Paged Selective Attention Engine & Intrusive LRU Slot Tables         │
 └───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Layer 1: Bare-Metal C++20 AVX2 SIMD & Memory Subsystems (`csrc/`)

### 1.1 AVX2 Sparse Pointer-Skipping GEMV (`csrc/turing_simd.hpp`)
- **64-Byte Alignment**: Allocations are aligned to cache-line boundaries via `posix_memalign` / `aligned_alloc`.
- **Active Channel Bitmask**: Channels are partitioned into 256-wide tiles controlled by a 32-bit active mask. When a tile is inactive ($0$), pointer arithmetic skips its 256 multiply-accumulate operations.
- **Vector Intrinsics**: Utilizes fused multiply-accumulate (`_mm256_fmadd_ps`), unaligned/aligned loads (`_mm256_load_ps`), and INT8 dot products (`_mm256_maddubs_epi16`).

### 1.2 Zero-Copy Memory-Mapped Weight Runtime (`csrc/turing_mmap.hpp`)
- **OS-Level Mapping**: Bypasses userspace memory copies via direct `mmap(PROT_READ, MAP_SHARED)`.
- **Kernel Prefetching**: Applies `madvise(MADV_WILLNEED)` to preload weight segments into the OS page cache asynchronously during cold starts.
- **`.tgate` Container Format**: Stores structured bitmask offset tables and quantized INT4/INT8 scales for direct zero-overhead pointer indexing.

### 1.3 Native C++ Paged Attention & Radix Forest (`csrc/turing_paged_attention.hpp`, `csrc/turing_radix_trie.hpp`)
- Evaluates attention exclusively across populated logical page slots (`active_page_mask`), skipping empty pages.
- Caches prefix branch projection bases in a cache-aligned Radix Tree for instantaneous prompt prefix cache hits.

### 1.4 Single-Step AVX2 Linear Recurrence (`csrc/turing_linear_recurrence.hpp`)
- Fuses $S_t = \alpha S_{t-1} + V_t K_t^T$ and $O_t = S_t Q_t$ using AVX2 `_mm256_fmadd_ps` for zero-allocation single-step token decode.

### 1.5 Zero-Copy SVD Wire Quantizer & Codec (`csrc/turing_svd_wire_codec.hpp`)
- Fuses SVD subspace projection ($K \cdot U$), absolute max reduction, symmetric INT8 quantization, and binary byte serialization into a single pass.

### 1.6 Native Deterministic Token Block Hasher (`csrc/turing_fast_hash.hpp`)
- Direct non-cryptographic 64-bit avalanche hashing (xxHash64 / Murmur3 mix) operating directly over raw `const uint32_t*` token buffers with 0 heap allocations.

### 1.7 DFlash 1D Dilated Depthwise Causal Convolution (`csrc/turing_dilated_conv1d.hpp`)
- Vectorized 1D Dilated Depthwise Causal Convolution with circular ring-buffer history window (0-copy, 0-transposition).

### 1.8 Fused Base GEMV + LoRA Rank-8 Contraction (`csrc/turing_lora_gemv.hpp`)
- Fuses Base Subspace GEMV + LoRA Rank-8 contraction directly in CPU registers without intermediate DRAM writes.

### 1.9 1-Pass AVX2 Residual Outlier Extraction (`csrc/turing_residual_outlier.hpp`)
- In-register `_mm256_max_ps` and horizontal bitonic reduction for coordinate outlier extraction ($O(d)$).

### 1.10 Lock-Free Atomic Compare-And-Swap Elastic Budget Controller (`csrc/turing_elastic_budget.hpp`)
- Lock-free atomic Compare-And-Swap (`ElasticBudgetController`) memory budget rebalancer.

### 1.11 4-Stream AVX2 Vectorized Birkhoff mHC Step (`csrc/turing_mhc_simd.hpp`)
- Vectorized 4-stream Pre-mapping + Res-mapping ($4 \times 4$ Birkhoff matrix) + Post-mapping.

### 1.12 Native C++20 AVX2 Prefix Token Hasher (`csrc/turing_prefix_router.hpp`)
- Direct FNV-1a / xxHash64 hashing over raw `const int32_t*` token buffers with 0 Python interpreter boxing.

### 1.13 Vectorized Speculative Parity Verifier (`csrc/turing_spec_verifier.hpp`)
- Vectorized `_mm256_cmpeq_epi32` token stream comparator comparing 8 tokens per 256-bit compare.

### 1.14 Zero-Copy Tensor Checksum Scanner (`csrc/turing_fast_hash_tensor.hpp`)
- Scans raw tensor memory pointers directly without allocating Python numpy arrays.

### 1.15 AVX2 SIMD Block Dequantizer (`csrc/turing_gguf_simd.hpp`)
- 256-bit AVX2 SIMD block dequantizers for `Q4_0`, `Q4_1`, `Q8_0`, `Q4_K_M`, `FP16`, and `BF16` with packed 32-value nibble interleaving and hardware `_mm256_cvtph_ps` floating-point conversions.

### 1.16 Thread-Safe Radix-SVD Forest (`csrc/turing_radix_trie.hpp`)
- Thread-safe `std::shared_mutex` read-write locked Radix Trie index supporting concurrent token lookup, prefix tree insertion, and immutable semantic anchor registration.

### 1.17 Native LFRU Expert Cache Fast (`csrc/turing_lru_cache.hpp`)
- Contiguous 64-byte aligned slots with composite scoring `(heat << 16) | (65535 - age)`, bitmask key pinning, and $25\%+4$ hysteresis promotion gate.

### 1.18 Native Streaming Reasoning Tag Filter (`csrc/turing_reasoning_scanner.hpp`)
- Zero-copy streaming state machine parsing `<think>...</think>` tokens into reasoning/content tuples.

### 1.19 Fast Safetensors Header Parser (`csrc/turing_safetensors_fast_header.hpp`)
- Zero-allocation SIMD metadata scanner for fast Safetensors JSON header parsing and offset extraction.

### 1.20 Direct Memory Pointer Tensor Hasher (`csrc/turing_fast_hash_tensor.hpp`)
- 256-bit AVX2 SIMD pointer scanning without allocating intermediate Python numpy bytes.

### 1.21 Native O(1) AI Traffic Admission Manager (`csrc/turing_traffic_manager.hpp`)
- Lock-free atomic byte tracking for admission control and QoS lane backpressure.

---

## 2. Layer 2: Core Algorithmic Engines (`turing/core/`)

### 2.1 Closed-Form Cross-Model KV Transfer ($W^*$) (`turing/core/cross_model_kv.py`)
- **Mathematical Formulation**: Intended to replace expensive quadratic prompt prefill on massive target models (e.g. 70B) by computing representations on an 8B model and mapping them in closed form:
  ```math
  W^* = \left( X_{\mathrm{src}}^\top X_{\mathrm{src}} + \lambda I \right)^{-1} X_{\mathrm{src}}^\top Y_{\mathrm{tgt}}
  ```
- **RoPE Position Inversion**: Rotary position embeddings are decoupled prior to projection ($\mathbf{k}_{\text{content}} = R_{\Theta}^{-1}(t)\mathbf{k}_{\text{RoPE}}$) and re-applied afterwards.
- **Null-Space Constraint**: Residual mapping errors are strictly constrained to the null space of target query singular vectors ($\Pi_{\text{null}} = I - V_r V_r^\top$).
- **Status**: experimental. No real model pair has been run through it, and the TTFT speedups earlier versions listed were removed (one was computed by multiplying the measured time by a constant).

### 2.2 Manifold-Constrained Hyper-Connections (`turing/core/mhc.py`)
- **Doubly-Stochastic Birkhoff Polytope**: $n=4$ parallel residual streams are mixed using matrix $P$ constrained via in-SRAM Sinkhorn-Knopp iteration:
  ```math
  P \leftarrow \mathrm{diag}(u)\, \exp(A / \tau)\, \mathrm{diag}(v) \quad \text{s.t.} \quad \sum_j P_{ij} = 1, \; \sum_i P_{ij} = 1
  ```
- **Dynamical Stability**: Enforces non-expansive representation propagation ($\|h_{\text{out}}\|_2 \le \|h_{\text{in}}\|_2$), mathematically preventing gradient explosion across ultra-deep transformer topologies.

### 2.3 Heterogeneous CPU–GPU MoE Dispatcher (`turing/core/heterogeneous_moe.py`)
- **Split Memory Layout**: Dense attention and shared router weights reside in GPU VRAM, while the expert pool is held in pinned host DRAM in packed W4A16 format.
- **Dynamic Routing Inequality**: Measures real-time PCIe bandwidth ($B_{\text{PCIe}}$) and CPU compute throughput ($T_{\text{CPU}}$) to dynamically partition expert execution:
  ```
  Dispatch to GPU  →  if  (size_INT4 / B_pcie) + (FLOPs / T_gpu)  <  (FLOPs / T_cpu)
  Dispatch to CPU  →  otherwise
  ```
- **Global GPU LRU Expert Slot Cache (`turing/core/expert_cache.py`)**: Multi-layer slot cache in VRAM keeps hot experts resident so they are not re-sent over PCIe. Hit rates have been measured only on synthetic routing traces; no MoE checkpoint has been run.

### 2.4 Hierarchical Sequence Paging & SVD Compression (`turing/core/paging.py`)
- **Huge Pages (512 tokens)**: Heavily Compressed Attention (HCA) pools 128 tokens into a single summary representation.
- **Medium Pages (64 tokens)**: Compressed Sparse Attention (CSA) maintains $m=4$ centroids with dynamic top-$k$ block selection.
- **Small Pages (16 tokens)**: Sliding-window uncompressed decode tail.
- **SVD Subspace INT8 Quantization**: Projects Key/Value caches into Rank-64 singular bases. Not in the serving path: the basis is fitted on random data, and at head_dim 64 rank 64 does not compress. The serving cache is the int8/int4 cache in `turing/core/kv_cache.py` ([benchmarks §3](docs/benchmarks.md#3-kv-cache-memory-and-decode-throughput)).

### 2.5 Frontier Speculative Suite (`turing/core/speculation.py`)
- **Subspace-EAGLE3**: Projects layer-$(L-1)$ states into Rank-64 subspace ($\mathbf{z}_t = \mathbf{U}_k^\top h_{L-1}$), drafting candidate tokens from it.
- **DFlash $O(1)$ Dilated Convolutions**: Generates candidate token representations concurrently in a single parallel step.
- **Entropy Confidence Tree Pruner**: Expands speculative DAG trees based on online Shannon entropy.

### 2.6 Multi-Turn Clean-Base Lineage & $k$-Slot Pooling (`turing/core/lineage.py`, `turing/core/kslot_pooling.py`)
- **`CleanBaseLineageBuffer`**: Stores frozen pristine receiver states (`peak_live_caches=2`) and recomputes residuals $\Delta C_R$ against that baseline every deliberation turn, so translation errors do not compound across turns as they do with naive re-injection onto an already-modified cache.
- **`CacheLineage`**: Cryptographic append-only ledger computing deterministic BLAKE2b digests of Key-Value memory buffers for drift auditability and tamper protection.
- **`KSlotCachePooler`**: Compresses $N$-token KV caches into $k=4$ learned summary slots per head/layer via multi-head query attention, so a transfer carries $k$ slots instead of $N$ positions. Not measured on a model; the committed random-data run was slower than its baseline.
- **`GatedZeroIdentityHead`**: Combines sigmoid gating with zero-initialized linear projections, mathematically guaranteeing that an untrained translator produces exactly $\Delta C_R = 0$.

---

## 3. Layer 3: GPU Hardware & Triton Kernels (`turing/kernels/`)

| Kernel Module | Implementation File | Key Hardware Mechanism |
| :--- | :--- | :--- |
| **Batched Continuous Paged Decode GEMV** | `triton_paged_decode_gemv.py` | Multi-sequence ($B=1\dots32$) PagedAttention decode GEMV over 2D block tables (not used by the serving path, which uses `triton_kv_decode.py`). |
| **Single-Step Linear Recurrence Attention** | `triton_linear_recurrence_step.py` | Single-step decode linear recurrence fusing state decay and output projection in SRAM. |
| **In-SRAM Fused MoE Block-GEMM** | `triton_fused_moe_block_gemm.py` | Fused token routing, SwiGLU projection $\text{SiLU}(x W_g) \odot (x W_u) W_d$, and accumulation in $O(1)$ dispatches. |
| **In-VRAM Speculative Verification** | `triton_spec_verify.py` | In-VRAM speculative token candidate verification without CPU `.item()` pipeline stalls. |
| **Fused $k$-Slot Pooling + RoPE** | `triton_kslot_pool.py` | In-SRAM inverse RoPE rotation + softmax attention + weighted sum reduction. |
| **SRAM-Fused SwiGLU** | `triton_swiglu.py` | Fused $M \times N \times K$ GEMM with on-the-fly Swish gating with zero intermediate DRAM writes. |
| **Packed INT4 W4A16 GEMM** | `triton_w4a16.py` | Packs 8 INT4 nibbles into 32-bit words; dequantizes directly into GPU shared memory (SRAM). |
| **Flash-Tree Attention** | `triton_flash_tree.py` | Verifies speculative candidate DAG trees in a single fused GPU attention pass. |
| **Subspace Recirculation** | `triton_recirculation.py` | 2D-tiled fused Tensor Core recirculation for multi-agent belief state refinement. |
| **Linear Recurrent Attention** | `triton_linear_recurrence.py` | Fused chunk-parallel linear attention kernel computing intra-chunk attention and inter-chunk state recurrence in SRAM. |
| **1-Pass Online Shannon Entropy** | `shannon_entropy_cuda.py` | Online 1-pass Softmax + Log-Softmax + Entropic Summation reduction directly in registers without materializing probabilities. |
| **Hexagonal Spatial Codebook** | `triton_hex_quant.py` | In-SRAM cosine similarity + Bitonic minimum index reduction against hexagonal codebook prototypes. |
| **DFlash 1D Dilated Conv** | `triton_dilated_conv.py` | In-SRAM Subspace Projection + 1D Dilated Depthwise Conv. |
| **Fused Base + LoRA GEMV** | `triton_fused_lora.py` | Single-block Tensor Core fused Base GEMV + LoRA Rank-8 contraction for low-latency tokens. |
| **1-Pass Outlier Reduction** | `triton_residual_outlier.py` | 1-Pass In-SRAM absolute max outlier reduction. |
| **Fused Gumbel Router** | `triton_fused_router.py` | Single-launch Mean Pool + RMSNorm + Gate GEMV + Bitonic Top-K tile mask. |
| **In-VRAM Tensor Checksum** | `triton_vram_hash.py` | In-VRAM parallel reduction rolling 64-bit checksum over KV tensor buffers (currently converts floats to integers instead of bitcasting them). |
| **Zero-Sync Quadtree MRP Draft** | `triton_quadtree_mrp.py` | Fused Matryoshka GEMV + Top-64 Bitonic sort + 2D Cartesian quadrant binning on GPU (0 host stalls). |
| **Fused Gated Zero-Identity** | `triton_gated_zero_identity.py` | Single-block Tensor Core GEMM + Sigmoid modulation for zero-drift KV residual update. |
| **Fused Chunk Context Filter** | `triton_chunk_filter.py` | 1-Pass In-SRAM 128-token HCA chunk summary + Bitonic Top-K selection and KV gather. |
| **Batched Cross-Model KV Transfer** | `triton_cross_kv_batched.py` | 1-Launch 80-layer batched representation transfer with in-register RoPE inverse and target re-encoding. |
| **Fused RMSNorm + Subspace SwiGLU** | `triton_fused_rmsnorm_swiglu.py` | In-SRAM RMSNorm + Subspace active tile SwiGLU + In-place residual addition. |
| **Fused QKV + Dynamic RoPE** | `triton_fused_qkv_rope.py` | Fused QKV linear projection with in-place dynamic NTK RoPE frequency rotation. |
| **GPU Batched Option Gather** | `triton_select_gather.py` | GPU-accelerated batched token logit gather and argmax reduction for DSL choice branching. |

None of these kernels has been timed on a model. The decode kernel the serving path uses (`triton_kv_decode.py`) is measured on real weights in [benchmarks §3](docs/benchmarks.md#3-kv-cache-memory-and-decode-throughput).

---

## 4. Layer 4: Production Serving & AI Traffic Management

### 4.1 Continuous Batching Scheduler (`turing/serving/engine.py`)
- **Chunked Prefill & Decode Interleaving**: Chunks long prefill sequences into lane-specific segments (512 for Interactive, 256 for Batch, 128 for Background) interleaved with active decode steps.
- **Triple API Gateway (`turing/serving/server.py`, `anthropic_api.py`, `ollama_api.py`)**:
  - OpenAI `/v1/chat/completions` protocol with `X-Turing-Lane` extraction.
  - Anthropic `/v1/messages` protocol with SSE streaming, reasoning tags, and tool use support.
  - Ollama REST API (`/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/ps`).
  - Response headers: `X-Turing-KV-Utilization`, `X-Turing-Queue-Depth`, and `Retry-After`.
- **Prometheus Telemetry (`/metrics`)**: Exposes `turing_vram_utilization_ratio`, `turing_admission_shed_total`, and per-lane active request counts.

### 4.2 AI Traffic Management & Admission Control (`turing/serving/traffic.py`)
- **`KVMemoryEstimator`**: Calculates static analytical memory footprint for prompt and generation tokens under dense FP16 and SVD INT8 formats.
- **`PrefixHashRouter`**: Computes deterministic 64-bit FNV-1a hashes over prompt token prefixes for consistent prefix-cache worker routing.
- **`AdmissionController`**: Monitors real-time VRAM allocation against high watermark ($0.90$, queues with `Retry-After: 2.0s`) and shed watermark ($0.95$, rejects with HTTP 503).
- **`LanePolicy`**: Enforces 3-lane QoS prioritization (`Interactive` > `Batch` > `Background`) with SLO-driven load shedding.

### 4.3 Concurrency-Adaptive Speculation Gating (`turing/serving/spec_gate.py`)
- **`SpeculationGatePolicy`**: Dynamically adjusts speculation tree width using a hysteresis band ($LOW=2, HIGH=4$), speculating at low concurrency and falling back to plain batch decode at $c \ge 4$ to avoid serial verification lock contention. Experimental: no speedup has been measured on a model.
- **`SpecExactParityVerifier`**: Validates byte-exact token equality between speculative and plain autoregressive decoding under greedy conditions.

### 4.4 Multi-Agent Deliberation Coordinator (`turing/demo/agent_system.py`)
- **`MultiAgentCoordinator`**: Orchestrates proposal generation, environment constraint evaluation, and automated multi-turn self-revision loops with `clean_base` lineage isolation.
- **`DynamicEnvironmentModel` (`turing/demo/world_model.py`)**: Computes real-time constraint penalties to guide multi-agent cloud and systems decisions.
- **`EpistemicUncertaintyGate` (`turing/demo/epistemic_gate.py`)**: Evaluates token entropy to dynamically trigger verification loops or fast-path pass-throughs.

### 4.5 Universal Architecture Registry & Reasoning Engine (`turing/models/`, `turing/serving/reasoning.py`)
- **`ArchitectureRegistry` (`turing/models/architecture_registry.py`)**: Maps 15+ Hugging Face architecture families (`LlamaForCausalLM`, `Qwen2ForCausalLM`, `DeepseekV3ForCausalLM`, `MistralForCausalLM`, `Gemma2ForCausalLM`, `GPT2LMHeadModel`, `OPTForCausalLM`) to `SubspaceCausalLM` and exposes the `AutoSubspaceModel` factory class.
- **`ModelResolver` (`turing/models/resolver.py`)**: Universally resolves canonical Hugging Face repository IDs, `provider/model/reasoning_effort` namespaces (`deepseek-ai/DeepSeek-R1/high`), LiteLLM prefixes (`huggingface/`), local directories, and CLI aliases.
- **`ReasoningBudgetManager` (`turing/serving/reasoning.py`)**: Scales token budgets and recommended sampling temperatures across `low` (1K), `medium` (4K), and `high` (16K) effort levels.
- **`ReasoningStreamFilter` (`turing/serving/reasoning.py`)**: Dynamically parses and filters `<think>...</think>` tokens during autoregressive streaming into OpenAI `delta.reasoning_content` and Anthropic `thinking` blocks.
