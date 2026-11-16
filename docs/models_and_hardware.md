# 🌐 Supported Models & Hardware Setup

Turing Engine resolves and loads HuggingFace and GGUF checkpoints on NVIDIA GPUs, Apple Silicon Macs and CPUs. This page separates what has been run from what is only named.

---

## 1. Models and Memory

> **What is measured today:** the HuggingFace loader reproduces HuggingFace's logits (prefill, chunked prefill and decode, `atol 1e-3`) for the **Llama**, **Llama-3.1 RoPE scaling** (`rope_type="llama3"`), **Qwen2/Qwen2.5** (q/k/v biases, also DeepSeek-R1-Distill-Qwen) and **Qwen3** (q/k norms) architectures, tested on small randomly initialised models of each (`tests/test_hf_parity.py`); quality on real weights is measured for SmolLM2 only. The loader refuses what it cannot reproduce (other RoPE scaling types, sliding-window attention, `o_proj` biases), and `serve`, `generate` and `chat` exit when weights fail to load instead of falling back to random weights (`--mock` runs synthetic weights on purpose). Other families (Gemma, Phi) and the GGUF loader have no parity test. Models run with real weights so far: SmolLM2 135M, 360M and 1.7B through the Turing engine, and GPT-2 and SmolLM2-1.7B-Instruct through HuggingFace `generate`. No MoE checkpoint and no model larger than 1.7B has been run end to end. The serving path keeps fp16 weights (int4 weight loading is not wired), so weight memory is about 2 bytes per parameter: a 70B model needs about 141 GB and does not fit on one 24 GB GPU.

Turing Engine features a **Universal Dynamic Model Resolver** (`ModelResolver`) and **Architecture Registry** (`ArchitectureRegistry`). It resolves any Hugging Face Hub repository ID or local checkpoint and derives the architecture from its `config.json`; exact output parity is only implemented and tested for Llama-architecture models.

### A. Model Geometries and fp16 Weight Memory

The rows below are names the resolver and `SIZING_PROFILES` know about; **none of them has been run**. The fp16 column is parameters × 2 bytes. An earlier "Turing Engine VRAM" column (for example 15.7 GB for Llama-3.3-70B, 3.5 GB VRAM + 42 GB RAM for a 320B MoE) was removed: it was not computed by any script, and it implies 0.7–1.8 bits per parameter, below what 4-bit weights with FFN pruning would give.

| Model Family | Model Repository ID | Parameters | fp16 weights | Fits one 24 GB GPU in fp16? |
| :--- | :--- | :---: | :---: | :---: |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B | ~3 GB | Yes |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | ~14 GB | Yes |
| **Meta AI** | `meta-llama/Llama-3.1-8B-Instruct` | 8B | ~16 GB | Yes |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14B | ~28 GB | No |
| **Mistral AI** | `mistralai/Mistral-Small-24B-Instruct-2501` | 24B | ~48 GB | No |
| **Google** | `google/gemma-2-27b-it` | 27B | ~54 GB | No |
| **Alibaba** | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32.5B | ~65 GB | No |
| **Meta AI** | `meta-llama/Llama-3.3-70B-Instruct` | 70.6B | ~141 GB | No |
| **Alibaba** | `Qwen/Qwen2.5-72B-Instruct` | 72.7B | ~145 GB | No |
| **DeepSeek** | `deepseek-ai/DeepSeek-V3` | 671B MoE (37B active) | ~1,342 GB | No |

Qwen2/Qwen2.5 and Qwen3 checkpoints reproduce HuggingFace's logits in the parity tests (small random models; no real Qwen checkpoint has been run). MoE checkpoints have no loader path.

### B. Multi-Model Pipeline Endpoints

`/v1/embeddings`, `/v1/rerank` and `/v1/extract` are served by `ElasticMultiModelManager`, whose model entries are placeholders today: embeddings pool random hidden states, rerank scores word overlap, and extract is a substring match. No embedding, reranker, NER or OCR model is loaded. The per-model and "7.42 GB total" figures shown in earlier versions were the sum of estimates declared in the benchmark script, not memory read from a device.

### C. Universal Model Identification & Tri-Part Namespaces

Turing Engine's `ModelResolver` seamlessly parses all industry-standard model naming patterns:

```bash
# 1. Canonical Hugging Face Hub Repository ID (Any open-weight model):
turing serve --model meta-llama/Llama-3.3-70B-Instruct
turing chat --model EleutherAI/pythia-70m

# 2. Tri-Part Provider/Model/Reasoning Namespace:
turing serve --model deepseek-ai/DeepSeek-R1/high

# 3. Colon Reasoning Effort Suffix:
turing chat --model meta-llama/Llama-3.3-70B-Instruct:low

# 4. LiteLLM / Model Gateway Prefix:
turing serve --model huggingface/meta-llama/Llama-3.1-8B-Instruct

# 5. Local Checkpoint Directory:
turing serve --model /path/to/custom/finetuned/weights/

# 6. Local Quantized .gguf File (Zero-copy binary reader):
turing chat --model ./models/llama-3.3-70b-q4_k_m.gguf
turing serve --model ./models/qwen2.5-coder-32b-q8_0.gguf

# 7. Ergonomic CLI Shortcuts:
turing chat --model deepseek-r1-1.5b
```

### C. ❓ Architecture Clarification: `CLI_ALIASES` vs. `SIZING_PROFILES`

To ensure full transparency and avoid confusion regarding model loading:

* **`CLI_ALIASES` (`turing/models/resolver.py`)**:
  - **What it is**: Purely optional **typing shortcuts for interactive developers** (e.g. typing `turing chat --model smollm2` instead of typing `turing chat --model HuggingFaceTB/SmolLM2-135M`).
  - **Dynamic Passthrough**: If you pass any full repo ID (e.g., `turing chat --model stabilityai/stablelm-base-alpha-3b` or `EleutherAI/pythia-70m`), `ModelResolver` skips aliases and loads directly from Hugging Face Hub.
* **`SIZING_PROFILES` (`turing/models/sizing_profiles.py`)**:
  - **What it is**: An offline catalog of theoretical parameter dimensions used **strictly by `turing bench`** for dry-run theoretical FLOP simulations, memory bandwidth calculations, and PCIe bottleneck modeling without needing to download 140GB weight checkpoints to disk.
  - **Zero Impact on Live Inference**: Live serving and chat **never use static sizing profiles**; they dynamically introspect the model's remote `config.json` on the fly via `ModelConfig.from_pretrained()`.


!!! info "MoE host paging: components only"
    The repository has an on-GPU LFRU expert slot cache (`GPULRUExpertCache`), routing-history pinning (`RoutingHistoryTracker`) and a one-layer-ahead prefetcher (`OneLayerAheadExpertPrefetcher`), tested on synthetic routing traces. No MoE checkpoint has been decoded, so there is no measured MoE throughput; earlier "18–50 tok/s" figures had no producing script. Apple M-series memory bandwidth tops out around 800 GB/s (Ultra parts), not 1,600 GB/s.

---

## 2. Universal Hardware Setup

Turing Engine auto-detects your silicon at startup. You can also explicitly verify or select your backend:

=== "NVIDIA (CUDA + Triton 3.x)"
    ```bash
    # Install standard CUDA build:
    pip install torch torchvision triton
    turing serve --model deepseek-r1-7b --device cuda
    ```

=== "Apple Silicon (Mac M1/M2/M3/M4 Metal)"
    ```bash
    # Direct Metal acceleration on unified memory:
    turing serve --model deepseek-r1-7b --device mps
    ```

=== "AMD GPUs (ROCm / HIP)"
    ```bash
    # Supports RX 7900 XTX / 8000 and MI300X with Wave32/Wave64 tuning:
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
    turing serve --model deepseek-r1-7b --device rocm
    ```

=== "Intel GPUs (Arc & Data Center Max)"
    ```bash
    # Intel OneAPI / SYCL acceleration:
    pip install intel-extension-for-pytorch
    turing serve --model deepseek-r1-7b --device xpu
    ```

=== "CPU (AVX2 / NEON SIMD)"
    ```bash
    # Pure bare-metal C++20 AVX2 SIMD fallback:
    turing serve --model deepseek-r1-1.5b --device cpu
    ```

---

## 3. Hardware Diagnostics CLI

Inspect your detected accelerator, compute backend, and available memory in one command:

```bash
python scripts/test_universal_backends.py
```
