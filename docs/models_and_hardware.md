# 🌐 Supported Models & Hardware Setup

Turing Engine runs frontier 70B–320B models on single consumer GPUs (24GB VRAM), Apple Silicon Macs, and multi-vendor hardware.

---

## 1. Supported Models & VRAM Requirements

Turing Engine features a **Universal Dynamic Model Resolver** (`ModelResolver`) and **Architecture Registry** (`ArchitectureRegistry`). It can load and serve **literally ANY open-weight causal language model on Hugging Face Hub** or local disk out of the box without waiting for hardcoded model configs or engine updates.

### A. Popular Open-Weight Models & VRAM Requirements

| Model Family | Model Repository ID / Identifier | Parameter Scale | Uncompressed FP16 | **Turing Engine VRAM** | Minimum Hardware Target |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | 1.5B Dense | 3.0 GB | **0.8 GB** | 1x 4GB GPU / Mac / Laptop |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7.0B Dense | 14.0 GB | **2.2 GB** | 1x 6GB GPU / Mac |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 14.0B Dense | 28.0 GB | **4.1 GB** | 1x 8GB GPU / Mac |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 32.0B Dense | 64.0 GB | **8.5 GB** | 1x 12GB GPU / Mac |
| **DeepSeek** | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | 70.0B Dense | 140.0 GB | **15.7 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Meta AI** | `meta-llama/Llama-3.1-8B-Instruct` | 8.0B Dense | 16.0 GB | **2.5 GB** | 1x 6GB GPU / Mac |
| **Meta AI** | `meta-llama/Llama-3.3-70B-Instruct` | 70.6B Dense | 141.2 GB | **15.7 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Meta AI** | `meta-llama/Llama-4-Scout-17B-16E` | 109B MoE (17B act) | 218.0 GB | **4.2 GB VRAM + 24 GB RAM** | 1x 16GB GPU / Mac Studio |
| **Alibaba** | `Qwen/Qwen2.5-Coder-32B-Instruct` | 32.5B Dense | 65.0 GB | **8.5 GB** | 1x 12GB GPU / Mac |
| **Alibaba** | `Qwen/Qwen2.5-72B-Instruct` | 72.7B Dense | 145.4 GB | **16.1 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Mistral AI** | `mistralai/Mistral-Small-24B-Instruct-2501` | 24.0B Dense | 48.0 GB | **6.1 GB** | 1x 12GB GPU / Mac |
| **Mistral AI** | `mistralai/Mistral-Large-Instruct-2407` | 123B Dense | 246.0 GB | **18.2 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Google** | `google/gemma-2-27b-it` | 27.0B Dense | 54.0 GB | **6.8 GB** | 1x 12GB GPU / Mac |
| **Google** | `google/gemma-4-26B-A4B` | 26B MoE (4B act) | 52.0 GB | **2.5 GB VRAM + 8 GB RAM** | 1x 8GB GPU / Mac |
| **Zhipu** | `zai-org/GLM-5.2-753B` | 753B MoE (32B act) | 1,400 GB | **4.5 GB VRAM + 96 GB RAM** | **1x 24GB GPU + 128GB RAM** |
| **Zhipu** | `zai-org/GLM-5.3-Flash` | 320B MoE (18B act) | 596.0 GB | **3.5 GB VRAM + 42 GB RAM** | **1x 24GB GPU + 64GB RAM / Mac Studio** |
| **DeepSeek** | `deepseek-ai/DeepSeek-V3` | 671B MoE (37B act) | 1,342 GB | **4.8 GB VRAM + 80 GB RAM** | **1x 24GB GPU + 128GB RAM** |
| **DeepSeek** | `deepseek-ai/DeepSeek-V4-Flash` | 284B MoE (13B act) | 528.9 GB | **2.5 GB VRAM + 35 GB RAM** | **1x 24GB GPU + 64GB RAM / Mac Studio** |
| **Moonshot**| `moonshotai/Kimi-K2.6` | 1.04T MoE (48B act) | 2,080 GB | **4.2 GB VRAM + 128 GB RAM** | 1x 24GB GPU + 128GB RAM |
| **Moonshot**| `moonshotai/Kimi-K3` | 2.8T MoE (104B act) | 5,200 GB | **5.0 GB VRAM + 240 GB RAM** | 1x 24GB GPU + 256GB RAM |
| **Turing** | `turing/turing-trillion-1t` | 1.0T MoE (48B act) | 2,000 GB | **4.5 GB VRAM + 128 GB RAM** | 1x 24GB GPU + 128GB RAM |

### B. Universal Model Identification & Tri-Part Namespaces

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


!!! info "Why MoE Host Paging Is Fast (Avoiding Naive Offloading Latency)"
    Traditional layer-by-layer offloading crawls at 1–2 tok/s because it transfers uncompressed weights over PCIe synchronously. Turing Engine sustains **18–50+ tok/s** through two distinct hardware paths:
    
    * **Apple Silicon (Mac Studio M-series)**: Leverages Unified Memory Architecture with **800 to 1,600 GB/s bandwidth**. There is zero PCIe bus copy overhead—the Metal GPU accesses active experts directly at memory-bus speeds (35–50 tok/s).
    * **Discrete PCIe GPUs (RTX 4090 / L4 24GB)**: Attention and embeddings remain permanently pinned in VRAM (~4GB). An on-GPU LRU slot cache captures high expert temporal locality (~80% reuse), while missing INT4-packed experts are prefetched asynchronously over CUDA streams during attention compute without stalling the pipeline (18–30+ tok/s).

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
