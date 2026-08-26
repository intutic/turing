# 🌐 Supported Models & Hardware Setup

Turing Engine runs frontier 70B–320B models on single consumer GPUs (24GB VRAM), Apple Silicon Macs, and multi-vendor hardware.

---

## 1. Supported Models & VRAM Requirements

| Model Family | Model Name | Parameter Scale | Uncompressed FP16 | **Turing Engine VRAM** | Minimum Hardware Target |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **DeepSeek** | `deepseek-r1-1.5b` | 1.5B Dense | 3.0 GB | **0.8 GB** | 1x 4GB GPU / Mac / Laptop |
| **DeepSeek** | `deepseek-r1-7b` | 7.0B Dense | 14.0 GB | **2.2 GB** | 1x 6GB GPU / Mac |
| **DeepSeek** | `deepseek-r1-14b` | 14.0B Dense | 28.0 GB | **4.1 GB** | 1x 8GB GPU / Mac |
| **DeepSeek** | `deepseek-r1-32b` | 32.0B Dense | 64.0 GB | **8.5 GB** | 1x 12GB GPU / Mac |
| **DeepSeek** | `deepseek-r1-70b` | 70.0B Dense | 140.0 GB | **15.7 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Meta AI** | `llama-3.3-70b` | 70.6B Dense | 141.2 GB | **15.7 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Meta AI** | `llama-4-scout` | 109B MoE (17B act) | 218.0 GB | **4.2 GB VRAM + 24 GB RAM** | 1x 16GB GPU / Mac Studio |
| **Alibaba** | `qwen-2.5-coder-32b` | 32.5B Dense | 65.0 GB | **8.5 GB** | 1x 12GB GPU / Mac |
| **Alibaba** | `qwen-2.5-72b` | 72.7B Dense | 145.4 GB | **16.1 GB** | **1x 24GB GPU (RTX 3090/4090, L4)** |
| **Google** | `gemma-2-27b` | 27.0B Dense | 54.0 GB | **6.8 GB** | 1x 12GB GPU / Mac |
| **Google** | `gemma-4-26b` | 26B MoE (4B act) | 52.0 GB | **2.5 GB VRAM + 8 GB RAM** | 1x 8GB GPU / Mac |
| **Zhipu** | `glm-5.3-flash` | 320B MoE (18B act) | 596.0 GB | **3.5 GB VRAM + 42 GB RAM** | **1x 24GB GPU + 64GB RAM / Mac Studio** |
| **DeepSeek** | `deepseek-v4-flash` | 284B MoE (13B act) | 528.9 GB | **2.5 GB VRAM + 35 GB RAM** | **1x 24GB GPU + 64GB RAM / Mac Studio** |
| **Moonshot**| `kimi-k3` | 2.8T MoE (104B act) | 5,200 GB | **5.0 GB VRAM + 240 GB RAM** | 1x 24GB GPU + 256GB RAM |

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
