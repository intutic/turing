# ⚡ Universal Hardware Support: Cross-Vendor Acceleration

Turing Engine provides **universal, cross-vendor hardware execution**, automatically discovering and dispatching computations to the highest performance accelerator available on your machine.

---

## 🌐 Supported Silicon & Hardware Matrix

| Hardware Vendor | Supported GPUs & Silicon | Primary Compute Backend | Acceleration Strategy |
| :--- | :--- | :--- | :--- |
| **NVIDIA** | RTX 3090 / 4090 / 5090, L4, A100, H100 | **CUDA + Triton 3.x** | Custom Tensor Core SwiGLU & Flash-Tree Triton kernels |
| **AMD** | Radeon RX 7900 XTX / 7900 GRE / 8000, Instinct MI250X / MI300X | **ROCm (HIP) + Triton** | Wave32 (RDNA) / Wave64 (CDNA) Matrix Core heuristics |
| **Intel** | Intel Arc A770 / A750 / B580 Battlemage, Data Center GPU Max | **Intel XPU (SYCL / OneAPI)** | Intel XMX Matrix Engines + IPEX bindings |
| **Apple** | M1 / M2 / M3 / M4 (Pro, Max, Ultra) | **Metal (MPS)** | Metal Performance Shaders + Vectorized Subspace Slicing |
| **Cross-Vendor** | Intel, AMD APUs, Qualcomm Snapdragon Adreno, ARM Mali | **Vulkan Compute** | SPIR-V Compute Shaders + Host-Visible Coherent Memory |
| **x86_64 / ARM CPU** | Intel Xeon, AMD EPYC, Ampere Altra, Apple Silicon CPU | **C++20 AVX2 / NEON SIMD** | 64-byte aligned SIMD fused FMA + zero-copy `mmap` |

---

## 🔍 Dynamic Hardware Auto-Discovery

Turing Engine automatically detects your system's hardware configuration at startup:

```python
from turing.kernels.dispatch import get_hardware_backend_info

info = get_hardware_backend_info()
print(f"Backend: {info['backend']} ({info['vendor']}) on {info['device_name']}")
```

You can also run the hardware probe directly from your terminal:
```bash
python scripts/test_universal_backends.py
```

---

## 🛠️ Ecosystem Setup Guides

### 1. AMD ROCm / HIP Setup (Linux)
On systems with AMD Radeon or Instinct GPUs:
```bash
# Install PyTorch with ROCm support:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2

# Install Turing Engine:
pip install turing-engine
```

### 2. Intel Arc / XPU Setup (Linux & Windows)
On systems with Intel Arc GPUs:
```bash
# Install PyTorch with Intel XPU / SYCL support:
pip install torch intel-extension-for-pytorch

# Launch with XPU acceleration:
turing chat --model deepseek-r1-1.5b --device xpu
```

### 3. Apple Silicon Metal (macOS)
On any Mac with M1/M2/M3/M4 chip:
```bash
# Direct pip install (pre-compiled native ARM64 wheel):
pip install turing-engine

# Launch instant terminal chat:
turing chat --model smollm2
```

### 4. Cross-Vendor Vulkan & CPU SIMD Fallback
On systems without proprietary vendor drivers or headless servers:
```bash
# Automatic fallback to C++20 AVX2 SIMD:
turing chat --model deepseek-r1-1.5b --device cpu
```
