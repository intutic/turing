# Contributing to Turing Engine

Thank you for your interest in contributing to **Turing Engine**! We welcome contributions from systems engineers, kernel developers, machine learning researchers, and open-source practitioners.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Operating System**: Red Hat Enterprise Linux (RHEL 8/9, Rocky Linux, AlmaLinux, Fedora), Ubuntu / Debian / Arch, macOS (Apple Silicon M1-M4 & Intel), or Windows 10/11 (Native MSVC or WSL2)
- **Python**: 3.9, 3.10, 3.11, or 3.12
- **C++ Compiler**: Modern `g++` (GCC 11+ / `gcc-toolset-12` on RHEL), `clang++`, or Microsoft Visual C++ (`cl.exe` from Build Tools 2019/2022) with **C++20** support
- **CPU Architecture**: x86_64 with **AVX2** SIMD support, or Apple Silicon (ARM64 with Accelerate/Metal)
- **GPU (Optional)**: NVIDIA GPU with CUDA 12.0+ and Triton 2.1+ for GPU kernel acceleration

### 2. Setting Up Your Environment
```bash
# Clone the repository
git clone https://github.com/intutic/turing.git
cd turing

# Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"

# Build the native C++20 SIMD PyBind11 extension
python setup.py build_ext --inplace
```

---

## 🧪 Testing & Verification

We enforce strict test cleanliness. All pull requests must pass the full test suite with **zero warnings**:

```bash
# Run the complete test suite with strict warning escalation
python -W error -m pytest -v

# Run a specific test module
python -W error -m pytest tests/test_cross_model_kv_transfer.py -v
```

### Adding New Tests
- When introducing new core algorithms or GPU kernels, add comprehensive unit tests in the `tests/` directory.
- For SIMD or Triton acceleration routines, include **numerical parity tests** comparing the optimized implementation against a standard PyTorch reference tensor output with strict tolerance (`torch.allclose(atol=1e-3, rtol=1e-3)`).

---

## 📐 Code Style & Conventions

### Python
- We follow standard **PEP 8** guidelines.
- Use explicit type annotations for function signatures.
- Avoid unnecessary external dependencies; prefer PyTorch, NumPy, and the standard library.

### C++ (`csrc/`)
- All C++ code must adhere to **C++20** standards.
- Maintain **64-byte alignment** for all SIMD vector loads (`posix_memalign`, `_mm256_load_ps`, `_mm256_fmadd_ps`).
- Native extensions must be built with `-fvisibility=hidden` and Link-Time Optimization (LTO).

### Triton GPU Kernels (`turing/kernels/`)
- Always include dynamic block sizing (`BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`) and power-of-two padding guards.
- Implement proper fallback to PyTorch/CPU when CUDA is not present.

---

## 🔄 Pull Request (PR) Workflow

1. **Fork the Repository**: Create a personal fork on GitHub.
2. **Create a Feature Branch**: `git checkout -b feature/my-new-kernel`
3. **Implement Your Changes**: Keep commits focused and provide clear, descriptive commit messages.
4. **Verify Tests**: Run `python -W error -m pytest -v` and ensure all 82+ tests pass cleanly.
5. **Submit PR**: Open a pull request against the `master` branch of `intutic/turing`. Include:
   - A clear summary of the motivation and changes.
   - Any benchmark results or hardware specifications (e.g. tested on NVIDIA L4 or Apple M3 Max).

---

## ✍️ Developer Certificate of Origin (DCO)

All contributions to Turing Engine must include a **Developer Certificate of Origin (DCO)** sign-off line in the commit message:

```
Signed-off-by: Your Name <your.email@example.com>
```

### How to Sign Off Commits
- Use the `-s` flag when committing:
  ```bash
  git commit -s -m "feat(subspace): optimize INT8 routing projection"
  ```
- To sign off your previous commit:
  ```bash
  git commit --amend -s --no-edit
  ```

By adding the sign-off, you certify that your contribution complies with the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

---

## 📄 License & Attribution

By contributing to Turing Engine, you agree that your contributions will be licensed under the **Business Source License 1.1 (BSL 1.1)** as specified in [`LICENSE`](LICENSE).
