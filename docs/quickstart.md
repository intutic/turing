# Quickstart Guide

Get up and running with Turing Engine in under 30 seconds.

---

## ⚡ Choose Your Setup Path

=== "1. Google Colab (Zero Setup)"
    Run immediately on a free cloud GPU without installing local tools:
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

=== "2. Pre-Built Binary Wheel"
    Install pre-compiled C++20 AVX2 native wheels directly from GitHub Releases:
    
    ```bash
    # Direct wheel download for Linux x86_64:
    pip install https://github.com/intutic/turing/releases/download/v0.1.1/turing_engine-0.1.1-cp311-cp311-linux_x86_64.whl

    # Direct wheel download for macOS Apple Silicon (universal2):
    pip install https://github.com/intutic/turing/releases/download/v0.1.1/turing_engine-0.1.1-cp311-cp311-macosx_10_9_universal2.whl
    ```

=== "3. Docker Container"
    Pull and run the official container from GitHub Packages:
    
    ```bash
    docker run -d -p 8000:8000 --gpus all ghcr.io/intutic/turing:latest
    ```

=== "4. Source Build"
    Clone and build the native C++20 extensions locally:
    
    ```bash
    git clone https://github.com/intutic/turing.git
    cd turing
    pip install -e ".[dev]"
    python setup.py build_ext --inplace
    ```

---

## 🖥️ Launching the Inference Server

Start serving a 70B model with Subspace acceleration:

```bash
turing serve --model llama-3.1-70b --port 8000 --device auto
```

### Test OpenAI Endpoint
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Explain Subspace Pruning in two sentences."}]
  }'
```

### Test Anthropic Endpoint
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: turing-local" \
  -d '{
    "model": "claude-3-5-sonnet",
    "max_tokens": 128,
    "messages": [{"role": "user", "content": "Hello Turing Engine!"}]
  }'
```
