# ⚡ Quickstart Guide

Get up and running with Turing Engine in under 30 seconds.

---

## ⚡ Choose Your Setup Path

=== "1. Google Colab (Zero Setup)"
    Run immediately on a free cloud GPU without installing local tools:
    
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

=== "2. Pre-Built Binary Wheel"
    Install pre-compiled C++20 AVX2 native wheels directly from GitHub Releases:
    
    ```bash
    # Direct wheel download for macOS Apple Silicon (arm64):
    pip install https://github.com/intutic/turing/releases/download/v0.1.6/turing_engine-0.1.6-cp311-cp311-macosx_15_0_arm64.whl
    ```

=== "3. Source Build"
    Clone and build the native C++20 extensions locally:
    
    ```bash
    git clone https://github.com/intutic/turing.git
    cd turing
    pip install -e ".[dev]"
    python setup.py build_ext --inplace
    ```

---

## 💬 1. Instant Terminal Chat

Chat directly in your terminal with real pretrained weights:
```bash
# Chat with SmolLM2 or DeepSeek-R1:
turing chat --model smollm2
turing chat --model deepseek-r1-1.5b
turing chat --model qwen3-coder-30b
```

---

## 🖥️ 2. Launching the Serving Server

Start serving a 70B model or MoE architecture with Subspace acceleration:

```bash
turing serve --model deepseek-r1-7b --port 8000 --device auto
```

### Test OpenAI Endpoint
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
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
