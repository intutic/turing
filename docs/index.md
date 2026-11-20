# Turing Engine

**An LLM inference server whose int8/int4 KV cache finds the layers low-bit storage would break and keeps them in full
precision**, with HuggingFace parity tests on 16 model types, 4-bit GPTQ weights, and OpenAI, Anthropic and Ollama
APIs on one port. It runs on NVIDIA GPUs, Apple Silicon and CPUs. With Qwen2.5-1.5B on an NVIDIA L4, its int4 KV cache
holds 2.4× as many tokens as a 16-bit cache in the same GPU memory, at +0.02 nats, and 4-bit decoding with the opt-in
4-bit LM head is 1.15–1.16× faster than vLLM at batch sizes 1 and 8.

[![PyPI](https://img.shields.io/pypi/v/turing-engine.svg)](https://pypi.org/project/turing-engine/)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-blue.svg)](https://pypi.org/project/turing-engine/)
[![Release](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/intutic/turing/blob/master/CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-1108-brightgreen.svg)](https://github.com/intutic/turing/actions)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## What it is

It loads HuggingFace checkpoints, llama.cpp GGUF files and its own 4-bit `.tgate4` directories, and reproduces the
HuggingFace model's logits when the KV cache is kept in the model's dtype. The KV cache defaults to int8 on CUDA (+0.003
nats on Qwen2.5-1.5B) and to the model's dtype elsewhere; every other lossy option (int4 KV, Quest page selection,
4-bit weights, activation sparsity) is opt-in.

## Get started

=== "pip"
    ```bash
    pip install turing-engine                       # Python 3.10–3.12
    turing chat --model smollm2                     # terminal chat
    turing serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
    ```

=== "Docker (CPU)"
    ```bash
    docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/intutic/turing:v2.0.0 --model smollm2
    ```

=== "Docker (CUDA)"
    ```bash
    docker run --rm --gpus all -p 127.0.0.1:8000:8000 ghcr.io/intutic/turing:v2.0.0-cuda \
      --model Qwen/Qwen2.5-0.5B-Instruct --device cuda
    ```

Then point any OpenAI, Anthropic or Ollama client at `http://localhost:8000`, or install the LangChain or LlamaIndex
client, `langchain-turing` or `llama-index-llms-turing`. More in the [quickstart](quickstart.md),
[serving reference](serving.md) and [integrations](integrations.md).

---

## What it is good for

* **Low-bit KV caches that stay correct without settings.** At startup Turing finds the layers int8/int4 storage
  damages and keeps them in the model's dtype. On Qwen2.5-1.5B-Instruct its int8 and int4 caches pass every pass-key
  retrieval to 32K and keep GSM8K within 5 points of bf16, where vLLM's default fp8 and its per-token int4 caches
  drop to 0.6–2.4% unless layer 0 is excluded by hand ([benchmarks §2](benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps)).
* **Measured correctness.** HuggingFace parity for 16 model types in the tests, 13 of them also on real checkpoints;
  13 real GGUF files agree with llama.cpp ([models & hardware](models_and_hardware.md)).
* **Evals that run any idea against real models**: KV and weight quality (`scripts/eval_kv_quality.py`), GSM8K and
  pass-key retrieval through the engine, and head-to-head benchmarks against vLLM and llama.cpp.
* **Experimental components with results on real models**, including the ones that did not work
  ([experimental components](experimental.md)).

## What it is not

A production server. On one NVIDIA L4 vLLM is 1.01–1.04× faster in fp16 and 1.06–1.12× at 4 bits; Turing is ahead
with its opt-in 4-bit LM head at batch 1 and 8, at 16K-token prompts in fp16, and in cold start
([comparison](comparison.md)). For production serving use vLLM, SGLang or TensorRT-LLM; on a laptop, llama.cpp or
Ollama.

---

## Where to go next

| To… | Read |
| :--- | :--- |
| Install, chat, serve, convert to 4-bit | [Quickstart](quickstart.md), [Docker](docker.md) |
| Look up flags, endpoints and API behaviour | [Serving & API](serving.md) |
| Check whether your model and hardware are covered | [Models & hardware](models_and_hardware.md) |
| Connect LangChain, LlamaIndex, LiteLLM, KServe or Helm | [Integrations](integrations.md), [Kubernetes & llm-d](llmd-integration.md) |
| Compare with vLLM and llama.cpp | [Comparison](comparison.md), [Benchmarks](benchmarks.md) |
| Understand or change the code | [Architecture](architecture.md) |
| Try the experimental components | [Experimental components](experimental.md) |
| Use it commercially | [Licensing](licensing.md) |
