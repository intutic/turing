# Turing Engine

**A small, readable LLM inference server for testing KV-cache and quantization ideas against real models, where every
number in the docs comes from a committed results file.**

[![Release](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/intutic/turing/blob/master/CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-705-brightgreen.svg)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## What it is

An inference and serving runtime for NVIDIA GPUs, Apple Silicon Macs and CPUs. It loads HuggingFace checkpoints,
llama.cpp GGUF files and its own 4-bit `.tgate4` directories, reproduces the HuggingFace model's logits by default, and
serves OpenAI, Anthropic and Ollama APIs from one process. Every lossy option (low-bit KV, Quest page selection, 4-bit
weights, activation sparsity) is opt-in.

## What it is not

A faster engine. On one NVIDIA L4, vLLM is 1.1–1.5× faster in fp16 and about 2× faster at 4 bits, and llama.cpp is
faster at batch 1 in 4 bits ([comparison](comparison.md)). For production serving use vLLM, SGLang or TensorRT-LLM;
on a laptop, llama.cpp or Ollama.

## What it is good for

* **Low-bit KV caches that stay correct without settings.** At startup Turing finds the layers int8/int4 storage
  damages and keeps them in the model's dtype. On Qwen2.5-1.5B-Instruct its int8 and int4 caches pass every pass-key
  retrieval to 32K and keep GSM8K within 5 points of bf16, where vLLM's default fp8 and its per-token int4 caches
  drop to 0.6–2.4% unless layer 0 is excluded by hand ([benchmarks §2](benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps)).
* **Measured correctness.** HuggingFace parity for 16 model types in the tests and on real checkpoints; 13 real GGUF
  files agree with llama.cpp ([models & hardware](models_and_hardware.md)).
* **Evals that run any idea against real models**: KV and weight quality (`scripts/eval_kv_quality.py`), GSM8K and
  pass-key retrieval through the engine, and head-to-head benchmarks against vLLM and llama.cpp.
* **Research components, measured honestly**, including the ones that did not work
  ([research findings](measured-differentiators.md)).

---

## Quickstart

```bash
git clone https://github.com/intutic/turing.git && cd turing && pip install -e .
turing chat --model smollm2
turing serve --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

---

## Navigation

* [Technical comparison](comparison.md)
* [Quickstart](quickstart.md)
* [Models & hardware](models_and_hardware.md)
* [Serving & API reference](serving.md)
* [Architecture](architecture.md)
* [Docker](docker.md)
* [Kubernetes & llm-d](llmd-integration.md)
* [Integrations](integrations.md)
* [Benchmarks](benchmarks.md)
* [Research findings](measured-differentiators.md)
* [Claims audit](claims-audit.md)
* [Licensing](licensing.md)
