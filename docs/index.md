# Turing Engine

**An inference server that reproduces HuggingFace's logits for 16 model families, with a paged int8/int4 KV cache, OpenAI, Anthropic and Ollama APIs in one process, and a standalone C++ engine.**

[![Release](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](https://github.com/intutic/turing/blob/master/CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-green.svg)](licensing.md)
[![Tests](https://img.shields.io/badge/Tests-667%20%2849%20need%20CUDA%29-brightgreen.svg)]()

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## What is Turing Engine?

An inference and serving runtime for NVIDIA GPUs, Apple Silicon Macs and CPUs. It loads HuggingFace checkpoints,
llama.cpp GGUF files and its own 4-bit `.tgate4` directories, and by default reproduces the HuggingFace model's
logits; every lossy option (KV quantization, Quest page selection, 4-bit weights, activation sparsity, FFN pruning)
is opt-in.

---

## Measured

Raw results are committed as `results_*.json`; the [benchmarks](benchmarks.md) page has the tables.

* **HuggingFace parity on real weights**: SmolLM2 135M–1.7B score the same mean NLL as HuggingFace to five decimals.
* **GGUF parity with llama.cpp**: 13 real GGUF files across 12 families agree with llama.cpp on the top-1 token at
  97–100% of 64 positions (`results_gguf_parity_mps.json`).
* **KV cache**: int8 costs no measurable NLL on SmolLM2 135M–1.7B, int4 about +0.02 nats; on an NVIDIA L4 the
  v1.7.0 engine decoded SmolLM2-360M at 1,715 tok/s at batch 64 and 2K context, against 827 for HuggingFace
  transformers.
* **C++ engine**: 7.4× the tokens per second of the previous release's binary on a synthetic model
  (`results_cpp_engine_mps.json`).

## Implemented, not yet measured

* **Engine**: automatic prefix caching, prompt-lookup speculative decoding, multi-LoRA serving, CUDA graphs with
  Quest page selection, batched decode on MPS and CPU.
* **Weights**: 4-bit GPTQ/RTN (`turing convert`) and TEAL-style activation sparsity; their quality is measured by
  `scripts/eval_kv_quality.py` and their speed by `scripts/benchmark_weight_formats.py`, but no run is committed yet.
* **API**: chat templates, tools, reasoning budgets, JSON schema by constrained decoding (llguidance), embeddings,
  rerank, extraction, Ollama `/api/pull`, llm-d KV events, Prometheus metrics.
* **Distributed**: tensor and pipeline parallelism and prefill/decode disaggregation, tested as CPU processes only.
* **Comparison with other engines**: none yet. `scripts/benchmark_e2e_generation.py` runs Turing and vLLM on the
  same prompts; its GPU run is pending.

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
* [Claims audit](claims-audit.md)
* [Licensing](licensing.md)
