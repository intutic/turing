# ⚡ Turing Engine

**A small, readable LLM inference server for testing KV-cache and quantization ideas against real models, where every
number in the docs comes from a committed results file.**

[![Docs: Live](https://img.shields.io/badge/Documentation-intutic.github.io%2Fturing-blue.svg)](https://intutic.github.io/turing/)
[![Release: v2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)
[![Tests: 705](https://img.shields.io/badge/Tests-705-brightgreen.svg)](https://github.com/intutic/turing/actions)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## Is it faster than vLLM or llama.cpp?

No. On one NVIDIA L4, on the same model and prompts, vLLM and llama.cpp are faster than Turing in every
configuration measured, and their 4-bit models answer GSM8K about as well (Qwen2.5-1.5B-Instruct, 512-token prompts,
256 new tokens; `results_4bit_speed_cuda.json`, `results_4bit_quality_*_cuda.json`):

<!-- table:vs_4bit -->
| Setting | Batch 1, tok/s | Batch 8, tok/s | GSM8K (n=500) |
| :--- | :---: | :---: | :---: |
| llama.cpp Q4_K_M | 186 | 790 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 169 | 1,078 | 50.2% ± 2.2% |
| Turing GPTQ | 87 | 557 | 52.6% ± 2.2% |
| Turing GPTQ + 4-bit LM head (opt-in) | — | — | — |
| Turing GPTQ + TEAL 25% | 90 | 531 | 47.2% ± 2.2% |
| Turing GPTQ + TEAL 40% | 93 | 530 | 43.6% ± 2.2% |
| vLLM fp16 (reference) | 75 | 533 | — |
| Turing fp16 (reference) | 65 | 440 | — |
<!-- /table -->

At short context in fp16 vLLM is 1.1–1.5× faster, and at 16K-token context 1.4× faster per request at equal quality
([comparison](docs/comparison.md)). For production serving use vLLM, SGLang or TensorRT-LLM; on a laptop, llama.cpp
or Ollama.

## What it is good for

* **Low-bit KV caches that do not silently break the model.** When a pool stores KV in int8 or int4, Turing measures
  at startup which layers that format damages (each layer's K and V are round-tripped alone and the change in the
  next-token distribution is measured on random tokens, in a few seconds) and keeps those layers in the model's
  dtype; outlier key channels are scaled down for int4. On Qwen2.5-1.5B-Instruct vLLM's default fp8, per-token fp8
  and per-token int4 caches collapse (GSM8K 0.6–2.4%); Turing's int8 and int4 keep it working with no setting:

<!-- table:vs_kv -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 3.51 |
| Turing int8 | — | 58.2% | 100.0% | +0.003 ± 0.002 | 3.56 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 3.54 |
| vLLM fp16 (reference) | — | 57.4% | 100.0% | — | 2.60 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.71 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
| vLLM fp8_per_token_head | — | 0.6% | 0.0% | +2.404 ± 0.364 | — |
| vLLM int8_per_token_head | — | 41.4% | 100.0% | +0.023 ± 0.007 | 4.72 |
| vLLM int4_per_token_head | — | 0.6% | 0.0% | +8.127 ± 0.495 | 4.41 |
| vLLM int4_per_token_head | layer 0 (manual) | 47.8% | 100.0% | +0.058 ± 0.010 | 4.30 |
| vLLM turboquant_k8v4 | — | 52.0% | 100.0% | +0.001 ± 0.001 | — |
<!-- /table -->

  vLLM fp8 with layer 0 skipped *by hand* matches Turing's quality at about half the time per request, which is the
  point: the calibration finds that layer automatically. It is also packaged on its own for vLLM users, under
  Apache-2.0: [`packages/kv-layer-guard`](packages/kv-layer-guard). Validated on six instruct models (Qwen2.5 1.5B
  and 7B, Llama 3.2, Gemma 2, Phi-3.5, Mistral), including one limit it does not catch
  ([benchmarks §2](docs/benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps)).
* **Measured correctness.** The exact model reproduces HuggingFace's logits on 16 model families in the tests and on
  real checkpoints ([models & hardware](docs/models_and_hardware.md)); 13 real GGUF files agree with llama.cpp.
* **A codebase small enough to change.** Paged KV pool, Triton decode kernels, GPTQ, TEAL sparsity, prefix caching,
  speculative decoding and the OpenAI, Anthropic and Ollama APIs are a few Python modules and kernels, with a
  standalone C++ engine beside them. The eval scripts (`scripts/eval_kv_quality.py`, `turing eval-accuracy`,
  `turing eval-niah`, the Turing-vs-vLLM and llama.cpp benchmarks) run any idea against real models.

**Not measured:** multi-GPU (tensor and pipeline parallelism are tested as CPU processes only), models larger than
OLMoE-1B-7B, SGLang and TensorRT-LLM.

---

## Quickstart

### 1. Install from source
```bash
git clone https://github.com/intutic/turing.git && cd turing
pip install -e .                     # builds the C++ extension
pip install -e ".[structured]"       # optional: JSON-schema and tool_choice enforcement (llguidance)
```
The package is not on PyPI yet; the release workflow (`.github/workflows/release.yml`) publishes wheels and Docker
images when a version tag is pushed. Docker: [docs/docker.md](docs/docker.md).

### 2. Chat in the terminal
```bash
turing chat --model smollm2                                   # alias for HuggingFaceTB/SmolLM2-135M
turing chat --model Qwen/Qwen2.5-0.5B-Instruct
turing chat --model ./models/qwen2.5-0.5b-instruct-q8_0.gguf  # llama.cpp GGUF, no conversion
```

### 3. Serve OpenAI, Anthropic and Ollama APIs on one port
```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct --port 8000
```
OpenAI (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`), Anthropic (`/v1/messages`) and Ollama
(`/api/chat`, `/api/generate`, `/api/pull`, …) clients can point at `http://localhost:8000`. The server listens on
`0.0.0.0` by default and needs no key unless started with `--api-key` (or `TURING_API_KEY`).

### 4. Quantize to 4-bit once, serve the result
```bash
turing convert --model Qwen/Qwen2.5-7B-Instruct --output ./qwen7b-gptq --method gptq
turing serve --model ./qwen7b-gptq
```

### 5. Turing DSL
```python
import turing

@turing.chain(model="smollm2", device="auto")
def three_answers(question: str):
    turing.gen(f"Question: {question}\nThink step by step.", max_tokens=64)
    branches = turing.fork(3, temperature=0.8)   # the context so far is prefilled once and shared
    for i, b in enumerate(branches):
        b.gen(f"\nAnswer {i + 1}:", max_tokens=64)
    return turing.join(branches, strategy="best")  # highest mean log-probability
```

### 6. Standalone C++ engine (`turing-cli`)
```bash
cmake -B build && cmake --build build
./build/turing-cli generate --model ./models/llama-3.2-1b-instruct-f16.gguf --prompt "Hello"
./build/turing-cli serve --model ./models/llama-3.2-1b-instruct-f16.gguf --port 8000   # binds 127.0.0.1
```
It runs Llama-layout GGUF models (RMSNorm, SwiGLU, RoPE, grouped-query attention) on the CPU with AVX2 or NEON, with
an optional int8 KV cache (`--kv-cache-dtype int8`).

### 7. Tensor and pipeline parallelism
```bash
./scripts/launch_distributed.sh Qwen/Qwen2.5-7B-Instruct 2 2   # MODEL TP PP, via torchrun; rank 0 serves HTTP
```
Sharded weights and KV pages over `torch.distributed`. The tests run TP 2/4, PP 2 and TP 2 × PP 2 as CPU (gloo)
processes and compare logits with a single process; no multi-GPU (NCCL) run has been recorded.

---

## Models

`tests/test_hf_parity.py` checks prefill, chunked prefill, decode and batched decode against HuggingFace
(atol 1e-3) for 25 configurations of these model types: **llama**, **mistral** (sliding window), **qwen2**,
**qwen3**, **gemma**, **gemma2** (soft-capping), **gemma3** (local/global RoPE), **phi** (Phi-2), **phi3** (incl.
LongRoPE), **gpt2**, **opt** (incl. post-LN and projected embeddings), **gpt_neox**, **mixtral**, **qwen2_moe**,
**qwen3_moe** and **olmoe**. RoPE scaling types: default, linear, dynamic, YaRN, LongRoPE, Llama 3 and proportional.
Any other architecture is refused before download (`hf_loader.FAMILIES`, `ArchitectureRegistry`); weights never fall
back to random values. Every family except OPT also loads from GGUF (Mistral and Mixtral through llama.cpp's
`llama` architecture). Details: [models & hardware](docs/models_and_hardware.md).

## Hardware

Recorded runs exist for an NVIDIA L4 (CUDA, Triton kernels, CUDA graphs), an Apple M4 Pro (MPS) and CPUs. `--device`
also accepts `rocm` (PyTorch's HIP build) and the auto-detection picks `xpu` or `vulkan` when PyTorch reports them,
but none of those has been run.

---

## Features

| Area | What the code does |
| :--- | :--- |
| KV cache | One paged pool per model (16-token pages), `native`/`int8`/`int4` storage with the layers that format damages kept in the model dtype (chosen at startup), copy-on-write forks; Quest page selection (`--kv-page-budget`), also inside CUDA graphs |
| Prefix caching | Automatic: full pages indexed by chained token hashes, reused copy-on-write, evicted LRU (`--no-prefix-caching`) |
| Speculative decoding | Prompt lookup (no draft model), `--draft-tokens` (default 4); output distribution equals plain decoding for greedy and sampling |
| Weights | fp32/fp16/bf16; 4-bit GPTQ or RTN (`turing convert`, `--quantization`); TEAL-style activation sparsity in batch-1 decode on CUDA (`--activation-sparsity`, off by default) |
| LoRA | Several PEFT adapters at once (`--lora NAME=PATH`), chosen per request by `model: "base:NAME"` or `X-Turing-Adapter` |
| API | Chat templates, tools (OpenAI and Anthropic; Hermes, Mistral and Llama 3 call formats), reasoning (`reasoning_content`, Anthropic thinking blocks, effort budgets), JSON schema by token-level constrained decoding, stop strings, usage with cached tokens |
| Traffic | Admission control sized to the KV pool and three QoS lanes, on by default (429/503 with `Retry-After`) |
| Other endpoints | Embeddings, rerank, `/v1/extract`, render endpoints, Prometheus `/metrics`, llm-d KV events |
| Distributed | Tensor and pipeline parallelism (torchrun); prefill/decode disaggregation over TCP (`--prefill-worker`, `--prefill-server`) |
| Evals | `turing eval-accuracy` (GSM8K exact match) and `turing eval-niah` (pass-key retrieval), through the engine on real weights |

The repository also holds research components outside the serving path: cross-model KV transfer, `clean_base`
lineage, k-slot pooling, an MoE expert-offload runner, hybrid linear/softmax attention (needs a model trained with it)
and untrained Matryoshka draft heads. What was measured on real models, including what did not work, is in
[research findings](docs/measured-differentiators.md).

---

## CLI

```bash
turing serve --model <id|.gguf|.tgate4 dir> [--port 8000]   # HTTP server
turing chat --model <id>                                    # terminal chat
turing generate --model gpt2 --prompt "Artificial intelligence is"
turing convert --model <hf id> --output <dir> [--method gptq|rtn]
turing eval-accuracy --model <id> --samples 100             # GSM8K, k-shot greedy
turing eval-niah --model <id> --context-len 1024 4096       # pass-key retrieval
turing info                                                 # devices and runtime
```

## Client SDKs

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
r = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                                   messages=[{"role": "user", "content": "Write a binary search in Python."}])
print(r.choices[0].message.content)

import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")
m = client.messages.create(model="Qwen/Qwen2.5-7B-Instruct", max_tokens=128,
                           messages=[{"role": "user", "content": "Hello"}])
print(m.content[0].text)

import ollama
client = ollama.Client(host="http://localhost:8000")
print(client.chat(model="Qwen/Qwen2.5-7B-Instruct", messages=[{"role": "user", "content": "Hello"}])["message"]["content"])
```

---

## Documentation

* [Quickstart](https://intutic.github.io/turing/quickstart/) · [Models & hardware](https://intutic.github.io/turing/models_and_hardware/) · [Serving & API](https://intutic.github.io/turing/serving/)
* [Architecture](https://intutic.github.io/turing/architecture/) · [Benchmarks](https://intutic.github.io/turing/benchmarks/) · [Claims audit](https://intutic.github.io/turing/claims-audit/)
* [Comparison](https://intutic.github.io/turing/comparison/) · [Docker](https://intutic.github.io/turing/docker/) · [Kubernetes & llm-d](https://intutic.github.io/turing/llmd-integration/) · [Integrations](https://intutic.github.io/turing/integrations/) · [Licensing](https://intutic.github.io/turing/licensing/)

---

## 📄 License

Turing Engine is licensed under the **Business Source License 1.1 (BSL 1.1)**.
* **Free for non-production use**: development, local testing, academic research, personal single-GPU workstations and benchmarking ([`LICENSE`](LICENSE), Additional Use Grant). Production use (multi-GPU clusters, commercial hosting, revenue-generating API traffic) needs a separate commercial license.
* `packages/kv-layer-guard/` (the KV layer calibration, usable with vLLM) is under the **Apache License 2.0** now.
* Converts automatically to **Apache 2.0** on **August 24, 2030** or the fourth anniversary of the first public release (as [`LICENSE`](LICENSE) §3 states).
