# ⚡ Turing Engine

**An LLM inference server whose int8/int4 KV cache finds the layers low-bit storage would break and keeps them in full
precision**, with HuggingFace parity tests on 16 model types, 4-bit GPTQ weights, and OpenAI, Anthropic and Ollama
APIs on one port. It runs on NVIDIA GPUs, Apple Silicon and CPUs. With Qwen2.5-1.5B on an NVIDIA L4, its int4 KV cache
holds 2.4× as many tokens as a 16-bit cache in the same GPU memory, at +0.02 nats, and 4-bit decoding (the LM head
in 4 bits too) is 1.07–1.33× faster than vLLM at batch sizes 1 to 64.

[![Docs: Live](https://img.shields.io/badge/Documentation-intutic.github.io%2Fturing-blue.svg)](https://intutic.github.io/turing/)
[![PyPI: turing-engine](https://img.shields.io/pypi/v/turing-engine.svg)](https://pypi.org/project/turing-engine/)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10–3.12-blue.svg)](https://pypi.org/project/turing-engine/)
[![Release: v2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)
[![Tests: 1353](https://img.shields.io/badge/Tests-1353-brightgreen.svg)](https://github.com/intutic/turing/actions)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

It loads HuggingFace checkpoints, llama.cpp GGUF files and its own 4-bit `.tgate4` directories, and matches
HuggingFace's logits when the KV cache is kept in the model's dtype. It is built for trying KV-cache and quantization
changes against real models, not for production serving ([how it compares](#how-it-compares)).

[Get started](#get-started) · [What it is good for](#what-it-is-good-for) · [How it compares](#how-it-compares) ·
[More ways to run it](#more-ways-to-run-it) · [Models](#models) · [Features](#features) · [Documentation](#documentation)

---

## Get started

### 1. Install
```bash
pip install turing-engine                  # Python 3.10–3.12
pip install "turing-engine[structured]"    # optional: JSON-schema and tool_choice enforcement (llguidance)
```
Wheels with the C++ extension are built for Linux x86_64 and aarch64 and macOS arm64; elsewhere (Windows, Intel
Macs) pip builds it from the source package, which needs a C++20 compiler. From a clone: `pip install -e .`.

Or run the server in Docker ([docs/docker.md](docs/docker.md)):
```bash
docker run --rm -p 127.0.0.1:8000:8000 ghcr.io/intutic/turing:v2.0.0 --model smollm2
docker run --rm --gpus all -p 127.0.0.1:8000:8000 ghcr.io/intutic/turing:v2.0.0-cuda --model Qwen/Qwen2.5-7B-Instruct --device cuda
```

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

---

## What it is good for

* **Low-bit KV caches that do not silently break the model.** When a pool stores KV in int8, fp8 or int4, Turing
  measures at startup which layers that format damages (each layer's K and V are round-tripped alone and the change in
  the next-token distribution is measured on random tokens, in a few seconds) and keeps those layers in the model's
  dtype; outlier key channels are scaled down for int4. On Qwen2.5-1.5B-Instruct vLLM's default fp8, per-token fp8 and
  per-token int4 caches collapse (GSM8K 0.6–2.4%); Turing's int8, fp8 and int4 keep it working with no setting:

<!-- table:vs_kv_short -->
| KV cache | Layers kept in bf16 | GSM8K | Pass-key 2K–32K | ΔNLL vs bf16 | s/request (16K, B=128) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Turing bf16 (reference) | — | 57.2% | 100.0% | — | 1.92 |
| Turing int8 | automatic | 58.6% | 100.0% | -0.000 ± 0.002 | 1.73 |
| Turing fp8 | automatic | 56.6% | 100.0% | +0.009 ± 0.005 | 1.70 |
| Turing int4 | automatic | 52.8% | 100.0% | +0.020 ± 0.005 | 2.11 |
| vLLM bf16 (reference) | — | 57.4% | 100.0% | — | 2.62 |
| vLLM fp8 | — | 2.4% | 5.0% | +2.286 ± 0.394 | 1.67 |
| vLLM fp8 | layer 0 (manual) | 56.8% | 100.0% | +0.013 ± 0.003 | 1.73 |
<!-- /table -->

  vLLM fp8 with layer 0 skipped *by hand* matches Turing's quality at 0.82–1.02× Turing's time per request (int4
  to fp8), which is the point: the calibration finds that layer automatically. vLLM's per-token int8 and int4
  caches are in the [full table](docs/comparison.md). It is also packaged on its own for vLLM users, under
  Apache-2.0: [`packages/kv-layer-guard`](packages/kv-layer-guard). Validated on six instruct models (Qwen2.5 1.5B
  and 7B, Llama 3.2, Gemma 2, Phi-3.5, Mistral), including one limit it does not catch
  ([benchmarks §2](docs/benchmarks.md#2-kv-cache-quality-and-the-layers-it-keeps)).
* **Correctness you can check.** The exact model reproduces HuggingFace's logits on 16 model families in the tests, 13 of them
  also on real checkpoints ([models & hardware](docs/models_and_hardware.md)); 13 real GGUF files agree with llama.cpp.
* **A codebase small enough to change.** Paged KV pool, Triton decode kernels, GPTQ, TEAL sparsity, prefix caching,
  speculative decoding and the OpenAI, Anthropic and Ollama APIs are a few Python modules and kernels, with a
  standalone C++ engine beside them. The eval scripts (`scripts/eval_kv_quality.py`, `turing eval-accuracy`,
  `turing eval-niah`, the Turing-vs-vLLM and llama.cpp benchmarks) run any idea against real models.

**Limitations:** tensor and pipeline parallelism are tested as CPU processes only, not on multiple GPUs; the largest
model run end to end is Qwen2.5-7B (7.6B).

---

## How it compares

Is it faster than vLLM or llama.cpp? On one NVIDIA L4 with models of 0.4B to 8B parameters it is ahead of vLLM in
fp16 by a few percent and at 4 bits with its default settings by more, and behind in the settings listed below. Same
model and prompts (Qwen2.5-1.5B-Instruct, 512-token prompts, 256 new tokens):

<!-- table:vs_4bit -->
| Setting | Batch 1, tok/s | Batch 8, tok/s | Batch 64, tok/s | GSM8K (n=500) |
| :--- | :---: | :---: | :---: | :---: |
| llama.cpp Q4_K_M | 182 | 782 | 2,176 | 52.4% ± 2.2% |
| vLLM GPTQ-Int4 (Qwen's checkpoint) | 174 | 1,116 | 3,185 | 50.2% ± 2.2% |
| vLLM AWQ (Qwen's checkpoint) | 172 | 1,107 | 3,159 | — |
| Turing GPTQ, fp16 LM head (`--no-quantize-lm-head`) | 175 | 1,129 | 3,135 | 50.4% ± 2.2% |
| Turing GPTQ (4-bit LM head, the default) | 232 | 1,400 | 3,401 | 51.8% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 25% | 107 | 953 | 2,600 | 46.2% ± 2.2% |
| Turing GPTQ, fp16 LM head, TEAL 40% | 112 | 960 | 2,614 | 42.0% ± 2.2% |
| vLLM fp16 (reference) | 76 | 543 | 2,298 | — |
| Turing fp16 (reference) | 78 | 549 | 2,287 | — |
<!-- /table -->

* **fp16:** Turing is at 99.5–102% of vLLM's throughput at batch 1–64 (repeated runs of one setting differ by up
  to 3%), and at 101–112% on SmolLM2 at 1,024-token prompts.
* **First token:** a prompt arriving alone gets its first token sooner than on vLLM (17 against 30–32 ms on
  SmolLM2-360M, 27 against 31–32 ms on Qwen2.5-1.5B, 64 against 68 ms on SmolLM2-1.7B). In a burst of 32 or 64
  SmolLM2 prompts the mean first token is 10–16% sooner than vLLM's when both engines take 8,192 tokens per step
  and from 4% sooner to 2% later when both take 2,048.
* **4 bits** (the LM head in 4 bits too, the default): Turing is 1.33 / 1.25 / 1.07× vLLM and 1.28 / 1.79 / 1.56×
  llama.cpp at batch 1 / 8 / 64 (llama.cpp measured on another L4). This quantizes one more matrix than vLLM's
  GPTQ checkpoint does; GSM8K 51.8% against vLLM's 50.2% (each ± 2.2%), and the 4-bit head adds 0.007–0.009 nats
  of next-token NLL on three models (`results_quality_head_cuda.json`).
* **4 bits with the LM head in fp16** (`--no-quantize-lm-head`): Turing is at 98–101% of vLLM. llama.cpp is 1.04×
  faster at batch 1; Turing is faster than llama.cpp at batch 8 and 64 (1.44× at both).
* **16K-token prompts, fp16 KV at equal memory:** Turing is 1.03–1.37× faster per request than vLLM, more at larger
  batches. With one byte per value vLLM's fp8 cache is 1.00–1.04× faster than Turing's fp8 and int8 when both
  engines take 2,048 tokens a step, and 1.02–1.08× at its own 8,192.
* **Speculative decoding:** with prompt lookup (8 drafts) Turing is at 100–121% of vLLM's n-gram speculator on code,
  rewrite and chat prompts; with a published EAGLE-3 head on Qwen3-8B it is at 98–101% of vLLM with the same head.
* **MoE models:** OLMoE in bf16 is level with vLLM at batch 1 and 8 and 3% behind at batch 32; Qwen1.5-MoE in 4
  bits is at 101–127% of vLLM on Qwen's GPTQ checkpoint (98–101% with its LM head in fp16, as in that checkpoint).
  Mixtral-8x7B does not fit a 24 GB GPU in 4 bits: with two of each layer's eight experts computed on the CPU
  (`--moe-expert-slots 6`) it runs at 13.7 / 16.5 tok/s at batch 1 / 2, against 13.2 / 14.0 for llama.cpp with 26
  of its layers on the GPU.
* **Cold start:** from process start to first token Turing takes 25.4 s from a cold page cache and 10.0 s warm,
  against vLLM's 51.7 s and 34.2 s (vLLM ran in an earlier run, alongside a Turing build that took 27.4 s and
  12.1 s: 1.9–2.8× sooner); llama.cpp with its 4-bit GGUF takes 7.1 s and 1.3 s.

Details and every other setting are in the [comparison](docs/comparison.md); the results files and the commands
that produced them are in [`benchmarks/`](benchmarks/). These are one GPU and five models: for production serving
(multi-GPU, larger models, a longer track record) use vLLM, SGLang or TensorRT-LLM; on a laptop, llama.cpp or
Ollama.

---

## More ways to run it

### Quantize to 4-bit once, serve the result
```bash
turing convert --model Qwen/Qwen2.5-7B-Instruct --output ./qwen7b-gptq --method gptq
turing serve --model ./qwen7b-gptq
```

### Turing DSL
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

### Standalone C++ engine (`turing-cli`)
```bash
cmake -B build && cmake --build build
./build/turing-cli generate --model ./models/llama-3.2-1b-instruct-f16.gguf --prompt "Hello"
./build/turing-cli serve --model ./models/llama-3.2-1b-instruct-f16.gguf --port 8000   # binds 127.0.0.1
```
It runs Llama-layout GGUF models (RMSNorm, SwiGLU, RoPE, grouped-query attention) on the CPU with AVX2 or NEON, with
an optional int8 KV cache (`--kv-cache-dtype int8`).

### Tensor and pipeline parallelism
```bash
./scripts/launch_distributed.sh Qwen/Qwen2.5-7B-Instruct 2 2   # MODEL TP PP, via torchrun; rank 0 serves HTTP
```
Sharded weights and KV pages over `torch.distributed`. The tests run TP 2/4, PP 2 and TP 2 × PP 2 as CPU (gloo)
processes and compare logits with a single process; multi-GPU (NCCL) runs are not tested yet.

---

## Models

`tests/test_hf_parity.py` checks prefill, chunked prefill, decode and batched decode against HuggingFace
(atol 1e-3) for 25 configurations of these model types: **llama**, **mistral** (sliding window), **qwen2**,
**qwen3**, **gemma**, **gemma2** (soft-capping), **gemma3_text** (local/global RoPE), **phi** (Phi-2), **phi3** (incl.
LongRoPE), **gpt2**, **opt** (incl. post-LN and projected embeddings), **gpt_neox**, **mixtral**, **qwen2_moe**,
**qwen3_moe** and **olmoe**. RoPE scaling types: default, linear, dynamic, YaRN, LongRoPE, Llama 3 and proportional.
Any other architecture is refused before download (`hf_loader.FAMILIES`, `ArchitectureRegistry`); weights never fall
back to random values. Every family except OPT also loads from GGUF (Mistral and Mixtral through llama.cpp's
`llama` architecture). Details: [models & hardware](docs/models_and_hardware.md).

## Hardware

Tested on NVIDIA GPUs (CUDA, Triton kernels, CUDA graphs; benchmarked on an L4), Apple Silicon (MPS) and x86_64 and
arm64 CPUs. `--device` also accepts `rocm` (PyTorch's HIP build), and auto-detection picks `xpu` or `vulkan` when
PyTorch reports them; those backends are untested.

---

## Features

| Area | What the code does |
| :--- | :--- |
| KV cache | One paged pool per model (16-token pages), `native`/`int8`/`int4` storage with the layers that format damages kept in the model dtype (chosen at startup), copy-on-write forks; Quest page selection (`--kv-page-budget`), also inside CUDA graphs |
| Prefix caching | Automatic: full pages indexed by chained token hashes, reused copy-on-write, evicted LRU (`--no-prefix-caching`) |
| Speculative decoding | Prompt lookup or an EAGLE-3 draft head (`--draft-head`), `--draft-tokens` (default 8); a request drafts only while it pays; output distribution equals plain decoding for greedy and sampling |
| Weights | fp32/fp16/bf16; 4-bit GPTQ or RTN (`turing convert`, `--quantization`); TEAL-style activation sparsity in batch-1 decode on CUDA (`--activation-sparsity`, off by default) |
| LoRA | Several PEFT adapters at once (`--lora NAME=PATH`), chosen per request by `model: "base:NAME"` or `X-Turing-Adapter` |
| API | Chat templates, tools (OpenAI and Anthropic; Hermes, Mistral and Llama 3 call formats), reasoning (`reasoning_content`, Anthropic thinking blocks, effort budgets), JSON schema by token-level constrained decoding, stop strings, usage with cached tokens |
| Traffic | Admission control sized to the KV pool and three QoS lanes, on by default (429/503 with `Retry-After`) |
| Other endpoints | Embeddings, rerank, `/v1/extract`, render endpoints, Prometheus `/metrics`, llm-d KV events |
| Distributed | Tensor and pipeline parallelism (torchrun); prefill/decode disaggregation over TCP (`--prefill-worker`, `--prefill-server`) |
| Evals | `turing eval-accuracy` (GSM8K exact match) and `turing eval-niah` (pass-key retrieval), through the engine on real weights |

The `turing.experimental` package holds research components outside the serving path: cross-model KV transfer, `clean_base`
lineage, k-slot pooling, an MoE expert-offload runner and hybrid linear/softmax attention (needs a model trained with
it). What each does, and how it performed on real models, is in
[experimental components](docs/experimental.md).

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

LangChain and LlamaIndex have their own packages (Apache-2.0, no dependency on the engine):
```bash
pip install langchain-turing           # from langchain_turing import ChatTuring  (tools, streaming with usage)
pip install llama-index-llms-turing    # from llama_index.llms.turing import Turing
```
LiteLLM, KServe, Helm and llm-d: [integrations](docs/integrations.md).

---

## Documentation

* [Quickstart](https://intutic.github.io/turing/quickstart/) · [Models & hardware](https://intutic.github.io/turing/models_and_hardware/) · [Serving & API](https://intutic.github.io/turing/serving/)
* [Architecture](https://intutic.github.io/turing/architecture/) · [Benchmarks](https://intutic.github.io/turing/benchmarks/)
* [Comparison](https://intutic.github.io/turing/comparison/) · [Docker](https://intutic.github.io/turing/docker/) · [Kubernetes & llm-d](https://intutic.github.io/turing/llmd-integration/) · [Integrations](https://intutic.github.io/turing/integrations/) · [Licensing](https://intutic.github.io/turing/licensing/)

---

## License

Turing Engine is licensed under the **Business Source License 1.1 (BSL 1.1)**.
* **Free for non-production use**: development, local testing, academic research, personal single-GPU workstations and benchmarking ([`LICENSE`](LICENSE), Additional Use Grant). Production use (multi-GPU clusters, commercial hosting, revenue-generating API traffic) needs a separate commercial license.
* `packages/` is under the **Apache License 2.0** now: `kv-layer-guard` (the KV layer calibration, usable with vLLM),
  `langchain-turing` and `llama-index-llms-turing`.
* Converts automatically to **Apache 2.0** on **August 24, 2030** or the fourth anniversary of the first public release (as [`LICENSE`](LICENSE) §3 states).
