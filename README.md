# ⚡ Turing Engine

**An inference server that reproduces HuggingFace's logits for 16 model families, with a paged int8/int4 KV cache, OpenAI, Anthropic and Ollama APIs in one process, and a standalone C++ engine.**

[![Docs: Live](https://img.shields.io/badge/Documentation-intutic.github.io%2Fturing-blue.svg)](https://intutic.github.io/turing/)
[![Release: v2.0.0](https://img.shields.io/badge/Release-v2.0.0-blue.svg)](CHANGELOG.md)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-green.svg)](LICENSE)
[![Tests: 667 (49 need CUDA)](https://img.shields.io/badge/Tests-667%20%2849%20need%20CUDA%29-brightgreen.svg)](https://github.com/intutic/turing/actions)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/intutic/turing/blob/master/demo/turing_quickstart_colab.ipynb)

---

## What it is, and what is measured

Turing Engine loads a HuggingFace checkpoint, a llama.cpp GGUF file or its own 4-bit `.tgate4` directory and serves
it through a continuous-batching engine. By default the model is exact: every lossy option (KV quantization, Quest
page selection, 4-bit weights, activation sparsity, FFN pruning) is opt-in.

**Measured** (raw results are committed as `results_*.json`; see [benchmarks](docs/benchmarks.md)):

* **HuggingFace parity on real weights.** SmolLM2 135M–1.7B: the exact model's mean NLL equals HuggingFace's to five
  decimals (`results_serving_defaults_mps.json`).
* **GGUF parity with llama.cpp.** 13 real GGUF files (Llama 3.2, a tiny Mixtral, Qwen2.5, Qwen3, Gemma 1/2/3, Phi-2,
  Phi-3 and Phi-3.5, GPT-2, Pythia, OLMoE) agree with llama.cpp on the top-1 token at 97–100% of 64 positions
  (`results_gguf_parity_mps.json`, Apple M4 Pro).
* **KV cache quality and throughput.** The int8 cache costs no measurable NLL on SmolLM2 135M–1.7B; int4 costs about
  +0.02 nats. On an NVIDIA L4, SmolLM2-360M at 2K context decoded 162 tok/s for one request (HuggingFace
  transformers: 29) and 1,715 tok/s at batch 64 (827) with the v1.7.0 engine
  (`results_kv_quality_cuda.json`, `results_kv_decode_cuda_smollm2_360m.json`).
* **C++ engine.** `turing-cli` decodes a synthetic 8-layer model at 125.6 tok/s on one M4 Pro core, 7.4× the
  previous release's binary (`results_cpp_engine_mps.json`).

**Not measured yet:** any comparison with vLLM, SGLang, TensorRT-LLM or llama.cpp throughput
(`scripts/benchmark_e2e_generation.py` runs Turing and vLLM on the same prompts; its GPU run is pending); the speed
of 4-bit weights and of activation sparsity; tensor and pipeline parallelism on GPUs (tested only with CPU
processes); MoE checkpoints larger than OLMoE-1B-7B.

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
| KV cache | One paged pool per model (16-token pages), `native`/`int8`/`int4` storage, copy-on-write forks; Quest page selection (`--kv-page-budget`), also inside CUDA graphs |
| Prefix caching | Automatic: full pages indexed by chained token hashes, reused copy-on-write, evicted LRU (`--no-prefix-caching`) |
| Speculative decoding | Prompt lookup (no draft model), `--draft-tokens` (default 4); output distribution equals plain decoding for greedy and sampling |
| Weights | fp32/fp16/bf16; 4-bit GPTQ or RTN (`turing convert`, `--quantization`); TEAL-style activation sparsity (`--activation-sparsity`, off by default) |
| LoRA | Several PEFT adapters at once (`--lora NAME=PATH`), chosen per request by `model: "base:NAME"` or `X-Turing-Adapter` |
| API | Chat templates, tools (OpenAI and Anthropic; Hermes, Mistral and Llama 3 call formats), reasoning (`reasoning_content`, Anthropic thinking blocks, effort budgets), JSON schema by token-level constrained decoding, stop strings, usage with cached tokens |
| Traffic | Admission control sized to the KV pool and three QoS lanes, on by default (429/503 with `Retry-After`) |
| Other endpoints | Embeddings, rerank, `/v1/extract`, render endpoints, Prometheus `/metrics`, llm-d KV events |
| Distributed | Tensor and pipeline parallelism (torchrun); prefill/decode disaggregation over TCP (`--prefill-worker`, `--prefill-server`) |
| Evals | `turing eval-accuracy` (GSM8K exact match) and `turing eval-niah` (pass-key retrieval), through the engine on real weights |

The repository also holds library components that are not in the serving path and have no measurement on a model:
cross-model KV transfer, `clean_base` lineage, k-slot pooling, hybrid linear/softmax attention (needs a model
trained with it), Matryoshka draft heads (untrained), an MoE expert-offload runner and assorted Triton and C++
kernels. See [architecture](docs/architecture.md) and the [claims audit](docs/claims-audit.md).

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
* Converts automatically to **Apache 2.0** on **August 24, 2030** or the fourth anniversary of the first public release (as [`LICENSE`](LICENSE) §3 states).
