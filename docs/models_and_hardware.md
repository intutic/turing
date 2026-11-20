# Models & Hardware

## Supported model types

The HuggingFace loader accepts the model types in `hf_loader.FAMILIES` and refuses any other type before downloading
(`ArchitectureRegistry` refuses the same set). Each type has a parity test against HuggingFace in
`tests/test_hf_parity.py`, run on small randomly initialised models of each configuration: prefill, chunked prefill,
decode, stateless calls and batched decode across requests, atol 1e-3.

| HF `model_type` | Examples | Configurations tested | GGUF |
| :--- | :--- | :--- | :---: |
| `llama` | Llama 2/3/3.1/3.2/3.3, SmolLM2, DeepSeek-R1-Distill-Llama | head_dim ≠ hidden/heads; RoPE default, linear, dynamic, YaRN, LongRoPE, Llama 3 | ✅ |
| `mistral` | Mistral 7B | sliding window | ✅ (as `llama`) |
| `qwen2` | Qwen2/2.5, DeepSeek-R1-Distill-Qwen | q/k/v biases | ✅ |
| `qwen3` | Qwen3 | q/k norms | ✅ |
| `gemma` | Gemma 1 | | ✅ |
| `gemma2` | Gemma 2 | soft-capping, sliding window | ✅ |
| `gemma3_text` | Gemma 3 (text) | local/global RoPE bases | ✅ |
| `phi` | Phi-2 | with and without q/k LayerNorm | ✅ |
| `phi3` | Phi-3, Phi-3.5 | sliding window; LongRoPE | ✅ |
| `gpt2` | GPT-2 | | ✅ |
| `opt` | OPT | pre-LN; post-LN with projected embeddings (OPT-350m) | ❌ |
| `gpt_neox` | Pythia, GPT-NeoX | parallel and sequential residual | ✅ |
| `mixtral` | Mixtral | top-2 MoE | ✅ (as `llama`) |
| `qwen2_moe` | Qwen1.5-MoE, Qwen2-MoE | shared expert | ✅ |
| `qwen3_moe` | Qwen3-MoE | | ✅ |
| `olmoe` | OLMoE | `clip_qkv`, full-projection q/k norms | ✅ |

Supported RoPE scaling types are default, linear, dynamic NTK, YaRN, LongRoPE, Llama 3 and proportional (proportional
has no parity configuration of its own).

The loader builds the model on the meta device and writes the checkpoint's weights into it. A parameter left
unwritten is an error, so a model never runs with random weights. `--mock` runs synthetic weights deliberately.

### Parity on real checkpoints

`scripts/eval_parity.py` takes the first 64 tokens of the WikiText-2 test split, prefills half, decodes the rest token
by token through the cache, and compares the logits with HuggingFace's eager forward.

- In fp32 the difference is rounding.
- In bf16, the "noise floor" column is HuggingFace against itself (its own cached decode against its full forward)
  and gives the scale for the column before it.
- ΔNLL is the difference a user would notice.

Apple M4 Pro, fp32; OLMoE and Mistral-7B in bf16 to fit in memory. Source: `benchmarks/results/results_parity_mps.json`.

<!-- table:parity_mps -->
| Checkpoint | Type | dtype | max \|Δlogit\| vs HF | HF cached vs full (noise floor) | Top-1 agreement | ΔNLL vs HF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| gpt2 | gpt2 | float32 | 0.000412 | — | 100.0% | -0.0000 |
| facebook/opt-125m | opt | float32 | 0.000114 | — | 100.0% | -0.0000 |
| facebook/opt-350m | opt | float32 | 5.82e-05 | — | 100.0% | +0.0000 |
| EleutherAI/pythia-160m | gpt_neox | float32 | 0.0147 | — | 100.0% | +0.0000 |
| HuggingFaceTB/SmolLM2-135M | llama | float32 | 6.1e-05 | — | 100.0% | -0.0000 |
| unsloth/gemma-3-270m | gemma3_text | float32 | 9.58e-05 | — | 100.0% | +0.0000 |
| Qwen/Qwen2.5-0.5B | qwen2 | float32 | 0.000303 | — | 100.0% | -0.0000 |
| Qwen/Qwen3-0.6B | qwen3 | float32 | 0.000102 | — | 100.0% | -0.0000 |
| unsloth/Llama-3.2-1B | llama | float32 | 0.000143 | — | 100.0% | +0.0000 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | qwen2 | float32 | 0.000125 | — | 100.0% | -0.0000 |
| unsloth/gemma-2-2b | gemma2 | float32 | 0.000105 | — | 100.0% | +0.0000 |
| microsoft/phi-2 | phi | float32 | 0.00768 | — | 100.0% | +0.0000 |
| microsoft/Phi-3-mini-4k-instruct | phi3 | float32 | 0.000501 | — | 100.0% | +0.0000 |
| microsoft/Phi-3.5-mini-instruct | phi3 | float32 | 0.000234 | — | 100.0% | +0.0000 |
| allenai/OLMoE-1B-7B-0924 | olmoe | bfloat16 | 1.27 | — | 98.4% | -0.0053 |
| mistralai/Mistral-7B-v0.1 | mistral | bfloat16 | 0.5 | — | 95.3% | +0.0002 |
<!-- /table -->

NVIDIA L4, bf16. Source: `benchmarks/results/results_parity_cuda.json`.

<!-- table:parity_cuda -->
| Checkpoint | Type | dtype | max \|Δlogit\| vs HF | HF cached vs full (noise floor) | Top-1 agreement | ΔNLL vs HF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| gpt2 | gpt2 | bfloat16 | 4 | 4.000 | 92.2% | -0.0026 |
| facebook/opt-125m | opt | bfloat16 | 0.188 | 0.312 | 96.9% | -0.0017 |
| facebook/opt-350m | opt | bfloat16 | 0.312 | 0.375 | 98.4% | -0.0016 |
| HuggingFaceTB/SmolLM2-135M | llama | bfloat16 | 0.875 | 0.750 | 93.8% | -0.0061 |
| unsloth/gemma-3-270m | gemma3_text | bfloat16 | 0.672 | 0.594 | 100.0% | -0.0046 |
| Qwen/Qwen2.5-0.5B | qwen2 | bfloat16 | 1.38 | 0.484 | 100.0% | +0.0151 |
| Qwen/Qwen3-0.6B | qwen3 | bfloat16 | 0.625 | 1.000 | 98.4% | -0.0073 |
| unsloth/Llama-3.2-1B | llama | bfloat16 | 0.266 | 0.312 | 100.0% | +0.0048 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | qwen2 | bfloat16 | 6.97 | 1.188 | 73.4% | -0.2354 |
| unsloth/gemma-2-2b | gemma2 | bfloat16 | 0.704 | 0.453 | 98.4% | +0.0023 |
| microsoft/phi-2 | phi | bfloat16 | 3.52 | 1.530 | 89.1% | +0.0052 |
| microsoft/Phi-3-mini-4k-instruct | phi3 | bfloat16 | 1.75 | 1.766 | 98.4% | +0.0001 |
| allenai/OLMoE-1B-7B-0924 | olmoe | bfloat16 | 1.89 | 1.262 | 96.9% | +0.0200 |
<!-- /table -->

MoE checkpoints on an x86_64 CPU, bf16. Qwen1.5-MoE-A2.7B is the smallest real `qwen2_moe` checkpoint. Source:
`benchmarks/results/results_parity_cpu.json`.

<!-- table:parity_cpu -->
| Checkpoint | Type | dtype | max \|Δlogit\| vs HF | HF cached vs full (noise floor) | Top-1 agreement | ΔNLL vs HF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| allenai/OLMoE-1B-7B-0924 | olmoe | bfloat16 | 1.72 | 0.938 | 100.0% | +0.0099 |
| Qwen/Qwen1.5-MoE-A2.7B | qwen2_moe | bfloat16 | 0.891 | 0.625 | 96.9% | +0.0033 |
<!-- /table -->

### GGUF parity against llama.cpp

`scripts/eval_gguf_parity.py` compares Turing on MPS in fp32 with llama.cpp (llama-cpp-python) on Metal with all
layers offloaded, on an Apple M4 Pro, over the first 64 tokens of the WikiText-2 test split.

- NLL is per token. The HF column is the original checkpoint in fp32, not the GGUF.
- "llama.cpp Metal vs CPU" is llama.cpp's own difference between its two backends and gives the scale for the first
  column.

Source: `benchmarks/results/results_gguf_parity_mps.json`.

<!-- table:gguf -->
| Family | GGUF file | Quantized types | max \|Δlogit\| vs llama.cpp | Top-1 agreement | llama.cpp Metal vs CPU max \|Δlogit\| | NLL Turing | NLL llama.cpp | NLL HF fp32 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | `Llama-3.2-1B-Instruct-Q8_0.gguf` | Q8_0 | 0.037 | 100.0% | 0.431 | 3.813 | 3.812 | 3.809 |
| llama-mixtral | `tiny-mixtral-30m-q8_0.gguf` | Q8_0 | 0.015 | 100.0% | 0.168 | 4.654 | 4.653 | 4.643 |
| qwen2 | `qwen2.5-0.5b-instruct-q8_0.gguf` | Q8_0 | 0.074 | 100.0% | 1.321 | 3.353 | 3.353 | 3.373 |
| qwen3 | `Qwen3-0.6B-Q8_0.gguf` | Q8_0 | 0.039 | 100.0% | 3.999 | 3.884 | 3.884 | 3.909 |
| gemma | `gemma-1.1-2b-it.Q8_0.gguf` | Q8_0 | 0.080 | 100.0% | 1.461 | 5.937 | 5.937 | 5.913 |
| gemma2 | `gemma-2-2b-it-Q8_0.gguf` | Q8_0 | 0.030 | 100.0% | 0.870 | 3.522 | 3.522 | 3.517 |
| gemma3 | `gemma-3-270m-it-Q8_0.gguf` | Q8_0 | 0.200 | 100.0% | 5.516 | 5.172 | 5.173 | 5.153 |
| phi2 | `phi-2.Q8_0.gguf` | Q8_0 | 0.691 | 98.4% | 2.649 | 3.377 | 3.370 | 3.383 |
| phi3 | `Phi-3-mini-4k-instruct-q4.gguf` | Q4_K, Q5_K, Q6_K | 0.251 | 98.4% | 5.222 | 2.933 | 2.932 | 2.772 |
| phi3-longrope | `Phi-3.5-mini-instruct-Q8_0.gguf` | Q8_0 | 0.247 | 100.0% | 8.282 | 2.828 | 2.828 | 2.832 |
| gpt2 | `gpt2.Q8_0.gguf` | Q8_0 | 0.164 | 100.0% | 2.770 | 4.427 | 4.427 | 4.412 |
| gptneox | `pythia-160m.Q8_0.gguf` | Q8_0 | 1.469 | 96.9% | 5.433 | 4.133 | 4.126 | 4.167 |
| olmoe | `olmoe-1b-7b-0924-q8_0.gguf` | Q8_0 | 0.961 | 100.0% | 0.973 | 3.195 | 3.198 | 3.200 |
<!-- /table -->

### Limitations

- The largest model run end to end is Qwen2.5-7B (7.6B parameters).
- No `qwen3_moe` checkpoint has been run; the smallest, Qwen3-30B-A3B, needs about 61 GB in bf16.
- No real checkpoint has been run through the tensor- or pipeline-parallel path.

## Memory

- **Weights:** 2 bytes per parameter in fp16/bf16, 4 in fp32 (the default on CPU and MPS).
- **4-bit weights** (`turing convert` or `--quantization`): attention and dense-MLP projections take half a byte per
  weight plus an fp16 scale and a uint8 zero per 64 weights. Embeddings, the LM head, routers and MoE experts stay in
  the model dtype.
- **KV cache:** the pool takes `--kv-cache-memory-fraction` (default 0.6) of the GPU memory left after the weights.

A 7B model in fp16 (about 15 GB of weights) leaves room for a KV pool on a 24 GB GPU. A 70B model needs about 141 GB
in fp16 and does not fit on one 24 GB GPU, even at 4 bits per projection weight.

## Model names

```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct          # HuggingFace repo id
turing serve --model /path/to/checkpoint/               # local HuggingFace directory
turing serve --model ./models/gemma-2-2b-it-Q8_0.gguf   # llama.cpp GGUF file
turing serve --model ./qwen7b-gptq                      # .tgate4 directory from turing convert
turing chat --model deepseek-r1-1.5b                    # alias
```

Aliases (`ModelResolver.CLI_ALIASES` in `turing/models/resolver.py`) are shortcuts for repo ids, for example
`smollm2` → `HuggingFaceTB/SmolLM2-135M`. An alias says nothing about whether the model has been tested, and an alias
that resolves to a model type outside the table above is refused at load. `SIZING_PROFILES`
(`turing/models/sizing_profiles.py`) holds layer geometries for `--mock` synthetic models only.

## Hardware

| Backend | Status |
| :--- | :--- |
| NVIDIA CUDA | Tested on NVIDIA L4: Triton decode kernels, CUDA graphs, int4 GEMM and input-sparse GEMV kernels ([benchmarks](benchmarks.md#3-4-bit-weights-and-activation-sparsity)) |
| Apple Silicon (MPS) | Tested on Apple M4 Pro. Attention uses an explicit path because PyTorch 2.9's MPS SDPA reuses one cached graph per shape with the first call's scale |
| CPU | PyTorch on CPU; the C++ extension and `turing-cli` use AVX2/FMA (x86_64) or NEON (arm64) |
| AMD ROCm | `--device rocm` uses PyTorch's HIP build (exposed as `cuda`); untested |
| Intel XPU, Vulkan | Selected by `--device auto` when PyTorch reports them; untested |

```bash
turing info                      # detected devices and runtime
turing serve --model smollm2 --device mps
```
