# 🌐 Models & Hardware

---

## 1. Supported model types

The HuggingFace loader accepts exactly the model types in `hf_loader.FAMILIES` and refuses every other one before
downloading (`ArchitectureRegistry` raises for the same set). Each type has a parity test against HuggingFace in
`tests/test_hf_parity.py` (prefill, chunked prefill, decode, stateless calls and batched decode across requests,
atol 1e-3, on small randomly initialised models of each configuration):

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

RoPE scaling types: default, linear, dynamic NTK, YaRN, LongRoPE, Llama 3 and proportional (the last without a
parity configuration of its own). Weights are written from the checkpoint into a model built on the meta device, and
any parameter left unwritten is an error, so a model never runs with random weights; `--mock` runs synthetic weights
on purpose.

### Real checkpoints

* **HuggingFace, real weights**: SmolLM2 135M, 360M and 1.7B match HuggingFace's mean NLL to five decimals
  ([benchmarks §2](benchmarks.md#2-quality-on-real-weights)). `scripts/eval_parity.py` compares other real
  checkpoints with HuggingFace (logit difference, top-1 agreement, NLL, and HF's own cached-vs-full gap as the noise
  floor for bf16); its results are not committed yet.
* **GGUF, real files against llama.cpp** (`scripts/eval_gguf_parity.py`, `results_gguf_parity_mps.json`): Apple
  M4 Pro, Turing on MPS in fp32 and llama.cpp (llama-cpp-python 0.3.37) on Metal with all layers offloaded, the first
  64 tokens of WikiText-2 test. NLL is per token; the HF column is the original HuggingFace checkpoint in fp32, not
  the GGUF. "llama.cpp Metal vs CPU" is llama.cpp's own difference between its two backends, a scale for the first
  column.

| Family | GGUF file | Quantized types | max \|Δlogit\| vs llama.cpp | top-1 agreement | llama.cpp Metal vs CPU max \|Δlogit\| | NLL Turing | NLL llama.cpp | NLL HF fp32 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| llama | `Llama-3.2-1B-Instruct-Q8_0.gguf` | Q8_0 | 0.037 | 100.0% | 0.431 | 3.813 | 3.812 | 3.809 |
| mixtral (`llama`) | `tiny-mixtral-30m-q8_0.gguf` | Q8_0 | 0.015 | 100.0% | 0.168 | 4.654 | 4.653 | 4.643 |
| qwen2 | `qwen2.5-0.5b-instruct-q8_0.gguf` | Q8_0 | 0.074 | 100.0% | 1.321 | 3.353 | 3.353 | 3.373 |
| qwen3 | `Qwen3-0.6B-Q8_0.gguf` | Q8_0 | 0.039 | 100.0% | 3.999 | 3.884 | 3.884 | 3.909 |
| gemma | `gemma-1.1-2b-it.Q8_0.gguf` | Q8_0 | 0.080 | 100.0% | 1.461 | 5.937 | 5.937 | 5.913 |
| gemma2 | `gemma-2-2b-it-Q8_0.gguf` | Q8_0 | 0.030 | 100.0% | 0.870 | 3.522 | 3.522 | 3.517 |
| gemma3 | `gemma-3-270m-it-Q8_0.gguf` | Q8_0 | 0.200 | 100.0% | 5.516 | 5.172 | 5.173 | 5.153 |
| phi2 | `phi-2.Q8_0.gguf` | Q8_0 | 0.691 | 98.4% | 2.649 | 3.377 | 3.370 | 3.383 |
| phi3 | `Phi-3-mini-4k-instruct-q4.gguf` | Q4_K, Q5_K, Q6_K | 0.251 | 98.4% | 5.222 | 2.933 | 2.932 | 2.772 |
| phi3 (LongRoPE) | `Phi-3.5-mini-instruct-Q8_0.gguf` | Q8_0 | 0.247 | 100.0% | 8.282 | 2.828 | 2.828 | 2.832 |
| gpt2 | `gpt2.Q8_0.gguf` | Q8_0 | 0.164 | 100.0% | 2.770 | 4.427 | 4.427 | 4.412 |
| gptneox | `pythia-160m.Q8_0.gguf` | Q8_0 | 1.469 | 96.9% | 5.433 | 4.133 | 4.126 | 4.167 |
| olmoe | `olmoe-1b-7b-0924-q8_0.gguf` | Q8_0 | 0.961 | 100.0% | 0.973 | 3.195 | 3.198 | 3.200 |

Qwen2-MoE and Qwen3-MoE were not run: their smallest real checkpoints (14.3B and 30.5B parameters) do not fit this
machine's memory in fp32.

### Not run

No model larger than the 7B-total-parameter OLMoE file above has been run end to end, no MoE model on a GPU, and no
real checkpoint through the tensor- or pipeline-parallel path.

---

## 2. Memory

Weights take 2 bytes per parameter in fp16/bf16 and 4 in fp32 (the default on CPU and MPS). With 4-bit weights
(`turing convert` or `--quantization`), attention and dense-MLP projections take half a byte per weight plus an fp16
scale and a uint8 zero per 64 weights; embeddings, the LM head, routers and MoE experts stay in the model dtype. The
KV pool takes `--kv-cache-memory-fraction` (0.6) of the GPU memory left after the weights. A 7B model in fp16 (about
15 GB of weights) leaves room for a KV pool on a 24 GB GPU; a 70B model needs about 141 GB in fp16 and does not fit
on one 24 GB GPU even at 4 bits per projection weight.

---

## 3. Model names

```bash
turing serve --model Qwen/Qwen2.5-7B-Instruct          # HuggingFace repo id
turing serve --model /path/to/checkpoint/               # local HuggingFace directory
turing serve --model ./models/gemma-2-2b-it-Q8_0.gguf   # llama.cpp GGUF file
turing serve --model ./qwen7b-gptq                      # .tgate4 directory from turing convert
turing chat --model deepseek-r1-1.5b                    # alias
```

Aliases (`ModelResolver.CLI_ALIASES` in `turing/models/resolver.py`) are typing shortcuts for repo ids, for example
`smollm2` → `HuggingFaceTB/SmolLM2-135M`. An alias does not mean the model was tested, and an alias resolving to a
model type outside the table above is refused at load. `SIZING_PROFILES` (`turing/models/sizing_profiles.py`) holds
layer geometries for `--mock` synthetic models only.

---

## 4. Hardware

| Backend | Status |
| :--- | :--- |
| NVIDIA CUDA | Measured on an L4: Triton decode kernels, CUDA graphs, int4 GEMM and sparse GEMV kernels (the last two have CUDA parity tests; their speed is not yet measured) |
| Apple Silicon (MPS) | Measured on an M4 Pro. Attention takes an explicit path, because PyTorch 2.9's MPS SDPA reuses one cached graph per shape with the first call's scale |
| CPU | PyTorch on CPU; the C++ extension and `turing-cli` use AVX2/FMA (x86_64) or NEON (arm64) |
| AMD ROCm | `--device rocm` uses PyTorch's HIP build (exposed as `cuda`); no recorded run |
| Intel XPU, Vulkan | Picked by `--device auto` when PyTorch reports them; no recorded run |

```bash
turing info                      # detected devices and runtime
turing serve --model smollm2 --device mps
```
