# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases before 1.6.0 are listed only as git tags (v0.1.0 to v1.5.0).

## Unreleased

### Changed
- Experimental components moved to `turing.experimental` (`turing/experimental/`, kernels in
  `turing/experimental/kernels/`, the demo in `turing/experimental/demo/`, tests in `tests/experimental/`): every module
  `turing serve` does not import. Imports such as `turing.core.lineage` or `turing.kernels.triton_kslot_pool` become
  `turing.experimental.lineage` and `turing.experimental.kernels.triton_kslot_pool`; `turing.SubspaceStructuredRouter`,
  `turing.DynamicEntropyRouter` and `turing.HierarchicalVirtualPageManager` are now in `turing.experimental`.

### Removed
- Experimental modules with no model result and no role in inference, with their tests and benchmark scripts:
  swarm and particle-swarm optimisers, the router and its annealer, hyper-connections (`mhc`), optimal-transport
  eviction, compressed convolutional attention (`cca`), the attention-pattern cache, the hybrid tensor mesh, matrix
  power, unified memory, the persistent GEMV, the async scheduler, and 26 kernels (among them a fused Adam optimizer,
  2-D convolution and Laplacian kernels, latent decode, Vulkan and ROCm helpers). Twelve one-off benchmark scripts that
  only exercised them are removed too. `turing serve` is unchanged.
- Benchmark results moved from the repository root to `benchmarks/results/`, with reproduction commands in
  `benchmarks/README.md`; scripts write there by default.
- Documentation reorganised into Getting started, User guide, Performance and Internals sections.
- Package description corrected (it still advertised 70B models on one GPU).

### Fixed
- Ollama `/api/generate` templates containing `{{` in a prompt, system message or response were refused; only the
  template's own text is checked now.

### Security
- Ollama template rendering is linear in the template's length (a crafted template made the old regular expression
  take seconds, CodeQL `py/polynomial-redos`).
- `/api/pull`, `/v1/extract`, `/v1/embeddings` and `/api/embed` no longer put exception text in responses; pull
  errors are logged on the server and the client gets a message naming the repository (CodeQL
  `py/stack-trace-exposure`).

## 2.0.0 - 2026-11-20

### Breaking
- **Python 3.10 or newer.** Wheels are built for Python 3.10–3.12 on Linux x86_64, Linux aarch64 and macOS arm64.
- Request fields and headers that never changed anything (`use_svd_kv`, `sparsity_ratio`, `X-Turing-SVD-*`,
  `X-Turing-Sparsity`) now return HTTP 400.
- Removed CLI commands: `bench`, `compare`, `transfer-bench`, `moe-bench`, `hybrid-bench`, `spec-bench`. Removed flags:
  `--enable-multi-model`, `--memory-mode hybrid`, `--block-size`.

### Added
- **Models:** 16 HuggingFace model types, each with a parity test against HuggingFace; RoPE linear, dynamic, YaRN,
  LongRoPE and Llama 3 scaling; sliding-window attention and soft-capping.
- **GGUF:** exact dequantization of 14 GGML types in Python and C++; every supported family loads from GGUF and is
  checked against llama.cpp on real files.
- **4-bit weights:** GPTQ and RTN (`turing convert`, `--quantization`), the `.tgate4` v2 format, a Triton int4 GEMM,
  an optional vendored Marlin kernel (`TURING_BUILD_MARLIN=1`, from vLLM v0.31.0) and an opt-in 4-bit LM head
  (`--quantize-lm-head`).
- **KV cache calibration:** int8 and int4 pools keep the layers their format damages in the model dtype, chosen at
  startup by a per-layer KL check on random tokens; int4 also scales outlier key channels by powers of two. On
  Qwen2.5-1.5B this takes int4 from +1.93 to +0.013 nats and pass-key retrieval from 0% to 100%.
- **Engine:** prefix caching, exact prompt-lookup speculative decoding with a load gate, multi-LoRA per batch row,
  requests that wait for KV pages instead of failing, and prefill packed across requests. FlashInfer prefill is used
  for model-dtype pages when `flashinfer` is installed.
- **API:** one generation core behind the OpenAI, Anthropic and Ollama endpoints and the DSL: chat templates, tools,
  reasoning budgets and thinking blocks, JSON-schema constrained decoding (llguidance), stop strings, usage,
  `--api-key`, admission control with three QoS lanes, render endpoints, embeddings, rerank, extract and pull,
  llm-d KV events and Prometheus `/metrics`.
- **Distributed:** tensor and pipeline parallelism over `torch.distributed`, and prefill/decode disaggregation over
  TCP.
- **C++ engine:** an aligned SIMD forward pass with an int8 KV cache, NEON support and an HTTP server (`turing-cli`).
- **TEAL activation sparsity** (`--activation-sparsity`, opt-in) with input-sparse fp16 and int4 GEMV kernels.
- **MoE expert offload:** `HeterogeneousMoERunner` runs a model's own experts from host memory, with hot experts
  cached on the GPU.
- **Evaluation:** `turing eval-accuracy` (GSM8K) and `turing eval-niah` (pass-key retrieval) through the engine;
  scripts for HF and GGUF parity, Turing-vs-vLLM and llama.cpp benchmarks, and vLLM's KV formats on the same checks.
- **Packages** (Apache-2.0): `kv-layer-guard` (the KV layer calibration for vLLM users, validated on six instruct
  models), `langchain-turing` (`ChatTuring`) and `llama-index-llms-turing` (`Turing`). The latter two are on PyPI.
- **Distribution:** wheels and the source package on PyPI as `turing-engine`; CPU and CUDA images at
  `ghcr.io/intutic/turing`; KServe runtime and Helm chart use the published CUDA image.

### Changed
- The KV cache defaults to int8 on CUDA (`--kv-cache-dtype auto`) and to the model's dtype elsewhere.
- Off CUDA the KV pool takes at most what 64 full-length requests need, instead of always 4 GiB.
- TEAL sparsity applies only to one-row decode on CUDA; prefill and batched decode stay dense.
- `turing` and `turing.core` import lazily; `jinja2>=3.1` is a dependency.
- Documentation rewritten against the code; every table is generated from a results file.

### Fixed
- **LongRoPE (Phi-3/3.5):** chunked prefill mixed short and long frequencies, so pass-key retrieval fell to 0% above
  4K tokens. Every chunk now uses the frequencies one forward over the prompt would, and the cache is recomputed when
  decoding crosses the original context. Phi-3.5-mini pass-key at 2K–16K is 100% with int8 and int4 KV.
- Every `.gguf` model crashed the CLI with `UnboundLocalError`.
- Rejected requests return 503 with `Retry-After` (KV pool full), 429 (admission queue full) or 400 (can never
  fit), in the error shape of the API used; streaming endpoints admit a request before sending the 200.
- TTFT and inter-token latency are timed after the forward pass; percentiles are null with fewer than two samples.
- Speculative decoding no longer counts the target's correction token as an accepted draft token, and drafting
  synchronizes with the host once per draft instead of per tree node.
- The C++ engine stops at `max_pos` instead of reading past the KV cache, and its HTTP server answers 400 on bad
  requests instead of crashing.
- CUDA graph capture and replay always run in inference mode; masked GQA prefill on CUDA no longer falls back to
  SDPA's math path (32K prefill ran out of memory).
- The GPU tensor checksum hashes bytes (0.25 and 0.5 used to hash alike).
- Cross-model KV transfer: RoPE strip/apply was a no-op on 4-D tensors off CUDA, and a CUDA kernel rotated heads at
  the wrong position.
- `--device rocm` maps to the HIP build's `cuda` device, including in `turing demo`.

### Removed
- The SVD-KV family (radix SVD forest, SVD paged kernel, SVD wire codec, hexagonal quantizer).
- The licence-key gate, the multi-model pipeline manager, Matryoshka draft heads in serving (prompt-lookup
  speculation replaces them), the never-launched sparse prefill kernel, the tiled RMSNorm-SwiGLU kernel and the
  random-KV "latent deliberation" demo.

## 1.7.0 - 2026-11-16

### Added
- Paged KV pool: one pool per model, sized once from free GPU memory, a page table per request and copy-on-write
  forks. Requests reserve every page they can need at admission.
- CUDA decode: fused quantize-and-append, split-K attention over each request's pages (optionally a Quest page
  selection) and a reduction, captured as a CUDA graph per batch size.
- HF parity for Qwen2/Qwen2.5, Qwen3 and Llama 3.1 RoPE scaling; the loader refuses architectures it would serve
  wrongly, and `serve`, `generate` and `chat` exit when weights fail to load (random weights only with `--mock`).

### Changed
- Pipeline stages and the distributed driver use the paged cache; the `(k, v)` tuple cache path is removed.

## 1.6.0 - 2026-11-15

### Fixed
- **HuggingFace weights:** the loader never copied the per-layer RMSNorm weights, and under transformers 5 it read
  the wrong RoPE base, so real-weight results from v0.1.0 to 1.5.0 came from a broken model. Logits are now tested
  against HuggingFace through prefill, chunked prefill and decode.

### Added
- KV cache with native, int8 and int4 storage and optional Quest page selection; fused CUDA decode.
- `scripts/eval_kv_quality.py` (paired WikiText-2 NLL) and `scripts/benchmark_kv_decode.py`.

### Changed
- Defaults: `--sparsity 0`, recirculation off, `--kv-cache-dtype auto`.

### Removed
- Unsupported performance claims from the documentation (memory savings, pruning accuracy, an unsourced benchmark
  table, a formula-generated serving comparison).
