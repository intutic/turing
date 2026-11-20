# Changelog

Releases before 1.6.0 are listed only as git tags (v0.1.0 to v1.5.0).

## 2.0.0 (2026-11-20)

### Added
- Models: every documented HuggingFace family (16 model types) loads with a parity test against HuggingFace; RoPE
  linear, dynamic, YaRN, LongRoPE and Llama 3 scaling; sliding window and soft-capping in decode (baff9c5).
- GGUF: exact dequantization of 14 types in Python and C++, and loading for every supported family, checked against
  llama.cpp on real files (f90241b, 10384df).
- 4-bit weights: GPTQ and RTN through `QuantLinear`, `.tgate4` v2, layer-by-layer `turing convert`; a Triton int4 GEMM
  for decode and a Triton dequantize for prefill (cd3caa8, 27e9215).
- TEAL-style activation sparsity (`--activation-sparsity`, opt-in) with input-sparse fp16 and int4 GEMV kernels
  (44b29cd).
- KV cache: low-bit pools keep the layers their format damages in the model dtype, chosen at startup by a per-layer
  KL calibration on random tokens (Qwen2.5-1.5B: int4 +1.93 → +0.013 nats, pass-key 0% → 100%, GSM8K 52.8%);
  int4 also scales outlier key channels (16× the median or more) by powers of two, exact in any dtype; int8 is stored
  unsmoothed (smoothing cost it 16 GSM8K points in bf16). Quest page selection inside CUDA graphs (ddf24ad).
  `decode_attention` kernel|sdpa switch (8675c3d).
- Engine: prefix caching, exact prompt-lookup speculative decoding with a load gate, stop tokens, top-k/top-p, logits
  processors, multi-LoRA per batch row (f9c7bfb, 3f4d12b, 8f185aa). Requests wait for KV pages instead of failing,
  and prefill fills a token budget across requests (0a12693).
- API: one generation core behind OpenAI, Anthropic and Ollama endpoints and the DSL: chat templates, tools, reasoning
  budgets and thinking blocks, llguidance JSON-schema decoding, stop strings, usage, `--api-key`, admission and QoS
  lanes, render endpoints (21b61e8). Real embeddings, rerank, extract and pull; vLLM-format llm-d KV events;
  Prometheus `/metrics` (f9c3052).
- Distributed: tensor and pipeline parallelism over torch.distributed and prefill/decode disaggregation over TCP
  (639473d).
- C++ engine: aligned SIMD forward pass with int8 KV, NEON, and a real HTTP server (9ba5199).
- Client packages (Apache-2.0, no dependency on Turing Engine): `langchain-turing` (`ChatTuring`, moved out of
  `turing.integrations.langchain`, which now re-exports it; streams token usage and tool calls, maps LangChain's
  `tool_choice="any"`, reports the server's error message; passed LangChain's standard chat-model tests against
  Qwen2.5-0.5B-Instruct on an Apple M4 Pro in 8ecc6f8) and `llama-index-llms-turing` (`Turing`, an `OpenAILike` with Turing's defaults). The
  KServe runtime loads the InferenceService's model from `/mnt/models`, has a `/health` readiness probe and no longer
  names an unpublished image.
- MoE offload: `HeterogeneousMoERunner` runs the model's own experts from host memory with hot experts cached on the
  GPU (1c6416b).
- Evaluation: `eval-niah` and `eval-accuracy` (GSM8K) through the engine with serving settings; `eval_parity.py`,
  `eval_gguf_parity.py`, `benchmark_e2e_generation.py` (Turing vs vLLM), `benchmark_weight_formats.py`,
  `benchmark_w4a16.py`; real-model evals for cross-model KV transfer, lineage, k-slot pooling and debate (d8dc852,
  03b59f5). Results are in `docs/measured-differentiators.md`.

- Measured comparison with vLLM 0.31 and llama.cpp on an L4 (Qwen2.5-1.5B: vLLM is 1.01–1.04× faster in fp16 and
  1.06–1.12× at 4 bits; with the opt-in 4-bit LM head Turing is 1.15× vLLM at batch 1 and 8; at 16K-token prompts
  Turing's fp16 is 1.0–1.22× faster per request and vLLM's fp8 with layer 0 kept 1.3–1.4× faster than Turing's int8;
  Turing's low-bit KV keeps quality where vLLM's default fp8 and per-token int4 do not), long context at equal KV memory,
  vLLM's KV caches on Turing's own GSM8K/pass-key/NLL checks (`scripts/benchmark_e2e_generation.py`,
  `scripts/eval_vllm_quality.py`, `scripts/benchmark_llamacpp.py`).
- `scripts/make_doc_tables.py`: every numeric table in the docs is generated from the committed results; a test fails
  when they disagree.
- `packages/kv-layer-guard` (Apache-2.0): the KV layer calibration for vLLM users, validated on six instruct models
  (Qwen2.5 1.5B and 7B, Llama 3.2 3B, Gemma 2 2B, Phi-3.5-mini, Mistral 7B). It keeps layer 0 of Qwen2.5-1.5B and
  layers 0, 1, 27 of Qwen2.5-7B, which restores GSM8K under vLLM's fp8 cache from 2.4% to 70.2% on the 7B (bf16
  74.6%), and keeps none elsewhere. Limit: Phi-3.5-mini loses 5.2 GSM8K points under fp8 and the guard does not flag
  it (docs/benchmarks.md §2).

- Speed plan (all on one L4): Marlin int4 GEMM vendored from vLLM v0.31.0 (optional, `TURING_BUILD_MARLIN=1`, serves
  1–1,024 rows; eddf763); opt-in 4-bit LM head (`--quantize-lm-head`); FlashInfer prefill for pages in the model dtype
  when `flashinfer` is installed, one plan per forward for all packed prompts (30ddea4, 669c2ee); decode attention on
  tensor cores, RoPE for q and k in one launch, the residual add fused into the next norm; the first token no longer
  waits behind a decode step; int4-KV prefill with a cached prefix through gather + SDPA (6a8303c). Qwen2.5-1.5B 4-bit
  batch-1 decode 87.7 → 154.8 tok/s (196.8 with the 4-bit head).

### Changed
- Documentation rewritten against the code; every number cites a committed results file (323a854, fd2cd21). The
  README positions Turing as a research engine for KV-cache and quantization work, not a production server.
- int4 GEMM and input-sparse GEMV kernels split K across programs (1.5× faster at one row); `INT4_MAX_ROWS` 16 → 64
  from the graph-timed crossover. Under CUDA graphs 4-bit weights decode 1.5–1.6× faster than fp16.
- TEAL activation sparsity applies only to one-row decode on CUDA; prefill and batched decode stay dense (a
  sparsified prefill broke attention-sink tokens: SmolLM2-1.7B +6.4 nats at 40%).
- `jinja2>=3.1` is a dependency (transformers' chat templates refuse older versions).
- `turing` and `turing.core` import lazily (1c6416b).

### Removed
- The SVD-KV family (radix SVD forest, SVD paged kernel, SVD wire codec, hex quantizer) and the request fields that
  never changed anything (`use_svd_kv`, `sparsity_ratio` and their headers now return 400) (1c6416b, 21b61e8).
- CLI `bench`, `compare`, `transfer-bench`, `moe-bench`, `hybrid-bench`, `spec-bench` and the formula-printing
  benchmarks (1c6416b).
- The licence-key gate, the multi-model pipeline manager and its placeholder encoders (`--enable-multi-model`),
  `--memory-mode hybrid`, Matryoshka draft heads in serving (prompt-lookup speculation replaces them), and the tiled
  fused RMSNorm-SwiGLU kernel.
- The random-KV "XKV latent deliberation" demo and its fabricated 7.2x speedup; the demo's no-op KV-reuse flags
  (03b59f5, ee0bb3f).

### Fixed
- LongRoPE (Phi-3/3.5): chunked prefill picked short or long frequencies per chunk, so prompts longer than the original
  4K context mixed both and pass-key retrieval fell to 0% above 4K. Every chunk now uses what one forward over the
  prompt would; when decoding crosses the original context the engine recomputes the cache with the long frequencies,
  as transformers does; prefix-cache entries are kept apart by frequency set. Phi-3.5-mini pass-key at 2K–16K: 100%
  with int8 and int4 KV (8115830).
- Demo agents are scored on the plan they write, not on fixed configurations (ee0bb3f).
- Cross-model KV: the native RoPE strip/apply was a no-op on 4-D tensors off CUDA (03b59f5).
- CUDA graph capture and replay always run in inference mode; KV pool and int4 scratch tensors are normal tensors
  (0350b68, 13d38dd, 41692f2).
- Masked GQA prefill on CUDA no longer falls back to SDPA's math path (32K prefill ran out of memory) (93271c1).
- Cross-model KV on CUDA: a Triton inverse-RoPE kernel rotated every head after the first at the wrong position (the
  lineage eval collapsed on the L4 and not on the Mac); removed with the duplicate CUDA transfer path.

### Fixed (c547190)
- CLI: every `.gguf` path crashed with `UnboundLocalError` (missing module-level `import os`). `chat`, `serve` and
  `generate` share one weight loader, so GGUF models also get `serve`'s and `generate`'s KV cache flags. `--max-tokens` alias added;
  `--block-size` removed (KV events use the pool's 16-token pages).
- Speculative decoding: candidates, entropy and argmaxes cross to the host once per draft instead of once per tree
  node. The target's correction token no longer counts as an accepted draft token (a test had asserted 100%
  acceptance for 3 of 4 matches).
- C++ engine: `forward_token` throws past `max_pos` instead of reading beyond the KV cache; the embedded HTTP server
  answers 400 instead of crashing.
- Serving: rejected requests return 503 with `Retry-After` (KV pool full, admission shedding), 429 (admission queue
  full) or 400 (can never fit), with OpenAI, Anthropic or Ollama error bodies. Streaming endpoints admit a request
  before sending the 200.
- Telemetry: TTFT and inter-token latency are stamped after the forward pass and sampling; percentiles are null with
  fewer than two samples instead of reporting the mean; Prometheus prints NaN for them.
- `--device rocm` maps to the HIP build's `cuda` device, or explains why it cannot.

### Removed (c547190)
- The sparse prefill kernel, which was never launched.
- The Qwen 2.4T guide and its two scripts. They published a GCP project ID and a VM IP, and the model needs Blackwell
  FP4 on many GPUs. A test now rejects tracked files containing public IPs or GCP project IDs.

### Also since 1.7.0
- Expert-cache eviction fixes, a 1 MiB / queue-depth weight reader and the cold-ingest ladder (57bf57e).
- Benchmark claims audited; unmeasured numbers removed from the docs and scripts (53ddf5a, 0281443).

### Final re-audit before release
- Docs: every claim re-checked against the code and results files; the false ones corrected (test count, int8 KV
  default on CUDA, largest model run, parity coverage, cold-start comparison, bf16 labels, TEAL speed, the C++
  tokenizer, serving error shapes, the speculation gate) and the Helm and kustomize defaults moved from a 70B model to
  Qwen2.5-7B-Instruct.
- The VRAM checksum kernel hashes the tensor's bytes (it converted floats to int, so 0.25 and 0.5 hashed alike).
- The quadtree draft path no longer synchronizes with the host about 41 times per draft (fixed tree built once per
  device, quadrant picks with `torch.where`).
- `turing demo --device rocm` resolves like the other commands.
- Removed the unused `svd_compression_ratio` estimator argument and `use_svd_compression` lineage flag, and the script
  that opened the August ecosystem issues.
- `packages.yml` publishes `langchain-turing` and `llama-index-llms-turing` to PyPI from their own tags.
- Off CUDA the KV pool takes at most what 64 full-length requests need (it always took 4 GiB, which Windows commits
  up front and MPS takes from shared memory); the `gguf` and `llguidance` test dependencies install only on Python
  3.10+, where they exist. CI had not run on the v2 branch; these were its failures.

## 1.7.0 (2026-11-16, c80e7d2)

- Paged KV pool: one pool per model, sized once from free GPU memory, a device page table per request (at most one
  partial page wasted), copy-on-write forks. The engine reserves every page a request can need at admission and
  rejects what cannot fit (7f38fdd).
- Decode on CUDA: a fused quantize-and-append, split-K attention over each request's pages (optionally a Quest page
  list selected for the whole batch on the device) and a reduction; the decode step is captured as a CUDA graph per
  batch bucket and replayed.
- Pipeline stages and the distributed driver run on the new cache; the `(k, v)` tuple cache path is removed.
- HF parity for Qwen2/Qwen2.5 (q/k/v biases), Qwen3 (per-head q/k RMSNorm) and Llama-3.1 RoPE scaling; the loader
  refuses o_proj biases, sliding-window attention and unsupported RoPE scaling instead of serving wrong logits.
  `serve`, `generate` and `chat` exit when weights fail to load; random weights only with `--mock` (2cec292).
- Measured on an NVIDIA L4: SmolLM2-360M decodes 162 tok/s at batch 1 (HuggingFace transformers: 29) and 1,715 at
  batch 64 (827). int8 and int4 KV hold 1.87x and 3.17x the tokens of the native cache.

## 1.6.0 (2026-11-15, f936008)

- **Real-weight results before 1.6.0 came from a broken model.** From v0.1.0 (2026-08-25) until b9eafd7
  (2026-11-15) the HuggingFace loader never copied the per-layer RMSNorm weights (they stayed at 1.0), and under
  transformers 5 it read the wrong RoPE base (it fell back to 10000 instead of reading `rope_parameters`). b9eafd7
  fixes both, honours `head_dim` and `rms_norm_eps`, runs RMSNorm in fp32, and tests logits against HuggingFace
  through prefill, chunked prefill and decode (max difference 1.8e-7).
- KV cache with native, int8 and int4 storage and optional Quest page selection; SDPA with native GQA. On CUDA one
  fused append, split-K attention and reduce launch per layer decodes every request in the batch.
- Defaults: `--sparsity 0` and recirculation off (50% pruning measured at +6.8 to +15.8 nats NLL on SmolLM2);
  `--kv-cache-dtype auto` (int8 on CUDA).
- Quality and throughput harnesses: `scripts/eval_kv_quality.py` (paired NLL on WikiText-2) and
  `scripts/benchmark_kv_decode.py` (batched decode against HuggingFace transformers), with L4 and M4 Pro results
  (e3b18be).
- Docs quote measurements only. Removed: "75% less memory", SVD KV "-75% (2.5GB)", "57% pruning with zero accuracy
  loss", the unsourced GSM8K/HumanEval/MMLU table, NIAH 1M-token retrieval and the formula-generated "2.57x vLLM"
  serving table.
