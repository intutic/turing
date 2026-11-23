# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Releases before 1.6.0 are listed only as git tags (v0.1.0 to v1.5.0).

## Unreleased

### Added
- 4-bit MoE experts: `--quantization` and `turing convert` now pack the stacked experts of an MoE layer too (GPTQ with
  each expert's own Gram matrices), read by int4 variants of the grouped MoE kernels. `.tgate4` files written before
  still load with their experts in the model dtype.
- Marlin kernel for 4-bit MoE experts (`turing/kernels/marlin_moe.py`, `csrc/marlin_moe/`, vendored from vLLM
  v0.31.0 like the dense one and built by the same `TURING_BUILD_MARLIN=1`). With it a layer keeps its experts on
  the GPU in Marlin's layout only (the packed buffers move to host memory) and runs every input size through it.
  Qwen1.5-MoE-A2.7B in 4 bits on an L4 (LM head in fp16) goes from 98 / 259 / 446 / 506 tok/s at batch 1 / 8 / 32 /
  64 with the Triton kernels to 116 / 348 / 615 / 814, and its first token at batch 64 from 4.7 s to 1.9 s; with
  the default 4-bit LM head it runs at 146 / 376 / 638 / 830 (vLLM on Qwen's GPTQ-Int4 checkpoint: 114 / 353 /
  625 / 824).
- `--moe-expert-slots N`: MoE expert offload in the serving path. `N` experts per layer stay on the GPU (a fraction
  is an average over the layers), the rest in host memory; works with experts in the model dtype and in 4 bits.
  A 4-bit expert that is not on the GPU is computed on the CPU for decode steps of up to 8 rows
  (`csrc/turing_w4_expert_cpu.hpp`: AVX2, AVX-512 where the CPU has it, selected at run time) while the GPU runs
  the resident ones, which are kept in Marlin's layout when that extension is built; the experts used most become
  the resident ones. Such a model's decode step is captured as a chain of CUDA graphs with a stop for the host at
  each layer (`StepCuts`). Experts in the model dtype are copied in on demand.
- Prefill graphs (`PrefillGraphs`): a prefill step whose prompts all start with no cached prefix, up to 1,024
  tokens each, is replayed from a CUDA graph: a prompt that arrives alone, or a step of a burst (several prompts of
  like lengths, plus the decode rows of the requests already generating). A graph is captured at its step shape's
  second use and kept only if it beats the eager forward on that model and GPU.
  `scripts/benchmark_first_token.py` measures one prompt alone.
- `--kv-cache-dtype fp8`: KV pages in float8 E4M3 (the format of vLLM's fp8 cache), read in place by FlashInfer's
  prefill kernel and by the decode kernels, with the layer guard of the other low-bit formats. Opt-in; `auto`
  remains int8.
- `--draft-head`: speculative decoding with a published EAGLE-3 draft head (`turing/models/eagle3.py`). The head
  runs once per draft position for all drafting requests, and a request's chain is as long as its drafts are
  likely to be reached by the target. Qwen3-8B on an L4 with chains of up to 5: 50.9 / 81.8 / 132.2 tok/s on code
  prompts at 1 / 2 / 4 requests (16.5 / 26.4 / 46.1 without drafts; vLLM with the same head 50.6 / 82.8 / 133.6).
- Overlapped decode on CUDA: the next decode step is launched before the host has read the current step's tokens.
- `turing bench-bw` (also `turing doctor`) reports the device's read bandwidth and matmul rates and, with `--model`,
  the least time a batch-1 decode step can take; `--tune` measures the int4 kernel crossover on the GPU present.
- Grouped-GEMM MoE kernels (`turing/kernels/triton_moe.py`): on CUDA, `SparseMoE` sorts token-expert pairs by expert
  on the GPU and runs all experts in two Triton kernels instead of looping over them on the host, so MoE models
  (Mixtral, Qwen MoE, OLMoE) now decode under CUDA graphs. OLMoE-1B-7B on an L4 decodes 4.9–6.4× faster at batch
  1–64, with GSM8K unchanged (35.0% against 33.5%, ± 3.4, n = 200).

### Changed
- The decode step merges its attention splits in one load per query head instead of a loop over the splits, which
  took longer than the attention itself at batch 1.
- A prompt that fits one prefill forward's budget is no longer split over two forwards: it starts the next one
  whole if the rest of the current one's budget is too small (a longer prompt still takes what is left of each).
  Whole prompts need no read of a cached prefix and their forwards can be replayed from a CUDA graph.
- A prefill budget above 2,048 tokens (`ContinuousBatchEngine(prefill_chunk_size=...)`, the benchmarks'
  `--prefill-chunk`) is spent over several scheduler steps of one 2,048-token forward each: a request gets its
  first token after the forward that ends its prompt, and the requests already decoding take their next token with
  the last forward of the budget. Before, the whole budget was one forward.
- The decode kernel reads int8 KV pages under fp16/bf16 queries without dequantizing them: the codes go into the
  tensor-core products, each token's scale and zero are applied once, and a block's V tile is read together with
  its K tile. fp8 pages become fp16/bf16 values by moving bits, with the layer's scales applied to the scores and
  the output.
- A prefill graph gives the pages of its padding back after each replay, and falls back to the eager forward
  when the pool has no page for them or for the padding rows' scratch page.
- The RoPE kernel rotates a tile of positions per program instead of one vector: a 2,048-token prefill step of
  SmolLM2-1.7B spent 0.60 ms a layer in it on an L4, now 0.12 (one 16K-token prompt on Qwen2.5-1.5B: 25 ms of
  1,386, now 9 of 1,364).
- A prefill forward's row count is measured per model and GPU (`SubspaceCausalLM.calibrate_forward_rows`, once at
  startup, kept with the KV calibration): cuBLAS picks a GEMM's kernel by its row count, and the best count
  differs by layer shape. On an L4 SmolLM2-1.7B's forward costs 13% more per token from 2,049 rows than at 2,048,
  and Qwen2.5-1.5B's 3.5% less at 2,046 rows or fewer than at 2,048. A forward that cuts or continues a prompt now
  takes the cheapest count (2,040 or 2,048), the decode rows riding in it included; a forward of whole new prompts
  keeps them whole, with the rows on top, unless rows past 2,048 cost the model over 5% more and no CUDA graph
  serves the step. Only what another forward takes anyway is cut. A forward that continues a prompt is filled
  with the next prompt's first tokens instead of stopping at whole prompts.
- Steps that prefill are launched ahead, as decode steps already were (`ContinuousBatchEngine._ride_ahead`): the
  forward of the next chunks, with the decoding requests as rows, is launched from the tokens of the step in
  flight while they are still on the device, and that step's tokens are read and emitted while it runs. The GPU
  no longer waits for the host between two such steps (vLLM does the same with its asynchronous scheduling). The
  packed prefill forward makes no blocking copy to the device any more.
- With a draft head on CUDA, the GPU no longer waits for the host inside a draft-and-verify cycle where it can be
  avoided. The steps of the head's chain are queued one behind the other (the chain made three blocking copies a
  draft position), and when every request being served drafts, the step that verifies the drafts is launched
  behind the chain while they are still on the device (`ContinuousBatchEngine._verify_ahead`): the host reads the
  drafts and the target's picks together when that step is done. On Qwen3-8B on an L4 the device idled 4.8% of
  a run of one request and 5.0% of four before, 1.4% and 1.6% now (vLLM: 0.3% and 0.2%;
  `results_device_time_cuda.json`).
- A 4-bit MoE layer with a shared expert (Qwen2-MoE) served by the Marlin kernels: the shared expert's gate and up
  projections are fused like an MLP's (one kernel less a layer), and in a captured step of up to 128 rows the
  routing and the routed experts are launched on a second CUDA stream that joins the capture, so a replay runs
  them beside the shared expert (`SparseMoE._two_branches`). vLLM's step does the same, which is what its one-row
  decode on Qwen1.5-MoE was faster by. The experts' Marlin kernels have a lock array of their own for it. With
  the LM head in fp16 Qwen1.5-MoE went from 109 / 340 / 609 / 803 tok/s at batch 1 / 8 / 32 / 64 to 116 / 348 /
  615 / 814 in one boot (vLLM: 114 / 353 / 625 / 824).
- In a prefill forward over pages FlashInfer reads in place (the model dtype, fp8), the decode rows are requests of
  one query each in a FlashInfer call of their own instead of the decode kernel's three launches a layer. A burst
  of 32 prompts of 16K tokens on Qwen2.5-1.5B with an fp8 cache spent 51.6 s in its prefill steps before and 50.0
  after (the rows in the chunks' own call cost more: 52.8 s, they set its tile size). FlashInfer is no longer
  asked for heads under 64 dimensions, which it has no kernel configuration for, and its buffers are created
  outside inference mode, so a caller outside it can plan with them too.
- The paged prefill kernel tunes its tiles once for every chunk of up to 512 tokens instead of once per power of
  two below that: each new bucket stalled the process for 0.6 to 3.8 s at its first chunk (eight builds and their
  timing), and the tail a prompt leaves for the next forward can be of any length.
- `scripts/benchmark_e2e_generation.py` warms up with one prompt a forward and a few tokens long too (a short chunk
  over a cached prefix), and `scripts/eval_kv_quality.py` has `int8_decode` and `fp8_decode` arms, which feed one
  token a step through the decode kernel.
- `scripts/benchmark_e2e_generation.py` warms up with requests that end one after the other (a decode graph for
  every row count) and, when the warm-up prompts are shorter than the measured ones, with one prompt of the
  measured length for both engines, so no kernel is built inside the first measured batch.
- A model with offloaded experts no longer launches its next decode step before reading the current one's tokens:
  its step waits for the host at every layer, so the overlap only delayed each token by a step.
- `turing serve` warns at startup when the weights leave less GPU memory than twice the usual working reserve, where
  a long prompt can run out of memory mid-request.
- `scripts/benchmark_speculation.py` draws its code prompts from Python's standard library instead of from this
  repository's own source, so every engine run gets the same prompts whatever the checkout.
- KV pages on demand: a request reserves the pages of its prompt at admission and takes the pages of its output as it
  decodes, instead of reserving prompt plus `max_tokens` up front, so a large `max_tokens` no longer limits how many
  requests run at once. When the pool runs out mid-decode the newest request is preempted and its cache recomputed
  later (its stream pauses, then continues where it stopped). Admission control now budgets prompts, not prompt plus
  `max_tokens`.
- With `--quantization`, the LM head is quantized to 4 bits too (it was opt-in with `--quantize-lm-head`);
  `--no-quantize-lm-head` keeps it in the model dtype. `.tgate4` files keep the head they were written with.
- Speculative drafts are verified as rows of the batched decode step, for every request in one step, instead of in a
  forward per request; the concurrency gate (`SpeculationGatePolicy`) now applies only where that is not possible
  (Quest page selection, LongRoPE).
- Speculative decoding stops for a request while its drafts do not pay for the rows they add, measured from the
  engine's own step times over about 16 steps; prompt lookup uses an incremental n-gram index instead of scanning the sequence each step.
- `--draft-tokens` defaults to 8 (it was 4): over code, rewrite and chat prompts at 1, 2 and 8 requests on an L4,
  8 drafts gave 2.4x the plain rate (geometric mean) against 2.1x with 4, and at least 1.27x in every case.
- The prefill budget per scheduler step is 2,048 prompt tokens (1,024 and 512 for the batch and background lanes); it
  was 512 (256, 128).
- With the Marlin extension built, 4-bit linear layers use it up to 4,096 rows (it was 1,024): prefill chunks of
  2,048 tokens went through dequantize + cuBLAS, which Marlin beats by 23% at that size on an L4.
- Prefill into int8 and int4 KV layers quantizes the chunk in the append kernel and dequantizes the cached prefix
  for FlashInfer in one launch per field; a 16K-token prompt prefills at 11.2K (int8) and 11.1K (int4) tokens/s on
  an L4, from 10.6K and 10.5K (fp16 KV: 11.3K).
- A prefill writes each layer's keys and values with page indices computed once per forward instead of once per
  layer, a chunk with no cached prefix attends through plain causal SDPA instead of a paged kernel, and an idle
  scheduler starts on a new request at once instead of at its next 1 ms poll. One 512-token prompt on
  Qwen2.5-1.5B on an L4: first token after 36 ms instead of 44.
- Replacing a model's KV pool (another `kv_cache_dtype` in the same process) frees the old pool before the new one
  is sized; it used to stay allocated until then, so the new pool got only what was left.
- `turing serve` gives the KV pool all free GPU memory after the weights but a reserve (a tenth of the GPU's
  memory, at least 2 GiB, at most half of what is free) instead of 60% of it, which left 4-7 GiB of an L4 unused.
  `--kv-cache-memory-fraction` still sets a share, and a model loaded through the library keeps 60%
  (`ModelConfig.kv_pool_fraction`).
- A scheduler step that prefills no longer runs a second forward for the requests that are decoding: they ride in
  the packed prefill forward as rows of one token (`SubspaceCausalLM.prefill_packed(..., riders=N)`, attention through
  the paged decode kernel), as vLLM schedules a step, and their drafts are verified there too. Engines with a
  draft head or Quest page selection decode as before.
- MoE layers run inputs of 1,024 or more rows per expert as one cuBLAS GEMM pair per expert; the grouped kernels use
  32-row blocks and 64 x 128 tiles, and skip the sort for a single token.
- MoE layers on CUDA launch about twenty fewer small kernels each: routing (softmax, top-k, normalisation) is one
  kernel instead of six, token-expert pairs are grouped by expert in three kernels instead of about fourteen
  PyTorch operations, the sum over a token's experts is one, and a single token's pair table is built once.
- Decode graphs for 12, 24, 48 and 96 rows too: a step is padded to the next graph size, and with powers of two
  only, 40 requests paid for 64 rows.
- The 4-bit MoE kernels and the dense batch-1 4-bit kernel read 256 columns of each weight row per step (whole
  quantization groups, 128 contiguous bytes) instead of 64; one token's experts run in a kernel without a
  tensor-core tile. On an L4 the expert kernels take 17-33% less time per layer
  (`results_moe_kernel_w4_qwen_cuda.json`, `results_moe_kernel_w4_olmoe_cuda.json`).
- Layers of an int8 or int4 KV cache take FlashInfer's prefill attention when it is installed, through temporary
  dequantized pages.
- Experimental components moved to `turing.experimental` (`turing/experimental/`, kernels in
  `turing/experimental/kernels/`, the demo in `turing/experimental/demo/`, tests in `tests/experimental/`): every module
  `turing serve` does not import. Imports such as `turing.core.lineage` or `turing.kernels.triton_kslot_pool` become
  `turing.experimental.lineage` and `turing.experimental.kernels.triton_kslot_pool`; `turing.SubspaceStructuredRouter`,
  `turing.DynamicEntropyRouter` and `turing.HierarchicalVirtualPageManager` are now in `turing.experimental`.

### Removed
- The untrained draft heads (`turing.experimental.speculation`: Matryoshka and quadtree heads, tree pruner), their two
  Triton kernels and C++ helpers.
- Experimental modules with no model result and no role in inference, with their tests and benchmark scripts:
  swarm and particle-swarm optimisers, the router and its annealer, hyper-connections (`mhc`), optimal-transport
  eviction, compressed convolutional attention (`cca`), the attention-pattern cache, the hybrid tensor mesh, matrix
  power, unified memory, the persistent GEMV, the async scheduler, and 26 kernels (among them a fused Adam optimizer,
  2-D convolution and Laplacian kernels, latent decode, Vulkan and ROCm helpers). Twelve one-off benchmark scripts that
  only exercised them are removed too. `turing serve` is unchanged.
- C++ extension (`turing.turing_csrc`): 34 functions and classes that nothing in the package or the standalone engine
  calls, with 31 headers and their tests. Among them `subspace_quantize`, `TuringPagedAttentionEngine`,
  `RadixTrieIndex`, `LRUExpertCacheFast`, `LFRUExpertCacheFast`, `NativeReasoningFilter`, `sinkhorn_ot_eviction`,
  `birkhoff_project`, `fused_adam_step`, the segment-pool and int8 serializer helpers, and the particle-swarm, matrix
  power, Laplacian and conv kernels. The macOS arm64 extension shrinks from 742 KB to 507 KB.
- 22 scripts that no document, test or workflow referenced: synthetic benchmark suites, an unused wheel builder and
  early GPU smoke scripts. The scripts left are listed in `benchmarks/README.md`.
- Benchmark results moved from the repository root to `benchmarks/results/`, with reproduction commands in
  `benchmarks/README.md`; scripts write there by default.
- Documentation reorganised into Getting started, User guide, Performance and Internals sections.
- Package description corrected (it still advertised 70B models on one GPU).

### Fixed
- The CUDA image could not start on a GPU: it had no C compiler, which Triton needs at its first kernel, and no
  library path for the NVIDIA driver where GKE mounts it. `deploy/Dockerfile` now installs `gcc` in the CUDA image
  and sets `LD_LIBRARY_PATH=/usr/local/nvidia/lib64`. Found by running `integrations/kserve` on GKE with KServe
  v0.21.0 (`benchmarks/results/results_kserve_gke.json`); with both added to the `v2.0.0-cuda` image at container
  start, the InferenceService was Ready 57 s after apply and served requests. The published `v2.0.0-cuda` image
  keeps the defect.
- `turing doctor --tune` kept `int4_max_rows = 16` on a GPU where the fused int4 GEMM was slower at every row
  count it measured (a Tesla T4): its grid now starts at 4 rows and it keeps 0 when the kernel never wins.
- The KV pool's reserve of working memory has a floor of 2 GiB instead of 1.5: on a 15 GB T4 a 2,048-token
  prefill of Qwen2.5-1.5B without FlashInfer ran out of memory. An L4's reserve (a tenth of its memory) does not
  change.
- With a draft head, the step times that decide whether drafting pays were wrong at both ends of a run of verify
  steps launched ahead: the first sample spanned two cycles and the one after the last about 0.1 ms. A run of
  more drafted rows than the largest decode graph holds (128) is no longer set up for a launch that is then
  refused, which read each request's drafts in a copy of its own.
- Once every prefill graph of a model had been dropped (none beat its eager forward), the next capture failed
  inside PyTorch's allocator (`use_count > 0`) and with it the request: the graphs' shared memory pool had been
  released with the last of them. A capture with no graph left now starts a new pool.
- A model switched from an int4 KV pool to another storage kept int4's key smoothing, and the layer guard for the
  new storage was then measured, and cached on disk, with smoothed keys. Smoothing is now dropped when the storage
  is not int4.
- The shared buffer 4-bit layers are dequantized into was freed when a larger one replaced it, while CUDA graphs
  captured with the old one still wrote there on replay.
- Overlapped decode waited for the step in flight before launching the next one: the next step's rows were selected
  with an index built on the host, which PyTorch copies to the GPU with a blocking copy, and a request's new KV pages
  were written to the page table the same way. The GPU idled for the host's launch time every step. The index is
  gone when every request continues, and page-table rows go through pinned memory without waiting.
- The KV pool's page table starts with 256 request rows instead of 8. Each time it grew, every decode graph captured
  before was captured again (about 0.1 s each), so the first requests after a burst stalled once.
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
