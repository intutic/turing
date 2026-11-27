# Plan: CPU-first inference, in waves

Spec: [SPEC.md](SPEC.md). Evidence: [research.md](research.md). Branch: `feat/cpu-first-inference`.
Every wave ends with its numbers measured against llama.cpp on the same machine, parity tests for new kernels, the
`python -W error -m pytest` suite green, and a results JSON under `benchmarks/results/`.

## Decisions needed before Wave 1 (CF = CPU-first)
- **CF1 Kernel base.** (A) Turing's C++ engine calls ggml's CPU kernels (MIT, linked as a library) for the matrix
  products and adds the new parts on top; (B) write every kernel in `csrc/` ourselves; (C) contribute to llama.cpp
  instead of Turing. Recommendation A: llama.cpp's kernels already have ARM repacking, KleidiAI SME2, AMX and
  ternary types, and matching them from scratch is months before any new idea is tested. B keeps everything ours.
- **CF2 First target.** Single user on the M4 Pro (where a CPU can get close to a GPU) or a multi-user CPU server.
  Recommendation: the Mac first; the server in Wave 7.
- **CF3 Where measurements run.** M4 numbers can only come from this Mac, and a run loads it for minutes. Server
  numbers need a rented x86 VM (GCP C4 / C4D, no GPU quota needed; costs money each time).
- **CF4 Lossy modes.** Ternary, activation sparsity and fewer experts per token cost quality. Recommendation: build
  them as opt-in modes that print their measured quality cost; exact stays the default.

## Wave 0: baselines and measurement harness
- CPU-only memory bandwidth of the M4 Pro (STREAM-style probe): the ceiling every target divides into.
- llama.cpp baselines on this Mac, CPU-only and Metal: Qwen3-30B-A3B Q4_K_M, an 8B Q4, gpt-oss-120b streamed by
  mmap from the SanDisk and from the internal SSD (download in progress), `llama-bench` JSON.
- A routing-trace recorder for MoE models (which experts each token uses) and a cache simulator: hit rate against
  cache size and pinned set, before any streaming code is tuned.
- Files: `scripts/benchmark_cpu_decode.py`, `scripts/moe_route_trace.py`; quality through `scripts/eval_kv_quality.py`.

## Wave 1: a CPU backend for Turing built on llama.cpp (CF1)
- Design (2026-11-27, after reading llama.cpp at 171527522): link libllama, the model code that sits on ggml's kernels,
  pinned to one commit under `third_party/`, rather than rewriting each architecture's graph on bare ggml. libllama
  already runs Qwen3, Qwen3-MoE and gpt-oss with ARM repacking, KleidiAI and AMX, so Turing's work starts at the new
  parts. A C++ shim (`csrc/cpu/`: load, prefill, decode, logits, sampling hooks) is bound into Python, and
  `turing serve --device cpu --model x.gguf` runs it behind Turing's OpenAI / Anthropic / Ollama APIs.
- Replaces the fp32 standalone engine for speed; that engine stays as the dependency-free path.
- Accept: logits of an F16 GGUF within atol 1e-3 of Hugging Face's on a small model; decode speed equal to llama-bench on
  the same VM (same kernels); the existing suite green with the backend absent (optional build, like Marlin).

## Wave 2: models larger than RAM, experts read ahead of the expert products
- Design: an eval callback on `ffn_moe_topk-<layer>` (llama.cpp calls it after the router and before the expert
  products, `ggml_backend_sched_compute_splits`) reads the chosen experts' byte ranges, found from the GGUF's tensor
  offsets, with parallel `pread` in >= 1 MB chunks (`csrc/turing_parallel_reader.hpp`, io_uring on Linux), skipping
  ranges `mincore` reports resident. The products then read them from the page cache instead of faulting 4-16 KB at a
  time. Exact by construction: no routing changes.
- Then: a hot set pinned from Wave 0's routing traces; next-layer prefetch measured, kept only if it pays.
- Accept: token-identical output with and without the prefetcher; on a VM whose memory is capped below the model
  (cgroup `MemoryMax`) with local NVMe, decode >= 2x llama.cpp's plain mmap, or the measured reason it cannot.

## Wave 3: speculative decoding on the CPU path
- Prompt lookup first (no extra model), then a small draft model / EAGLE-3 heads (Turing has both on CUDA); draft length
  tuned per device, switched off when acceptance makes it slower.
- Accept: >= 1.3x on code and math prompts, >= 1.0x on chat, outputs identical to plain greedy decoding.

## Wave 4: ternary and 2-bit
- GGUF TQ1_0 / TQ2_0 / I2_S in the loaders; lookup-table kernels after T-MAC / bitnet.cpp TL (MIT, notices kept).
- Accept: BitNet-b1.58-2B4T and Llama3-8B-1.58 run; >= 1.5x over Q4 of a same-size model at batch 1; quality as delta
  NLL with standard error beside the speed.

## Wave 5: activation sparsity on the CPU
- TEAL thresholds (Turing already calibrates them for CUDA in `turing/models/activation_sparsity.py`) with
  input-channel-major weights so a zero activation skips a contiguous block.
- Accept: >= 1.2x at 40% on a dense 8B at batch 1, delta NLL reported; off by default.

## Wave 6: matrix-unit prefill
- Apple SME2 on M4 and Intel AMX on Xeon through KleidiAI / oneDNN for batched prompt products.
- Accept: >= 2x time-to-first-token over NEON / AVX-512 on a 2,048-token prompt.

## Wave 7: CPU server
- The continuous-batching engine on the CPU path with shared prefixes and INT8; a rented Xeon 6 or EPYC VM.
- Accept: tok/s at batch 1-64 against llama.cpp server and vLLM-CPU on the same VM, with tok/s per dollar beside a
  GPU's published figure. No parity claim with GPUs.

## Order and overlap
Wave 0 first. Wave 1 is the base for 2-6. After Wave 1, Waves 2, 3 and 5 touch different files and can run side by
side; 4 and 6 follow 1. Wave 7 last.

## Status (2026-11-27)
- **Wave 0 done** (`benchmarks/results/results_cpu_baseline_c4a.json`): c4a-standard-16 reads 283 GB/s; llama.cpp decodes
  Llama-3.1-8B Q4_K_M at 38.7 tok/s and Qwen3-30B-A3B at 64.9; Turing's fp32 C++ engine 0.73.
- **Wave 1 done**: libllama backend, `turing serve --backend libllama`; matches llama-bench on the same VM (Qwen3-8B 39.2
  vs 38.6 tok/s decode, Qwen3-30B-A3B 65.1 vs 67.2) and Hugging Face's logits within 1e-3.
- **Wave 2 done, adaptive** (`results_cpu_waves123_c4a.json`, `results_cpu_waves23_c4a.json`): gpt-oss-120b (63 GB) from
  a 2.7 GB/s SSD with the cache capped at 28 GB decodes at 12.1 tok/s with the read-ahead against 3.16 with plain mmap
  (3.8x; prompt 47.9 against 4.6, 10.5x), at 20 GB 8.09 against 1.96 (4.1x); text identical. With 40 GB of cache (4.4%
  of expert bytes missing) plain mmap was faster (17.4 against 12.8), so the read-ahead now reads only while >= 5% of
  the chosen experts' bytes miss the cache. Reading beside the products (async) paid less than before them (blocking).
  A 48 GB machine already runs gpt-oss-120b from SSD at usable speed with plain mmap.
- **Wave 3 done, gated**: prompt lookup with Qwen3's chat template, thinking off: a code edit decodes 37.4 tok/s against
  28.7 (1.31x, 72% of drafts accepted, text identical); an open prompt 36.7 against 37.4 (the gate drafted 12 tokens in
  198 steps). Ungated, verifying 4 drafts costs ~2.5 one-token steps on this CPU. Greedy only in exact arithmetic:
  batched and one-token kernels round apart with 4-bit weights and fp16 KV, so a near-tie can flip a token.
- **Wave 4 done** (`results_cpu_waves45_c4a.json`, `results_cpu_waves46_c4.json`): BitNet-b1.58-2B-4T, run through
  `third_party/patches/0001`, has the same WikiText-2 perplexity in TQ2_0 / TQ1_0 as in F16 (15.070 / 15.070 / 15.074
  +/- 0.43) and decodes 85.3 tok/s in TQ2_0 against 41.4 in Q4_K_M on a Xeon 6 (2.06x); prompt processing is 2.4x
  slower. Llama-3.1-8B (not trained ternary, speed only) decodes 1.48x faster in TQ2_0 on a Neoverse-V2.
- **Wave 5 no-go as designed**: a column-sparse 4-bit kernel streams 29-38 GB/s against ggml's 126-154 GB/s dense, 2x
  slower than dense even at 50% sparsity. TEAL on the CPU needs a different kernel (skip units of a cache line or more,
  outputs held in registers) before a llama.cpp patch is worth writing; not built.
- **Wave 7 built** (`--max-batch-size N`): several requests decode together, exact against HF; throughput being measured.
- **Wave 6 measured on x86** (`results_cpu_waves46_c4.json`): AMX through ggml's extra buffer types speeds Qwen3-8B
  prompt processing 3.3x for Q4_0 (187 against 57 tok/s) and Q8_0 (201 against 61) on a Xeon 6, 1.13x for Q4_K_M.
  Turing's backend uses those buffer types by default (repack on). SME2 on Apple M4 is unmeasured (CF3: cloud only).
