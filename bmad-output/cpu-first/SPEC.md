# SPEC: CPU-first inference in Turing Engine

Branch `feat/cpu-first-inference`. Sources for every number: [research.md](research.md). Plan: [plan.md](plan.md).

## Problem
Large-model inference needs a GPU today because CPUs read model weights more slowly, multiply less per clock, and
cannot hold models larger than RAM. The user wants to run frontier-class open models on CPUs they already own,
starting with a MacBook Pro M4 Pro (48 GB unified memory, internal SSD 4.97 GB/s, SanDisk Extreme Pro V2 on USB
0.97 GB/s, both measured 2026-11-27), and later on CPU servers. Turing's CPU path today runs fp32 matrix-vector
products on dequantized weights (8x the bytes of a 4-bit model), has no ARM low-bit kernels, and its activation
sparsity and MoE offload exist only for CUDA.

## Capabilities
1. Low-bit CPU kernels that read packed weights directly (4/8-bit GGUF k-quants, MXFP4) on NEON (dotprod, i8mm)
   and AVX-512/VNNI, at llama.cpp's CPU speed or better, including MoE expert paths.
2. Models larger than RAM: experts streamed from SSD with large parallel reads, a hot set kept resident, the cache
   sized from recorded routing traces; outputs identical to a fully resident run.
3. Speculative decoding on the CPU path (prompt lookup, draft model / EAGLE-3 heads), switched off where it slows.
4. Ternary and 2-bit models (TQ1_0, TQ2_0, I2_S) with lookup-table kernels.
5. Activation sparsity (TEAL) on the CPU for dense models.
6. Matrix-unit prefill: Apple SME2 on M4, Intel AMX on Xeon, through vendor libraries.
7. Multi-user CPU serving on the existing continuous batcher, measured on a rented CPU server.

## Constraints
- Every speed or quality claim is measured on the stated hardware, against llama.cpp on the same machine and build
  date; quality as delta NLL with a standard error (`scripts/eval_kv_quality.py` pattern). No "faster than" without
  a head-to-head.
- New kernels carry a reference parity test (atol <= 1e-3), as CLAUDE.md rule 2 requires.
- Exact by default: lossy modes (ternary, sparsity, fewer experts) are opt-in and report their quality cost.
- Work lands on `feat/cpu-first-inference`, reaching master only through an approved PR.
- Business Source License 1.1 for Turing; ported MIT/Apache code (ggml, T-MAC, bitnet.cpp, KleidiAI) keeps its
  notices.

## Non-goals (the research found no evidence they work on trained LLMs, or no way to buy them)
H-matrix or Monarch conversion of trained weights; weights compiled into machine code ("JIT-baking"); wavelet
attention; fast-feed-forward trees; MADDNESS / product-quantized activations; ReLU retraining; neuron-level flash
reads; trained routing predictors; processing-in-memory hardware (simulator study only, if ever). Parity with GPUs
for high-load multi-user serving, per watt or per dollar: the research puts a two-socket CPU server at 15-40% of
one H100's 8B throughput at best.

## Success metrics (single user, batch 1, M4 Pro CPU cores unless stated)
| Metric | Target |
|---|---|
| Qwen3-30B-A3B Q4 decode, in RAM | >= llama.cpp CPU on the same Mac (research estimate 30-45 tok/s) |
| gpt-oss-120b (63 GB) decode, streamed | >= 2x llama.cpp mmap; >= 7 tok/s from the internal SSD, >= 3 from the SanDisk |
| Speculative decoding | >= 1.3x on code/math prompts, never below 1.0x (auto-off) |
| Ternary 2-4B | >= 1.5x over Q4 of the same model, quality cost reported |
| TEAL 40% on a dense 8B | >= 1.2x, delta NLL reported |
| SME2 prefill | >= 2x time-to-first-token over NEON |
| CPU server (rented Xeon 6) | throughput vs llama.cpp and vLLM-CPU at batch 1-64, tok/s per $ beside a GPU's |
