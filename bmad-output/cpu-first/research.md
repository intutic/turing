# Research: can CPUs replace GPUs for LLM inference? (2026-11-27)

Four parallel web reviews of the ideas in the user's pasted chat, each claim checked against papers, code and
benchmarks. Verdicts: VERIFIED, CORRECTED (right figure given), NOT FOUND. Nothing here was run locally except the
two drive measurements marked "measured".

## The two physical facts everything else follows from
- **One user (batch 1) is memory-bandwidth bound.** Each token reads every active weight once, so tokens/s is about
  bandwidth / bytes per token. Fewer bits per weight, fewer active weights (MoE, sparsity), and reusing one read for
  several tokens (speculative decoding) are what help. At 4 bits or more the CPU waits for memory; at 2 bits or less
  unpacking becomes the limit, which is where lookup-table kernels pay.
- **Many users are compute bound.** Batching shares each weight read across users, so a CPU runs out of FLOPs:
  ~21 TFLOPS AMX per Xeon socket against ~990 dense BF16 (1,979 FP8) on one H100, a 25-50x gap, while the bandwidth
  gap is only 2-5x (CPU sockets ship at 614-1,200 GB/s). Best published CPU serving: 2x Xeon 6980P, Llama2-7B INT8,
  661 tok/s at batch 15 (Intel, 2025-03-13); one H100 serves Llama 3.1 8B FP8 at 15,220 tok/s (NVIDIA TRT-LLM).

## Drives on this Mac (measured, F_NOCACHE, `scratchpad/ssd/diskbench.py`)
| Drive | Link | Sequential read |
|---|---|---|
| Internal SSD, MacBook Pro M4 Pro | - | 4.97 GB/s (write 3.71) |
| SanDisk Extreme Pro Portable V2 ("Extreme Pro 55AF") | USB 10 Gb/s (Macs have no USB 3.2 Gen 2x2) | 0.97 GB/s (write 1.04) |

## Claims from the chat
| Claim | Verdict |
|---|---|
| 100B BitNet at 5-7 tok/s on one CPU | CORRECTED: 6.58 tok/s on an 800 GB/s M2 Ultra, an untrained dummy model; 1.70 on an i7-13700H ([2410.16144](https://arxiv.org/html/2410.16144v2)). No trained 100B BitNet exists |
| bitnet.cpp 1.37-5.07x on ARM | VERIFIED, but against llama.cpp fp16 |
| T-MAC up to 4x over llama.cpp | CORRECTED: end to end 1.1x at 4-bit, 2.3x at 2-bit (M2 Ultra) ([2407.00088](https://arxiv.org/html/2407.00088v2)) |
| T-MAC 71 tok/s M2 Ultra, 11 tok/s Pi 5 | VERIFIED, for BitNet-3B |
| MADDNESS: matmul as O(log N) tree lookups | CORRECTED: fixed-depth trees; no LLM result exists |
| PowerInfer-2: Mixtral 47B 11.68 tok/s on a phone | CORRECTED: TurboSparse-Mixtral (retrained, 150B tokens), mostly in RAM; 2.13 tok/s with 7 GB ([2406.06282](https://arxiv.org/html/2406.06282v3)) |
| LLM in a Flash: 2x DRAM size | VERIFIED, needs ReLU sparsity and trained predictors; OPT-6.7B ~1.5 tok/s on M1 Max CPU ([2312.11514](https://arxiv.org/abs/2312.11514)) |
| 30-35B MoE from SSD on a 16 GB laptop at 10-25 tok/s | VERIFIED for 11-15 tok/s ([llama.cpp #23324](https://github.com/ggml-org/llama.cpp/discussions/23324)) |
| "Edge0" by "AutoArc" | CORRECTED: Edge0 by AutoArk exists; its predictor *changes* routing (4 of 8 experts), -3.9 points avg ([2609.18063](https://arxiv.org/html/2609.18063v2)) |
| 80-90% of neurons near zero | CORRECTED: ReLU models only; SwiGLU models give 40-50% without training (TEAL) |
| FFF / tree MoE cuts compute 100x | CORRECTED: 78x on BERT's FFN, trained from scratch; no conversion of trained LLMs ([2311.10770](https://arxiv.org/abs/2311.10770)) |
| SVD cuts >95% with almost no loss | CORRECTED: 20% costs +1.4-2.3 ppl on LLaMA-7B; 60% is ppl 15 even with LoRA ([SVD-LLM](https://arxiv.org/html/2403.07378v5)) |
| Monarch/butterfly O(N log N) | CORRECTED: Monarch is O(N^1.5); trained weights convert badly ([BLAST](https://arxiv.org/html/2410.21262v2)) |
| H-matrices keep weights in L2/L3 | NOT FOUND (one preprint, no speed, code 404) |
| Wavelet/SSM attention replacement | NOT FOUND for conversion; linear-attention conversions keep 79-90% of MMLU (LoLCATs, Llamba) |
| JIT-baking weights halves memory reads | CORRECTED: bytes still come from DRAM as instructions; no LLM work does it. "jitllm" exists but JIT-compiles GPU kernels |
| Speculative decoding 2-4x on CPU, lossless | PARTLY: lossless, typically 1.0-2.0x on CPU; can slow down (chat 1.01x) |
| Distillation "healing" recovers 98-99% | CORRECTED: 99-101% on averages, 79-93% on MMLU |
| PIM-AI: 6.94x lower cost, 10-20x lower energy | CORRECTED: simulated only ([2411.17309](https://arxiv.org/html/2411.17309v1)); 10-20x is vs phone chips, vs GPUs 15-40% less energy |
| UPMEM DDR5 DIMMs, 70B in RAM, free cloud | CORRECTED / NOT FOUND: DDR4 only, no 70B run published, no free cloud found |
| Samsung/SK hynix PIM enterprise-only | CORRECTED: not for sale at all |
| Tenstorrent n150d $999-1,099, 70B on 4 cards | Price VERIFIED; 70B needs 4 dual-chip n300 (QuietBox, $15k) |
| Hailo-8 runs LLMs | CORRECTED: no DRAM, cannot; Hailo-10H runs <= 2B |
| Mythic 100x | NOT SUPPORTED for LLMs |

## What is real and worth building (ranked across the four reviews)
1. **Low-bit CPU kernels at full bandwidth** (int4/int8, NEON dotprod/i8mm, AVX-512 VNNI, repacked layouts) and CPU
   MoE paths. Every other gain is measured against this; Turing has none of it on ARM today.
2. **Exact expert streaming for models larger than RAM**: non-expert weights resident, whole experts read with
   parallel `pread` in >= 1 MB chunks (flash-moe found mmap 5x slower than pread; macOS's own file cache beat a
   hand-built LRU by 38%), cache sized from routing traces. Estimated gpt-oss-120b on this Mac: ~7-14 tok/s from
   the internal SSD, ~3-4 from the SanDisk (arithmetic, unmeasured).
3. **Speculative decoding on the CPU path**, reusing Turing's EAGLE-3 and prompt lookup; 1.0-2.0x, auto-off.
4. **Ternary / 2-bit with lookup-table kernels** (T-MAC, bitnet.cpp TL kernels, MIT): ~1.5-2x over Q4; ternary
   post-training costs ~10 points at 8B, 4-5 at 32-70B (CAT-Q, ICML 2026).
5. **TEAL activation sparsity on the CPU**: 1.2-1.5x at 25-40%, needs input-channel-major weights.
6. **SME2 (M4: 8.4 TOPS int8 peak) / AMX prefill** via KleidiAI / oneDNN: time-to-first-token, little decode gain.
7. **CPU server serving** with shared prefixes (Turing has a prefix cache) and INT8 AMX; honest ceiling 15-40% of an
   H100 for 8B, but it can hold 1.5 TB MoE models in RAM.

## Open risks
CPU-only bandwidth on the M4 Pro is unmeasured; gpt-oss-120b's expert hit rate at a ~35 GB cache is unmeasured;
most CPU serving numbers are vendor-published; paper baselines are often old llama.cpp or fp16, so real gaps today
are smaller; laptops throttle under sustained load.
