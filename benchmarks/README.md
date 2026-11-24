# Benchmarks

`results/` holds the JSON files behind every number in the README and the docs. Each file records the script that
wrote it, its arguments and the environment (GPU, PyTorch and CUDA versions). The tables in the docs are generated
from these files:

```bash
python scripts/make_doc_tables.py          # rewrite the tables in README.md and docs/
python scripts/make_doc_tables.py --check  # fail if any table differs from the results files
```

## Hardware

GPU results (`*_cuda.json`) come from one NVIDIA L4 (24 GB) on a Google Cloud `g2-standard-16`, PyTorch 2.9 with
CUDA 12.9; vLLM 0.31 and llama.cpp ran on the same machine. It is a Spot VM: it was restarted twice during the
session of 2026-11-21 and twice on 2026-11-22 (02:42 and 10:48 UTC), so rows of one table can come from different
physical GPUs, and each file's `note` names the session its rows are from. The comparisons with vLLM were measured
again on a second VM (2026-11-22 and 23, restarted once at 00:44 UTC: the 4-bit MoE and EAGLE-3 tables, in one
boot) and on a third (2026-11-23 and 24, six boots: the 512-token, short-prompt and 16K tables) in blocks of the
rows that are compared with each other; a block runs within one boot, and every row carries its block and boot.
llama.cpp's 4-bit rows are from the third VM, in the same blocks as the rows beside them. The two smallest
comparisons were repeated there too (`results_moe_w4_speed_repeat_cuda.json`, `results_eagle3_repeat_cuda.json`),
the MoE model also converted with `turing convert --symmetric` (`results_moe_w4_gsm8k_symmetric_cuda.json` is its
GSM8K). `results_4bit_speed_no_marlin_cuda.json` is the 4-bit main row in a build
without the Marlin extension. `*_mps.json` files come from an Apple M4 Pro.

## Reproducing the main comparison

Qwen2.5-1.5B-Instruct, 512-token prompts, 256 new tokens, batches 1, 8 and 64:

```bash
M=Qwen/Qwen2.5-1.5B-Instruct
OUT=benchmarks/results/results_4bit_speed_cuda.json
SP="--context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out $OUT"

python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP                                     # fp16
python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq
python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq --quantize-lm-head
python scripts/benchmark_e2e_generation.py --engine vllm   --model $M $SP                                     # in a vLLM environment
python scripts/benchmark_e2e_generation.py --engine vllm   --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 $SP
python scripts/benchmark_llamacpp.py bench --gguf qwen2.5-1.5b-instruct-q4_k_m.gguf --batches 1,8,64 --out $OUT
```

Each engine runs at its own step budget unless told otherwise: Turing's engine takes 2,048 prompt tokens per forward
(2,040 where that measured cheaper for the model), vLLM's `LLM` class 8,192 tokens per step on an L4. `--prefill-chunk 8192` (Turing) and `--max-num-batched-tokens 2048`
(vLLM) give the other engine's budget; every row records the one it ran with. The comparison's step-budget table has
SmolLM2-360M and SmolLM2-1.7B (`--context 1024 --batches 1,8,32,64`, `results_e2e_generation_cuda.json`) and the 16K
rows below at both.

GSM8K for the same settings (500 problems, 8-shot, greedy):

```bash
turing eval-accuracy --model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32 \
  --out benchmarks/results/results_4bit_quality_turing_cuda.json
python scripts/eval_vllm_quality.py --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 --dtype float16 --checks gsm8k \
  --out benchmarks/results/results_4bit_quality_vllm_cuda.json
```

## Reproducing the KV-cache comparison

16K-token prompts at equal KV memory, and quality (GSM8K, pass-key retrieval, WikiText-2 NLL) per cache format:

```bash
LC="--model $M --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out benchmarks/results/results_longctx_cuda.json"
python scripts/benchmark_e2e_generation.py --engine vllm   $LC --kv-cache-dtype native     # records vLLM's KV memory
python scripts/benchmark_e2e_generation.py --engine turing $LC --kv-cache-dtype int8 --kv-pool-gib-from benchmarks/results/results_longctx_cuda.json
python scripts/benchmark_e2e_generation.py --engine vllm   $LC --vllm-kv-cache-dtype fp8 --kv-skip-layers 0

python scripts/eval_kv_quality.py --help          # NLL per KV format
turing eval-niah --help                           # pass-key retrieval through the engine
python scripts/eval_vllm_quality.py --help        # the same checks for vLLM's cache formats
```

## Other measurements

| Script | Results | Docs |
| :--- | :--- | :--- |
| `scripts/eval_parity.py`, `scripts/eval_gguf_parity.py` | `results_parity_*.json`, `results_gguf_parity_mps.json` | Models & hardware |
| `scripts/benchmark_kv_decode.py` | `results_kv_decode_*.json` | Benchmarks |
| `scripts/benchmark_w4a16.py` | `results_w4a16_crossover_*.json`, `results_w4a16_prefill_rows_cuda.json` (256 to 4,096 rows), `results_w4a16_gemv_groups_cuda.json` (the few-row kernel of a build without Marlin) | Benchmarks |
| `scripts/benchmark_cold_start.py` | `results_cold_start_*.json` | Comparison |
| `scripts/benchmark_e2e_generation.py --model allenai/OLMoE-1B-7B-0924-Instruct --dtype bf16`, `turing eval-accuracy` | `results_moe_speed_*.json`, `results_moe_gsm8k_*.json` | Benchmarks |
| `turing convert --method gptq`, then `scripts/benchmark_e2e_generation.py --model <converted directory>` and `turing eval-accuracy` (Qwen1.5-MoE-A2.7B-Chat; vLLM on `Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4`) | `results_moe_w4_speed*_cuda.json`, `results_moe_w4_gsm8k_cuda.json`, `results_moe_w4_quality*_cuda.json` | Benchmarks |
| `scripts/benchmark_e2e_generation.py --moe-expert-slots N`, `turing eval-accuracy --moe-expert-slots N`; for Mixtral-8x7B a 4-bit directory from `turing convert` with `--moe-expert-slots 6`, against `scripts/benchmark_llamacpp.py bench --ngl 26` on its Q4_K_M GGUF | `results_moe_offload_serve_cuda.json`, `results_moe_offload_gsm8k_cuda.json` | Benchmarks |
| `scripts/benchmark_first_token.py` (`--kv-cache-dtype int8` for the pool `turing serve` uses on CUDA) | `results_first_token_cuda.json`, `results_first_token_int8_cuda.json` | Benchmarks |
| `scripts/benchmark_e2e_generation.py --batches 12,24,48` | `results_batch_sizes_cuda.json` | Comparison |
| `scripts/benchmark_moe_offload_olmoe.py` | `results_moe_offload_cuda.json` | Benchmarks |
| `scripts/benchmark_ingest_ladder_all_tiers.py`, `scripts/benchmark_ring_reader_ab.py` | `results_ingest_ladder*.json`, `results_ring_reader_ab_mps.json` | Benchmarks |
| `scripts/benchmark_cpp_engine.py` | `results_cpp_engine_*.json` | Benchmarks |
| `scripts/benchmark_prefill_attention.py` | `results_prefill_attention*_cuda.json` | — |
| `scripts/profile_decode_step.py` | `results_decode_profile_*.json` (kernel launches and device time per decode step, bandwidth floor) | — |
| `scripts/profile_engine_ttft.py` | first-token latency breakdown | — |
| `scripts/benchmark_weight_formats.py` | prints batch-1 decode speed and weight memory per weight format | — |
| `scripts/benchmark_vectorized_batch_union.py` | prints MoE offload against `SparseMoE` on one layer of a small Mixtral checkpoint | — |
| `scripts/diag_int4_kv.py`, `scripts/diag_teal.py` | print per-layer int4 KV error and the cost of TEAL per projection | — |
| `scripts/benchmark_moe_kernel.py` | `results_moe_kernel_*_cuda.json` (grouped MoE kernels by block and tile size, against one GEMM pair per expert; `--layers 16` reads the weights cold, `--w4` packs the experts in 4 bits) | Benchmarks |
| `scripts/benchmark_speculation.py`, `scripts/diag_spec_divergence.py` | `results_speculation_cuda.json`, `results_speculation_repeat_cuda.json` (one cell four more times), `results_spec_divergence_cuda.json`; with `--draft-head RedHatAI/Qwen3-8B-speculator.eagle3 --model Qwen/Qwen3-8B --dtype bf16`: `results_eagle3_cuda.json`, `results_eagle3_repeat_cuda.json` (the code workload at chains of up to 5 three more times; the last with both engines' output counted alike), `results_spec_divergence_eagle3_cuda.json` | Benchmarks |
| `turing doctor`, `turing doctor --model … --tune` and `scripts/benchmark_e2e_generation.py` on a Tesla T4: two VMs, the second also at a lowered power limit, with its decode step by kernel and its cold start | `results_t4_cuda.json` | Models & Hardware |
| `integrations/kserve` on a GKE cluster with KServe v0.21.0, and the published and a rebuilt CUDA image under Docker on an L4 (applied by hand, no script) | `results_kserve_gke.json` | Integrations |
| `turing serve` over HTTP: chat requests' time to the first streamed token and the server's stages, before and after the vocabulary lookup was cached (scratch client and tracer, no script in the repository) | `results_serve_first_token_cuda.json` | Integrations |
| Five bursts of 8 and of 64 prompts of 512 tokens (Qwen2.5-1.5B) and of 8 and 32 prompts of 1,024 tokens (SmolLM2-1.7B) in each engine, every request's first-token time (scratch script around `benchmark_e2e_generation.py`'s prompts) | `results_burst_first_token_cuda.json` | Comparison |
| `scripts/profile_device_time.py` (`--engine turing` or `vllm`, each under its own Python; `--batch` for a decode step, `--draft-head` for the EAGLE-3 code workload; `--code` records the commit of the tree that ran) | `results_device_time_cuda.json` (the time the GPU was busy and idle in one run, and the kernel time that ran beside another kernel) | Benchmarks |
| `scripts/verify_model_resolutions.py` | checks every model alias against the Hugging Face Hub | — |
| `scripts/eval_xkv_transfer.py`, `eval_lineage.py`, `eval_kslot.py`, `eval_debate.py` | `results_xkv_transfer_*`, `results_lineage_cuda`, `results_kslot_cuda`, `results_debate_cuda` | Experimental components |

Every script takes `--help`. Scripts write to `benchmarks/results/` by default.
