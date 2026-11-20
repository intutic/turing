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
CUDA 12.9; vLLM 0.31 and llama.cpp ran on the same machine. `*_mps.json` files come from an Apple M4 Pro.

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
| `scripts/benchmark_w4a16.py` | `results_w4a16_crossover_*.json` | Benchmarks |
| `scripts/benchmark_cold_start.py` | `results_cold_start_*.json` | Comparison |
| `scripts/benchmark_e2e_generation.py --model allenai/OLMoE-1B-7B-0924-Instruct --dtype bf16`, `turing eval-accuracy` | `results_moe_speed_*.json`, `results_moe_gsm8k_*.json` | Benchmarks |
| `scripts/profile_engine_ttft.py` | first-token latency breakdown | — |
| `scripts/eval_xkv_transfer.py`, `eval_lineage.py`, `eval_kslot.py`, `eval_debate.py` | `results_xkv_transfer_*`, `results_lineage_cuda`, `results_kslot_cuda`, `results_debate_cuda` | Experimental components |

Every script takes `--help`. Scripts write to `benchmarks/results/` by default.
