#!/bin/bash
# After the diag queue: re-measure what the fixes changed. Split-K int4 GEMM and input-sparse GEMVs (speed, row
# crossover), TEAL only in one-row CUDA decode (quality token by token, GSM8K at batch 1), int4 KV with automatically
# chosen full-precision layers (quality, NIAH, GSM8K).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "DIAG DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase5" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p5_cuda_tests python -W error -m pytest -q tests/test_w4a16.py tests/test_activation_sparsity.py tests/test_triton_kv_decode.py tests/test_kv_cache.py tests/test_kv_transfer.py
step p5_w4a16_crossover python scripts/benchmark_w4a16.py --out results_w4a16_crossover_cuda.json
step p5_formats_1p7b python scripts/benchmark_weight_formats.py --model HuggingFaceTB/SmolLM2-1.7B --out results_weight_formats_cuda.json
step p5_formats_qwen python scripts/benchmark_weight_formats.py --model Qwen/Qwen2.5-1.5B --out results_weight_formats_qwen_cuda.json
step p5_quality python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,exact_decode,int8,int4,int4_nosmooth,quest25,rtn4,gptq4,teal25,teal40,teal50 --context 2048 --decode 256 --windows 16 --out results_quality_cuda.json
NIAH="--model Qwen/Qwen2.5-1.5B-Instruct --context-len 2048 8192 16384 32768 --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 --out results_niah_v2_cuda.json"
step p5_niah_int4 turing eval-niah $NIAH --kv-cache-dtype int4
GSM="--model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype bf16"
step p5_gsm_int4 turing eval-accuracy $GSM --samples 500 --batch-size 32 --kv-cache-dtype int4 --out results_gsm8k_v2_cuda.json
step p5_gsm_native_b1 turing eval-accuracy $GSM --samples 200 --batch-size 1 --out results_gsm8k_teal_cuda.json
step p5_gsm_teal40_b1 turing eval-accuracy $GSM --samples 200 --batch-size 1 --activation-sparsity 0.4 --out results_gsm8k_teal_cuda.json
echo "PHASE5 DONE $(date +%T)" >> ~/l4_status.txt
