#!/bin/bash
# Power-of-two key smoothing (bf16-exact) and speed measured on the serving path (CUDA-graph replays).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
echo "--- phase6" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p6_cuda_tests python -W error -m pytest -q tests/test_w4a16.py tests/test_activation_sparsity.py tests/test_triton_kv_decode.py tests/test_kv_cache.py tests/test_kv_transfer.py
step p6_w4a16_crossover python scripts/benchmark_w4a16.py --out results_w4a16_crossover_cuda.json
step p6_formats_1p7b python scripts/benchmark_weight_formats.py --model HuggingFaceTB/SmolLM2-1.7B --out results_weight_formats_cuda.json
step p6_formats_qwen python scripts/benchmark_weight_formats.py --model Qwen/Qwen2.5-1.5B --out results_weight_formats_qwen_cuda.json
step p6_quality python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,exact_decode,int8,int8_nosmooth,int4,int4_nosmooth,quest25,rtn4,gptq4,teal25,teal40,teal50 --context 2048 --decode 256 --windows 16 --out results_quality_cuda.json
step p6_quality_instruct_bf16 python scripts/eval_kv_quality.py --models Qwen/Qwen2.5-1.5B-Instruct --arms exact_forward,int8,int8_nosmooth,int4,int4_nosmooth --dtype bf16 --windows 8 --out results_quality_instruct_bf16_cuda.json
GSM="--model Qwen/Qwen2.5-1.5B-Instruct --samples 500 --batch-size 32 --device cuda --dtype bf16 --out results_gsm8k_v3_cuda.json"
step p6_gsm_native turing eval-accuracy $GSM
step p6_gsm_int8 turing eval-accuracy $GSM --kv-cache-dtype int8
step p6_gsm_int4 turing eval-accuracy $GSM --kv-cache-dtype int4
NIAH="--model Qwen/Qwen2.5-1.5B-Instruct --context-len 2048 8192 16384 32768 --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 --out results_niah_v3_cuda.json"
step p6_niah_native turing eval-niah $NIAH
step p6_niah_int8 turing eval-niah $NIAH --kv-cache-dtype int8
step p6_niah_int4 turing eval-niah $NIAH --kv-cache-dtype int4
echo "PHASE6 DONE $(date +%T)" >> ~/l4_status.txt
