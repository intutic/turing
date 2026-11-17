#!/bin/bash
# Final code: the full CUDA suite (W10.2), and the KV-quality numbers with int8 unsmoothed and int4 smoothing only
# 16x+ outlier channels (quality table, GSM8K, pass-key).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE10 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase11" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p11_full_cuda_suite python -W error -m pytest -q
step p11_quality python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,exact_decode,int8,int4,int4_nosmooth,quest25,rtn4,gptq4,teal25,teal40,teal50 --context 2048 --decode 256 --windows 16 --out results_quality_cuda.json
step p11_quality_instruct_bf16 python scripts/eval_kv_quality.py --models Qwen/Qwen2.5-1.5B-Instruct --arms exact_forward,int8,int4,int4_nosmooth --dtype bf16 --windows 8 --out results_quality_instruct_bf16_cuda.json
GSM="--model Qwen/Qwen2.5-1.5B-Instruct --samples 500 --batch-size 32 --device cuda --dtype bf16 --out results_gsm8k_final_cuda.json"
step p11_gsm_native turing eval-accuracy $GSM
step p11_gsm_int8 turing eval-accuracy $GSM --kv-cache-dtype int8
step p11_gsm_int4 turing eval-accuracy $GSM --kv-cache-dtype int4
NIAH="--model Qwen/Qwen2.5-1.5B-Instruct --context-len 2048 8192 16384 32768 --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 --out results_niah_final_cuda.json"
step p11_niah_native turing eval-niah $NIAH
step p11_niah_int8 turing eval-niah $NIAH --kv-cache-dtype int8
step p11_niah_int4 turing eval-niah $NIAH --kv-cache-dtype int4
echo "PHASE11 DONE $(date +%T)" >> ~/l4_status.txt
