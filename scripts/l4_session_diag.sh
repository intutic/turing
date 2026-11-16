#!/bin/bash
# After the vLLM rerun: the debate eval (needed jinja2 >= 3.1) and the int8-KV GSM8K drop on Qwen2.5-1.5B-Instruct
# (57.2% native vs 40.2% int8 in bf16 with key smoothing): smoothing off, fp16 instead of bf16, and the quality eval
# on the instruct model in bf16.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "VLLM DONE" ~/l4_status.txt; do sleep 60; done
echo "--- diag" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step debate2 python scripts/eval_debate.py --device cuda --model Qwen/Qwen2.5-1.5B-Instruct --problems 100 --batch 16 --out results_debate_cuda.json
GSM="--model Qwen/Qwen2.5-1.5B-Instruct --samples 500 --batch-size 32 --device cuda --out results_gsm8k_diag_cuda.json"
step gsm_int8_nosmooth turing eval-accuracy $GSM --dtype bf16 --kv-cache-dtype int8 --no-kv-smoothing
step gsm_int8_fp16 turing eval-accuracy $GSM --dtype fp16 --kv-cache-dtype int8
step gsm_native_fp16 turing eval-accuracy $GSM --dtype fp16
step quality_instruct_bf16 python scripts/eval_kv_quality.py --models Qwen/Qwen2.5-1.5B-Instruct --arms exact_forward,int8,int8_nosmooth,int4,int4_nosmooth --dtype bf16 --windows 8 --out results_quality_instruct_bf16_cuda.json
echo "DIAG DONE $(date +%T)" >> ~/l4_status.txt
