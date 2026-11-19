#!/bin/bash
# Speed plan checkpoint 4: the few-row int4 GEMV kernel (batch-1 4-bit decode was 89.5 tok/s at checkpoint 3; the tiled
# kernel read the wide layers at ~97 GB/s). Kernel parity first, then its crossover on Qwen2.5-1.5B's shapes, the decode
# profile, batch 1/8/64 against checkpoint 2's vLLM rows, GSM8K, and the full suite. From ~/jflux-speed2.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt4" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt4_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_w4a16.py tests/test_engine_generation.py
step ckpt4_w4a16 python3 scripts/benchmark_w4a16.py --rows 1,2,4,8,16,64 --out results_w4a16_crossover_ckpt4_cuda.json
step ckpt4_profile python3 scripts/profile_decode_step.py --model $M --settings rtn4,rtn4h --out results_decode_profile_ckpt4_cuda.json
E="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt4_cuda.json"
step ckpt4_e2e_gptq $E --quantization w4a16-gptq
step ckpt4_e2e_gptq_head $E --quantization w4a16-gptq --quantize-lm-head
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt4_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt4_cuda.json
step ckpt4_tests python3 -W error -m pytest -q -p no:cacheprovider
echo "CKPT4 DONE $(date +%T)" >> ~/l4_status.txt
