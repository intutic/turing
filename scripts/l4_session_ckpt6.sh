#!/bin/bash
# Speed plan checkpoint 6 (P2, launches per step): q and k RoPE in one launch, each layer's residual add inside the next
# norm (checkpoint 4: a 4-bit batch-1 step was 7.5 ms with 5.5 ms of device time over 375 launches). Parity, the decode
# profile, batch 1/8/64 fp16 and 4-bit, GSM8K, the full suite. From ~/jflux-speed2, synced after checkpoint 5.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt6" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt6_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_fused_ops.py tests/test_hf_parity.py tests/test_engine_generation.py tests/test_triton_kv_decode.py
step ckpt6_profile python3 scripts/profile_decode_step.py --model $M --settings fp16,rtn4h --out results_decode_profile_ckpt6_cuda.json
E="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt6_cuda.json"
step ckpt6_e2e_fp16 $E
step ckpt6_e2e_gptq $E --quantization w4a16-gptq
step ckpt6_e2e_gptq_head $E --quantization w4a16-gptq --quantize-lm-head
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt6_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt6_cuda.json
step ckpt6_tests python3 -W error -m pytest -q -p no:cacheprovider
echo "CKPT6 DONE $(date +%T)" >> ~/l4_status.txt
