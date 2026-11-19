#!/bin/bash
# Speed plan checkpoint 8: Marlin for every row count (checkpoint 7: it beat the Triton kernels from 1 to 256 rows) and
# its copies built before the KV pool is sized. Parity, decode profile, batch 1/8/64, GSM8K, the full suite, then a 7B
# model in 4 bits (memory headroom of the Marlin copies). From ~/jflux-speed2 (the extension is already built there).
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt8" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt8_build env TURING_BUILD_MARLIN=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 python3 setup.py build_ext --inplace
step ckpt8_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_marlin.py tests/test_w4a16.py tests/test_engine_generation.py tests/test_hf_parity.py
step ckpt8_profile python3 scripts/profile_decode_step.py --model $M --settings rtn4,rtn4h --out results_decode_profile_ckpt8_cuda.json
E="python3 scripts/benchmark_e2e_generation.py --engine turing --context 512 --new-tokens 256 --warmup-context 512"
step ckpt8_e2e_gptq $E --model $M --batches 1,8,64 --quantization w4a16-gptq --out results_e2e_ckpt8_cuda.json
step ckpt8_e2e_gptq_head $E --model $M --batches 1,8,64 --quantization w4a16-gptq --quantize-lm-head --out results_e2e_ckpt8_cuda.json
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt8_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt8_cuda.json
step ckpt8_tests python3 -W error -m pytest -q -p no:cacheprovider
step ckpt8_e2e_7b_rtn $E --model Qwen/Qwen2.5-7B-Instruct --batches 1,8 --quantization w4a16-rtn --out results_e2e_7b_ckpt8_cuda.json
echo "CKPT8 DONE $(date +%T)" >> ~/l4_status.txt
