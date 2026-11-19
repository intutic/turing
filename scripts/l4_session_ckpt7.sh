#!/bin/bash
# Speed plan checkpoint 7 (P6.1, decision D13): the vendored Marlin int4 GEMM (csrc/marlin, optional extension). Build it
# for the L4 (sm_89), check parity, measure it against the Triton kernels per (shape, rows), then end to end at batch
# 1/8/64 with GSM8K and the full suite. From ~/jflux-speed2.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt7" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt7_ninja python3 -m pip install --user -q ninja
step ckpt7_build env TURING_BUILD_MARLIN=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 python3 setup.py build_ext --inplace
step ckpt7_import python3 -c "from turing.kernels.marlin import HAS_MARLIN; import turing.turing_csrc; assert HAS_MARLIN; print('ok')"
step ckpt7_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_marlin.py tests/test_w4a16.py
step ckpt7_w4a16 python3 scripts/benchmark_w4a16.py --shapes 1536x2048,1536x1536,1536x17920,8960x1536,1536x151936,4096x14336,14336x4096 --rows 1,2,4,8,16,32,64,128,256 --out results_w4a16_crossover_ckpt7_cuda.json
E="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt7_cuda.json"
step ckpt7_e2e_gptq $E --quantization w4a16-gptq
step ckpt7_e2e_gptq_head $E --quantization w4a16-gptq --quantize-lm-head
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt7_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt7_cuda.json
step ckpt7_tests python3 -W error -m pytest -q -p no:cacheprovider
echo "CKPT7 DONE $(date +%T)" >> ~/l4_status.txt
