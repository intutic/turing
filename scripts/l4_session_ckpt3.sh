#!/bin/bash
# Speed plan checkpoint 3: the fixes checkpoint 2's numbers called for (int4 GEMM launch shape on narrow layers, int4-KV
# prefill through gather + SDPA once a prefix is cached, tuned prefill tiles, a first token no longer held behind a
# decode forward), measured against checkpoint 2's rows (same VM type and versions); the batched-decode profile for P6.2;
# GSM8K for P8.2. Runs from the speed-plan checkout synced to ~/jflux-speed2 (L4_DIR=jflux-speed2 scripts/l4_sync.sh).
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt3" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt3_build_ext python3 setup.py build_ext --inplace
step ckpt3_tests python3 -W error -m pytest -q -p no:cacheprovider
step ckpt3_w4a16 python3 scripts/benchmark_w4a16.py --out results_w4a16_crossover_ckpt3_cuda.json
step ckpt3_attn_turing python3 scripts/benchmark_prefill_attention.py --out results_prefill_attention_ckpt3_cuda.json
P=scripts/profile_engine_ttft.py
step ckpt3_ttft_fp16 python3 $P --model $M --out results_engine_ttft_ckpt3_cuda.json
step ckpt3_ttft_fp16_mb64 python3 $P --model $M --max-batch-size 64 --out results_engine_ttft_ckpt3_mb64_cuda.json
step ckpt3_ttft_rtn4 python3 $P --model $M --quantization w4a16-rtn --out results_engine_ttft_rtn4_ckpt3_cuda.json
for b in 8 64; do
  step ckpt3_decode_b$b python3 $P --model $M --max-batch-size 64 --decode-batch $b --out results_decode_batch_ckpt3_cuda.json
  step ckpt3_decode_b${b}_rtn4 python3 $P --model $M --max-batch-size 64 --decode-batch $b --quantization w4a16-rtn --out results_decode_batch_ckpt3_cuda.json
done
E="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt3_cuda.json"
step ckpt3_e2e_fp16 $E
step ckpt3_e2e_gptq $E --quantization w4a16-gptq
step ckpt3_e2e_gptq_head $E --quantization w4a16-gptq --quantize-lm-head
LC="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 16384 --new-tokens 256 --batches 16,32 --warmup-context 512 --out results_longctx_ckpt3_cuda.json --kv-pool-gib-from results_longctx_ckpt2_cuda.json"
for kv in native int8 int4; do step ckpt3_lc_$kv $LC --kv-cache-dtype $kv; done
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt3_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt3_cuda.json
echo "CKPT3 DONE $(date +%T)" >> ~/l4_status.txt
