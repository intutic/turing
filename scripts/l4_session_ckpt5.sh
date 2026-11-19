#!/bin/bash
# Speed plan checkpoint 5 (P6.2): decode attention with tensor-core products in the model dtype (checkpoint 3's batch-64
# profile: the kernel took 8.6 of 23.6 ms per step, fp32 FMA products padded to 16 query rows). Parity first, then the
# batched-decode profile, batch 1/8/64 fp16 on the three models against checkpoint 2's vLLM rows, 16K long context,
# pass-key retrieval, GSM8K, the full suite. From ~/jflux-speed2, synced after checkpoint 4.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- ckpt5" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
step ckpt5_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_triton_kv_decode.py tests/test_paged_decode_gemv.py tests/test_hf_parity.py
P=scripts/profile_engine_ttft.py
for b in 8 64; do step ckpt5_decode_b$b python3 $P --model $M --max-batch-size 64 --decode-batch $b --out results_decode_batch_ckpt5_cuda.json; done
E="--context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt5_cuda.json"
for m in HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B $M; do
  step ckpt5_e2e_$(basename $m) python3 scripts/benchmark_e2e_generation.py --engine turing --model $m $E
done
step ckpt5_e2e_gptq python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $E --quantization w4a16-gptq
LC="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M --context 16384 --new-tokens 256 --batches 16,32 --warmup-context 512 --out results_longctx_ckpt5_cuda.json --kv-pool-gib-from results_longctx_ckpt2_cuda.json"
for kv in native int8 int4; do step ckpt5_lc_$kv $LC --kv-cache-dtype $kv; done
step ckpt5_niah python3 -m turing.cli eval-niah --model $M --device cuda --dtype fp16 --context-len 4096 16384 --out results_niah_ckpt5_cuda.json
GSM="--model $M --samples 500 --device cuda --dtype fp16 --batch-size 32"
step ckpt5_gsm_fp16 python3 -m turing.cli eval-accuracy $GSM --out results_gsm8k_ckpt5_fp16_cuda.json
step ckpt5_tests python3 -W error -m pytest -q -p no:cacheprovider
echo "CKPT5 DONE $(date +%T)" >> ~/l4_status.txt
