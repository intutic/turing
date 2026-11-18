#!/bin/bash
# Speed plan checkpoint 2 (P4 prefill/TTFT, P5 paged prefill kernel, P6 batched decode, P7.2 cold start), run from the
# speed-plan checkout synced to ~/jflux-speed2 (L4_DIR=jflux-speed2 scripts/l4_sync.sh) after checkpoint 1, so that
# checkpoint's code stays untouched while it runs.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
until grep -q "CKPT1 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- ckpt2" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
step ckpt2_build_ext python3 setup.py build_ext --inplace
step ckpt2_tests python3 -W error -m pytest -q -p no:cacheprovider
# P4.1: where the engine's batch-1 TTFT goes (model-level prefill alongside)
step ckpt2_ttft_fp16 python3 scripts/profile_engine_ttft.py --model $M --out results_engine_ttft_cuda.json
step ckpt2_ttft_rtn4 python3 scripts/profile_engine_ttft.py --model $M --quantization w4a16-rtn --out results_engine_ttft_rtn4_cuda.json
# P5: attention alone (Turing kernel, the old gather + SDPA, FlashInfer from vLLM's environment), then model-level
step ckpt2_attn_turing python3 scripts/benchmark_prefill_attention.py --out results_prefill_attention_cuda.json
step ckpt2_attn_flashinfer $VPY scripts/benchmark_prefill_attention.py --kv native --out results_prefill_attention_flashinfer_cuda.json
step ckpt2_profile_qwen python3 scripts/profile_decode_step.py --model $M --settings fp16,rtn4h --out results_decode_profile_ckpt2_cuda.json
# P6.1: the int4 kernel against fp16 cuBLAS by row count (sets INT4_MAX_ROWS)
step ckpt2_w4a16 python3 scripts/benchmark_w4a16.py --out results_w4a16_crossover_ckpt2_cuda.json
# long context (phase 8's settings): vLLM first, its KV budget given to Turing
LC="--model $M --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_ckpt2_cuda.json"
step ckpt2_lc_vllm_fp16 $VPY scripts/benchmark_e2e_generation.py --engine vllm $LC --kv-cache-dtype native
step ckpt2_lc_vllm_fp8_skip0 $VPY scripts/benchmark_e2e_generation.py --engine vllm $LC --vllm-kv-cache-dtype fp8 --kv-skip-layers 0
TUR="python3 scripts/benchmark_e2e_generation.py --engine turing $LC --kv-pool-gib-from results_longctx_ckpt2_cuda.json"
step ckpt2_lc_turing_fp16 $TUR --kv-cache-dtype native
step ckpt2_lc_turing_int8 $TUR --kv-cache-dtype int8
step ckpt2_lc_turing_int4 $TUR --kv-cache-dtype int4
# P6.2: batch 1/8/64 on the three models, fp16 and 4-bit, both engines
E="--context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_e2e_ckpt2_cuda.json"
for m in HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B $M; do
  t=$(basename $m)
  step ckpt2_e2e_turing_$t python3 scripts/benchmark_e2e_generation.py --engine turing --model $m $E
  step ckpt2_e2e_vllm_$t $VPY scripts/benchmark_e2e_generation.py --engine vllm --model $m $E
done
step ckpt2_e2e_turing_gptq_head python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $E --quantization w4a16-gptq --quantize-lm-head
step ckpt2_e2e_vllm_gptq $VPY scripts/benchmark_e2e_generation.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 $E
# P7.2: cold start with the KV calibration and Triton tuning kept on disk (one untimed run fills them)
step ckpt2_cold_fill python3 scripts/benchmark_cold_start.py --models $M --formats safetensors,tgate4 --caches warm --runs 1 --out /tmp/cold_fill.json
step ckpt2_cold_turing python3 scripts/benchmark_cold_start.py --models $M --formats safetensors,tgate4 --out results_cold_start_vs_turing_ckpt2_cuda.json
echo "CKPT2 DONE $(date +%T)" >> ~/l4_status.txt
