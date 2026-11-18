#!/bin/bash
# Phi-3.5-mini after the LongRoPE fix (chunked prefill now uses the prompt's length; the cache is recomputed when
# decoding crosses 4096): Turing's pass-key retrieval again, int8 and int4, at 2K/8K/16K. And vLLM's GSM8K with
# max_model_len 4096, to see whether its 16K max_model_len is what cost it 8 points against Turing. Runs after perf1.
cd ~/jflux; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
until grep -q "PERF1 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phifix" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=microsoft/Phi-3.5-mini-instruct
for kv in int8 int4; do
  step phifix_niah_$kv turing eval-niah --model $M --context-len 2048 8192 16384 --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 \
    --kv-cache-dtype $kv --out results_niah_phi35_longrope_fix_cuda.json
done
step phifix_vllm_gsm_4k env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python scripts/eval_vllm_quality.py \
  --model $M --checks gsm8k --max-model-len 4096 --kv-cache-dtype auto --out results_quality_vllm_phi35_4k_cuda.json
echo "PHIFIX DONE $(date +%T)" >> ~/l4_status.txt
