#!/bin/bash
# vLLM rows again with time to first token: vLLM's v1 engine reports it as RequestStateStats.first_token_latency, which
# the benchmark first read under older field names (the phase 7 and 8 vLLM rows have no TTFT).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "PHASE11 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase12" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm"
step p12_e2e_vllm $VPY --model HuggingFaceTB/SmolLM2-360M
step p12_e2e_vllm_1p7b $VPY --model HuggingFaceTB/SmolLM2-1.7B
LC="--model Qwen/Qwen2.5-1.5B-Instruct --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_cuda.json"
step p12_longctx_vllm_fp16 $VPY $LC --kv-cache-dtype native
step p12_longctx_vllm_fp8 $VPY $LC --kv-cache-dtype fp8
echo "PHASE12 DONE $(date +%T)" >> ~/l4_status.txt
