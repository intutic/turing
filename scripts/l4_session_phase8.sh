#!/bin/bash
# Long context at equal KV memory, Turing vs vLLM on one L4: Qwen2.5-1.5B-Instruct, 16K-token prompts, 256 new tokens,
# batches of 16..128 requests. vLLM runs first at gpu_memory_utilization 0.85 and records its KV budget; every Turing
# arm gets the same bytes (--kv-pool-gib-from), so KV capacity differs only by bytes per token.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE7 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase8" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
LC="--model Qwen/Qwen2.5-1.5B-Instruct --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_cuda.json"
VLLM="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm $LC"
TUR="python scripts/benchmark_e2e_generation.py --engine turing $LC --kv-pool-gib-from results_longctx_cuda.json"
step p8_vllm_fp16 $VLLM --kv-cache-dtype native
step p8_vllm_fp8 $VLLM --kv-cache-dtype fp8
step p8_turing_fp16 $TUR --kv-cache-dtype native
step p8_turing_int8 $TUR --kv-cache-dtype int8
step p8_turing_int4 $TUR --kv-cache-dtype int4
step p8_turing_int4_quest $TUR --kv-cache-dtype int4 --kv-page-budget 0.25
echo "PHASE8 DONE $(date +%T)" >> ~/l4_status.txt
