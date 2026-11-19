#!/bin/bash
# Speed plan P8.4, third pass: the long-context Quest row (int4 KV, 25% of pages) with the current code, so the table's
# Turing rows all come from one version. Started by a waiter on the VM once FINAL2 DONE appears. From ~/jflux-speed2.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- final3" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
LC="--model $M --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_final2_cuda.json"
step final3_lc_turing_int4_quest python3 scripts/benchmark_e2e_generation.py --engine turing $LC --kv-pool-gib-from results_longctx_final_cuda.json --kv-cache-dtype int4 --kv-page-budget 0.25
echo "FINAL3 DONE $(date +%T)" >> ~/l4_status.txt
