#!/bin/bash
# W9 measurements on the L4 (g2-standard-16), after phase 3 finishes: cold-start TTFT and .tgate4 load time, and MoE
# expert offload on real OLMoE routing.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE3 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase4" >> ~/l4_status.txt
step() { n=$1; shift; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step cold_start python scripts/benchmark_cold_start.py --models HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --runs 5 --out results_cold_start_cuda.json
step moe_offload python scripts/benchmark_moe_offload_olmoe.py --out results_moe_offload_cuda.json
echo "PHASE4 DONE $(date +%T)" >> ~/l4_status.txt
