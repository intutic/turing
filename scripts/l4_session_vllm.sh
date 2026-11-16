#!/bin/bash
# vLLM end-to-end rerun after phase 4: FlashInfer JIT-builds its sampler and needs the venv's ninja on PATH.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "PHASE4 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- vllm" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step e2e_vllm3 env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-360M
step e2e_vllm_1p7b3 env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-1.7B
echo "VLLM DONE $(date +%T)" >> ~/l4_status.txt
