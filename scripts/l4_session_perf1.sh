#!/bin/bash
# Speed plan step P0: where a decode step's time goes (kernel launches, device time, graph vs eager, prefill, the
# GPU's measured read bandwidth), on the two models of the head-to-heads. Runs after the validation queue.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "VALIDATION DONE" ~/l4_status.txt; do sleep 60; done
echo "--- perf1" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step perf1_profile_qwen python3 scripts/profile_decode_step.py --model Qwen/Qwen2.5-1.5B-Instruct --out results_decode_profile_cuda.json
step perf1_profile_smol360 python3 scripts/profile_decode_step.py --model HuggingFaceTB/SmolLM2-360M --long-context 8192 --out results_decode_profile_smollm2_360m_cuda.json
echo "PERF1 DONE $(date +%T)" >> ~/l4_status.txt
