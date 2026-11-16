#!/bin/bash
# W6 real-model evals on the L4, after phase 2 finishes.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE2 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase3" >> ~/l4_status.txt
step() { n=$1; shift; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step xkv_selfcontrol python scripts/eval_xkv_transfer.py --device cuda --pairs HuggingFaceTB/SmolLM2-360M:HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-0.5B:Qwen/Qwen2.5-0.5B --out results_xkv_transfer_selfcontrol_cuda.json
step debate python scripts/eval_debate.py --device cuda --model Qwen/Qwen2.5-1.5B-Instruct --problems 100 --batch 16 --out results_debate_cuda.json
step lineage_l100 python scripts/eval_lineage.py --device cuda --ridge-lambda 100 --calib-windows 48 --out results_lineage_lambda100_cuda.json
step xkv_l100 python scripts/eval_xkv_transfer.py --device cuda --ridge-lambda 100 --calib-windows 48 --out results_xkv_transfer_lambda100_cuda.json
step xkv python scripts/eval_xkv_transfer.py --device cuda --calib-windows 48 --out results_xkv_transfer_cuda.json
step lineage python scripts/eval_lineage.py --device cuda --calib-windows 48 --out results_lineage_cuda.json
step kslot python scripts/eval_kslot.py --device cuda --out results_kslot_cuda.json
echo "PHASE3 DONE $(date +%T)" >> ~/l4_status.txt
