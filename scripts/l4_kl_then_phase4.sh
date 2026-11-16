#!/bin/bash
# The KL diagnostic on the CPU first, then phase 4 alone (its timings must not share the machine).
cd ~/jflux
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES= python3 scripts/diag_int4_kv.py --kl --tokens 512 \
  --models Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-0.5B,HuggingFaceTB/SmolLM2-360M --out ~/diag_kl.json > ~/l4_logs/diag_kl.log 2>&1
echo "KL DONE $(date +%T)" >> ~/l4_status.txt
exec bash scripts/l4_session_phase4.sh
