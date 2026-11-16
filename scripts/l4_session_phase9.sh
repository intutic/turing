#!/bin/bash
# vLLM's KV-cache quality with Turing's own GSM8K, pass-key and WikiText-2 checks (the numbers Turing's native/int8/int4
# get in phase 6), Qwen2.5-1.5B-Instruct in bf16: fp16 (auto) and fp8 KV.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "PHASE8 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase9" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
V="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python scripts/eval_vllm_quality.py"
step p9_vllm_quality_fp16 $V --kv-cache-dtype auto
step p9_vllm_quality_fp8 $V --kv-cache-dtype fp8
echo "PHASE9 DONE $(date +%T)" >> ~/l4_status.txt
