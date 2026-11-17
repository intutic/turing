#!/bin/bash
# vLLM's own low-bit KV caches against Turing's int8/int4 on Qwen2.5-1.5B-Instruct (bf16). vLLM fp8 with default scales
# scored GSM8K 2.4% and NIAH 5% (fp16: 57.4%, 100%); vLLM 0.31 also has per-token-head int8/int4/fp8, TurboQuant, and
# kv_cache_dtype_skip_layers (Turing keeps its fragile layers in the model dtype automatically). Quality with Turing's
# checks, then long-context speed at the same KV memory as phase 8.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "PHASE12 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase13" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
Q="$VPY scripts/eval_vllm_quality.py"
step p13_q_fp8_skip0 $Q --kv-cache-dtype fp8 --kv-skip-layers 0
step p13_q_fp8_pth $Q --kv-cache-dtype fp8_per_token_head
step p13_q_int8_pth $Q --kv-cache-dtype int8_per_token_head
step p13_q_int4_pth $Q --kv-cache-dtype int4_per_token_head
step p13_q_int4_pth_skip0 $Q --kv-cache-dtype int4_per_token_head --kv-skip-layers 0
step p13_q_turboquant $Q --kv-cache-dtype turboquant_k8v4
LC="--model Qwen/Qwen2.5-1.5B-Instruct --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_cuda.json"
S="$VPY scripts/benchmark_e2e_generation.py --engine vllm $LC"
step p13_s_fp8_skip0 $S --vllm-kv-cache-dtype fp8 --kv-skip-layers 0
step p13_s_int8_pth $S --vllm-kv-cache-dtype int8_per_token_head
step p13_s_int4_pth $S --vllm-kv-cache-dtype int4_per_token_head
step p13_s_int4_pth_skip0 $S --vllm-kv-cache-dtype int4_per_token_head --kv-skip-layers 0
echo "PHASE13 DONE $(date +%T)" >> ~/l4_status.txt
