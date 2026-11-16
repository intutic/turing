#!/bin/bash
# After phase 6: the W6 cross-model evals again with the CUDA RoPE strip fixed (same 64 calibration windows as the MPS
# runs), and the end-to-end Turing-vs-vLLM runs (Turing rows lost to a rerun of phase 2's unconditional rm; vLLM
# now records TTFT).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE6 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase7" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p7_cuda_tests python -W error -m pytest -q tests/test_cross_model_kv_transfer.py tests/test_measured_differentiators.py tests/test_lineage_and_pooling.py tests/test_xkv_latent_transfer.py
step p7_xkv python scripts/eval_xkv_transfer.py --device cuda --out results_xkv_transfer_cuda.json
step p7_xkv_l100 python scripts/eval_xkv_transfer.py --device cuda --ridge-lambda 100 --out results_xkv_transfer_lambda100_cuda.json
step p7_xkv_self python scripts/eval_xkv_transfer.py --device cuda --pairs HuggingFaceTB/SmolLM2-360M:HuggingFaceTB/SmolLM2-360M,Qwen/Qwen2.5-0.5B:Qwen/Qwen2.5-0.5B --out results_xkv_transfer_selfcontrol_cuda.json
step p7_lineage python scripts/eval_lineage.py --device cuda --out results_lineage_cuda.json
step p7_lineage_l100 python scripts/eval_lineage.py --device cuda --ridge-lambda 100 --out results_lineage_lambda100_cuda.json
step p7_kslot python scripts/eval_kslot.py --device cuda --out results_kslot_cuda.json
step p7_e2e_turing python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-360M
step p7_e2e_turing_1p7b python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-1.7B
step p7_e2e_vllm env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-360M
step p7_e2e_vllm_1p7b env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-1.7B
echo "PHASE7 DONE $(date +%T)" >> ~/l4_status.txt
