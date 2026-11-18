#!/bin/bash
# Speed plan step P0: where a decode step's time goes (kernel launches, device time, graph vs eager, prefill, the
# GPU's measured read bandwidth), on the two models of the head-to-heads. Runs after the validation queue.
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "VALIDATION DONE" ~/l4_status.txt; do sleep 60; done
echo "--- perf1" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step perf1_profile_qwen python3 scripts/profile_decode_step.py --model Qwen/Qwen2.5-1.5B-Instruct --out results_decode_profile_cuda.json
step perf1_profile_smol360 python3 scripts/profile_decode_step.py --model HuggingFaceTB/SmolLM2-360M --long-context 8192 --out results_decode_profile_smollm2_360m_cuda.json
# P0.4: process start -> first token for the three engines on this VM's disk (cold and warm page cache)
M=Qwen/Qwen2.5-1.5B-Instruct
export PATH="/usr/local/cuda/bin:$PATH"
step perf1_cold_turing python3 scripts/benchmark_cold_start.py --models $M --formats safetensors --out results_cold_start_vs_turing_cuda.json
step perf1_cold_vllm python3 scripts/benchmark_cold_start.py --models $M --formats safetensors --engine vllm --python ~/vllm-venv/bin/python --out results_cold_start_vs_vllm_cuda.json
step perf1_llamacpp_build bash -c "cd ~ && (test -d llama.cpp/.git || { rm -rf llama.cpp; git clone --depth 1 https://github.com/ggml-org/llama.cpp; }) && cd llama.cpp && rm -rf build && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF && cmake --build build -j 16 --target llama-batched-bench llama-server"
step perf1_gguf python3 -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='$HOME/gguf'))"
step perf1_cold_llamacpp python3 scripts/benchmark_llamacpp.py coldstart --gguf $HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
echo "PERF1 DONE $(date +%T)" >> ~/l4_status.txt
