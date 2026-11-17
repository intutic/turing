#!/bin/bash
# llama.cpp for phase 10, rebuilt from clean: a Spot preemption stopped the first build mid-compile, and the rerun
# linked against the half-built objects (undefined CUDA template instances). Runs after phase 10 so the 16-core build
# does not share the machine with its timings.
cd ~/jflux; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE10 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase10b" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p10b_llamacpp_build bash -c "cd ~ && (test -d llama.cpp/.git || { rm -rf llama.cpp; git clone --depth 1 https://github.com/ggml-org/llama.cpp; }) && cd llama.cpp && rm -rf build && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF && cmake --build build -j 16 --target llama-batched-bench llama-server"
GGUF=$HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
step p10b_speed_llamacpp python scripts/benchmark_llamacpp.py bench --gguf $GGUF --out results_4bit_speed_cuda.json
step p10b_gsm_llamacpp python scripts/benchmark_llamacpp.py gsm8k --gguf $GGUF
echo "PHASE10B DONE $(date +%T)" >> ~/l4_status.txt
