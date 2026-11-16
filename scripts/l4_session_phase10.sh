#!/bin/bash
# 4-bit batch-1/batch-8 decode, Turing vs vLLM vs llama.cpp on one L4, Qwen2.5-1.5B-Instruct: speed (512-token prompts,
# 256 new tokens) and GSM8K (500 problems, 8-shot, greedy, Turing's evaluator for every engine), all fp16.
cd ~/jflux; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
python() { python3 "$@"; }
until grep -q "PHASE9 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase10" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
step p10_llamacpp_build bash -c "cd ~ && (test -d llama.cpp || git clone --depth 1 https://github.com/ggml-org/llama.cpp) && cd llama.cpp && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF && cmake --build build -j 16 --target llama-batched-bench llama-server"
step p10_gguf python3 -c "from huggingface_hub import hf_hub_download as d; print(d('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='$HOME/gguf'))"
GGUF=$HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
M=Qwen/Qwen2.5-1.5B-Instruct
SP="--context 512 --new-tokens 256 --batches 1,8 --warmup-context 512 --out results_4bit_speed_cuda.json"
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
step p10_speed_turing_fp16 python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP
step p10_speed_turing_gptq python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq
step p10_speed_turing_gptq_teal25 python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq --activation-sparsity 0.25
step p10_speed_turing_gptq_teal40 python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq --activation-sparsity 0.4
step p10_speed_turing_teal40 python scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --activation-sparsity 0.4
step p10_speed_vllm_fp16 $VPY scripts/benchmark_e2e_generation.py --engine vllm --model $M $SP
step p10_speed_vllm_gptq $VPY scripts/benchmark_e2e_generation.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 $SP
step p10_speed_llamacpp python scripts/benchmark_llamacpp.py bench --gguf $GGUF --out results_4bit_speed_cuda.json
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --out results_4bit_quality_turing_cuda.json"
step p10_gsm_turing_gptq turing eval-accuracy $GSM --batch-size 32
step p10_gsm_turing_gptq_teal25 turing eval-accuracy $GSM --batch-size 1 --activation-sparsity 0.25
step p10_gsm_turing_gptq_teal40 turing eval-accuracy $GSM --batch-size 1 --activation-sparsity 0.4
step p10_gsm_vllm_gptq $VPY scripts/eval_vllm_quality.py --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 --dtype float16 --checks gsm8k --out results_4bit_quality_vllm_cuda.json
step p10_gsm_llamacpp python scripts/benchmark_llamacpp.py gsm8k --gguf $GGUF
echo "PHASE10 DONE $(date +%T)" >> ~/l4_status.txt
