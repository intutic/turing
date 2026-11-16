#!/usr/bin/env bash
# Every CUDA measurement of the v2.0 release, in order, on the temporary Spot L4 VM (see CLAUDE.md for creating it).
# Run under nohup; progress goes to ~/l4_status.txt, logs to ~/l4_logs/. Results land in the repo root as JSON.
#   nohup bash scripts/l4_session.sh > ~/l4_session.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p ~/l4_logs
step() { local name=$1; shift; echo "start $name $(date -u +%H:%M:%S)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$name.log 2>&1;
         echo "end $name exit=$? $(date -u +%H:%M:%S)" >> ~/l4_status.txt; }

PARITY="gpt2,facebook/opt-125m,facebook/opt-350m,EleutherAI/pythia-160m,HuggingFaceTB/SmolLM2-135M,unsloth/gemma-3-270m,Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,unsloth/Llama-3.2-1B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,unsloth/gemma-2-2b,microsoft/phi-2,microsoft/Phi-3-mini-4k-instruct,allenai/OLMoE-1B-7B-0924"
KV_SETTINGS="native_eager,native_sdpa,native,int8,int8_sdpa,int4,int8_quest25,hf_transformers"

step pytest python -W error -m pytest -q -p no:cacheprovider
step parity python scripts/eval_parity.py --device cuda --dtype bf16 --models "$PARITY"
step w4a16_crossover python scripts/benchmark_w4a16.py
step formats_1p7b python scripts/benchmark_weight_formats.py --model HuggingFaceTB/SmolLM2-1.7B --out results_weight_formats_cuda.json
step formats_qwen python scripts/benchmark_weight_formats.py --model Qwen/Qwen2.5-1.5B --out results_weight_formats_qwen_cuda.json
step decode_360m python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --settings "$KV_SETTINGS" --contexts 2048,8192,32768 --batches 1,8,32,64 --out results_kv_decode_cuda_smollm2_360m.json
step decode_1p7b python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-1.7B --settings "$KV_SETTINGS" --contexts 2048,8192,32768 --batches 1,8,32 --out results_kv_decode_cuda_smollm2_1p7b.json
step quest_sweep python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --settings int8,int8_quest25 --contexts 2048,4096,8192,16384,32768 --batches 1,8,64 --quest-min-tokens 0 --out results_quest_threshold_cuda_smollm2_360m.json
step quality python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,int8,int4,quest25,rtn4,gptq4,teal25,teal40,teal50 --context 2048 --decode 256 --windows 16 --out results_quality_cuda.json
step e2e_turing python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-360M
step e2e_turing_1p7b python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-1.7B
step vllm_venv bash -c "python3 -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install -q vllm datasets"
step e2e_vllm ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-360M
step e2e_vllm_1p7b ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-1.7B
step ingest python scripts/benchmark_ingest_ladder_all_tiers.py --dir /tmp --label "l4 $(uname -r)"
step cpp_build bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4 --target turing-cli"
step cpp_engine python scripts/benchmark_cpp_engine.py --binary current=build/bin/turing-cli --out results_cpp_engine_x86.json
echo "ALL DONE $(date -u +%H:%M:%S)" >> ~/l4_status.txt
