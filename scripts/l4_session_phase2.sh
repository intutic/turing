#!/usr/bin/env bash
# Second pass after l4_session.sh: steps that need code changed during the run (Triton dequantize kernel) and steps
# that failed for a missing package. Same status file and logs.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
python() { python3 "$@"; }
step() { local name=$1; shift; echo "start $name $(date -u +%H:%M:%S)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$name.log 2>&1;
         echo "end $name exit=$? $(date -u +%H:%M:%S)" >> ~/l4_status.txt; }
PARITY="gpt2,facebook/opt-125m,facebook/opt-350m,EleutherAI/pythia-160m,HuggingFaceTB/SmolLM2-135M,unsloth/gemma-3-270m,Qwen/Qwen2.5-0.5B,Qwen/Qwen3-0.6B,unsloth/Llama-3.2-1B,deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B,unsloth/gemma-2-2b,microsoft/phi-2,microsoft/Phi-3-mini-4k-instruct,allenai/OLMoE-1B-7B-0924"
step build python setup.py build_ext --inplace
step w4a16_tests python -W error -m pytest -q -p no:cacheprovider tests/test_w4a16.py tests/test_activation_sparsity.py
step w4a16_crossover2 python scripts/benchmark_w4a16.py
step parity2 python scripts/eval_parity.py --device cuda --dtype bf16 --models "$PARITY"
step formats_1p7b2 python scripts/benchmark_weight_formats.py --model HuggingFaceTB/SmolLM2-1.7B --out results_weight_formats_cuda.json
step formats_qwen2 python scripts/benchmark_weight_formats.py --model Qwen/Qwen2.5-1.5B --out results_weight_formats_qwen_cuda.json
step quality2 python scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,int8,int4,quest25,rtn4,gptq4,teal25,teal40,teal50 --context 2048 --decode 256 --windows 16 --out results_quality_cuda.json
step decode_360m_32k python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --settings native,int8,int4,int8_quest25,native_eager --contexts 32768 --batches 1,8,32 --out results_kv_decode_cuda_smollm2_360m_32k.json
step decode_sdpa2 python scripts/benchmark_kv_decode.py --model HuggingFaceTB/SmolLM2-360M --settings native,native_sdpa,int8_sdpa --contexts 2048,8192 --batches 1,8,32,64 --out results_kv_decode_sdpa_cuda_smollm2_360m.json
GSM="--model Qwen/Qwen2.5-1.5B-Instruct --samples 500 --batch-size 32 --device cuda --dtype bf16 --out results_gsm8k_cuda.json"
step gsm_native2 turing eval-accuracy $GSM
step gsm_int8_2 turing eval-accuracy $GSM --kv-cache-dtype int8
step gsm_int4_2 turing eval-accuracy $GSM --kv-cache-dtype int4
step gsm_gptq4_2 turing eval-accuracy $GSM --quantization w4a16-gptq
step gsm_teal40_2 turing eval-accuracy $GSM --activation-sparsity 0.4
rm -f results_e2e_generation_cuda.json  # first-pass numbers included graph capture in the timed run
step e2e_turing2 python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-360M
step e2e_turing_1p7b2 python scripts/benchmark_e2e_generation.py --engine turing --model HuggingFaceTB/SmolLM2-1.7B
step vllm_venv2 bash -c "python3 -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install -q vllm datasets"
step e2e_vllm2 env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-360M
step e2e_vllm_1p7b2 env PATH="$HOME/vllm-venv/bin:$PATH" ~/vllm-venv/bin/python scripts/benchmark_e2e_generation.py --engine vllm --model HuggingFaceTB/SmolLM2-1.7B
NIAH="--model Qwen/Qwen2.5-1.5B-Instruct --context-len 2048 8192 16384 32768 --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 --out results_niah_smoothed_cuda.json"
step niah_int4_smoothed turing eval-niah $NIAH --kv-cache-dtype int4
step niah_int8_smoothed turing eval-niah $NIAH --kv-cache-dtype int8
echo "PHASE2 DONE $(date -u +%H:%M:%S)" >> ~/l4_status.txt
