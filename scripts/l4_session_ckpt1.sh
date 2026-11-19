#!/bin/bash
# Speed plan checkpoint 1 (P1 one launch per int4 linear + fused q/k/v and gate/up, P2 fused norm/SiLU/RoPE and decode
# attention, P3 4-bit LM head), run from the speed-plan checkout synced to ~/jflux-speed (L4_DIR=jflux-speed
# scripts/l4_sync.sh) after the Phi-3.5 rerun in ~/jflux. Also finishes P0's llama.cpp cold start (cmake was missing).
cd ~/jflux-speed; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
until grep -q "PHIFIX DONE" ~/l4_status.txt; do sleep 60; done
echo "--- ckpt1" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
GGUF=$HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
# P0.4, finished: llama.cpp built and timed from process start to first token
step ckpt1_cmake bash -c "command -v cmake || python3 -m pip install --user -q cmake"
step ckpt1_llamacpp_build bash -c "cd ~/llama.cpp && rm -rf build && cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF && cmake --build build -j 16 --target llama-batched-bench llama-server"
step ckpt1_cold_llamacpp python3 scripts/benchmark_llamacpp.py coldstart --gguf $GGUF
# correctness first: every CUDA parity test of the new kernels, and the rest of the suite (with this checkout's own
# native extension)
step ckpt1_build_ext python3 setup.py build_ext --inplace
step ckpt1_tests python3 -W error -m pytest -q -p no:cacheprovider
# P1-P3 where a step's time goes now, and P2.4's torch.compile experiment (decision D11)
step ckpt1_profile_qwen python3 scripts/profile_decode_step.py --model $M --settings fp16,rtn4,rtn4h,fp16+compile --out results_decode_profile_ckpt1_cuda.json
step ckpt1_profile_smol360 python3 scripts/profile_decode_step.py --model HuggingFaceTB/SmolLM2-360M --long-context 8192 --settings fp16,rtn4,rtn4h,fp16+compile --out results_decode_profile_ckpt1_smollm2_360m_cuda.json
# the head-to-head again, every engine in this session (phase 10's settings)
SP="--context 512 --new-tokens 256 --batches 1,8 --warmup-context 512 --out results_speed_ckpt1_cuda.json"
step ckpt1_speed_turing_fp16 python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $SP
step ckpt1_speed_turing_gptq python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq
step ckpt1_speed_turing_gptq_head python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $SP --quantization w4a16-gptq --quantize-lm-head
step ckpt1_speed_vllm_fp16 $VPY scripts/benchmark_e2e_generation.py --engine vllm --model $M $SP
step ckpt1_speed_vllm_gptq $VPY scripts/benchmark_e2e_generation.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 $SP
step ckpt1_speed_llamacpp python3 scripts/benchmark_llamacpp.py bench --gguf $GGUF --out results_speed_ckpt1_cuda.json
# quality: P3's decision D12 (delta NLL <= 0.01 on all three models, GSM8K within one standard error) and a check that
# P1/P2 left the 4-bit model's GSM8K where it was (52.6%)
step ckpt1_nll python3 scripts/eval_kv_quality.py --models HuggingFaceTB/SmolLM2-360M,HuggingFaceTB/SmolLM2-1.7B,Qwen/Qwen2.5-1.5B --arms exact_forward,gptq4,gptq4_head --context 2048 --decode 256 --windows 16 --out results_quality_lm_head_cuda.json
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --batch-size 32"
step ckpt1_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --out results_4bit_quality_turing_ckpt1_cuda.json
step ckpt1_gsm_gptq_head python3 -m turing.cli eval-accuracy $GSM --quantize-lm-head --out results_4bit_quality_turing_ckpt1_head_cuda.json
echo "CKPT1 DONE $(date +%T)" >> ~/l4_status.txt
