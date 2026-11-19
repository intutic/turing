#!/bin/bash
# Speed plan P8.4: the final head-to-head in one L4 session, after checkpoints 1-8 (Marlin built in this checkout). The
# phases behind the docs tables (8, 10, 12) again, Turing, vLLM 0.31 and llama.cpp side by side, written to *_final_*
# files that replace the canonical ones on the Mac after review; phase 13 (vLLM's own KV formats) is unchanged code and
# is not rerun. FlashInfer (D10 fallback for native prefill) is installed into Turing's Python first. From ~/jflux-speed2.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- final" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
GGUF=$HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
# FlashInfer 0.7.0.post1 (vLLM's) into the system Python, torch held at its version
step final_flashinfer bash -c "python3 -m pip install --user -q flashinfer-python==0.7.0.post1 torch==\$(python3 -c 'import torch; print(torch.__version__.split(\"+\")[0])') && python3 -c 'import flashinfer, torch; print(flashinfer.__version__, torch.__version__)'"
step final_build env TURING_BUILD_MARLIN=1 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=8 python3 setup.py build_ext --inplace
step final_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_flashinfer_prefill.py tests/test_triton_kv_prefill.py tests/test_marlin.py tests/test_hf_parity.py tests/test_engine_generation.py
step final_ttft python3 scripts/profile_engine_ttft.py --model $M --out results_engine_ttft_final_cuda.json
# phase 10: batch 1/8 speed, three engines, fp16 and 4-bit (the 4-bit LM head is opt-in: its own rows)
SP="--context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_4bit_speed_final_cuda.json"
T="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $SP"
step final_sp_turing_fp16 $T
step final_sp_turing_gptq $T --quantization w4a16-gptq
step final_sp_turing_gptq_head $T --quantization w4a16-gptq --quantize-lm-head
step final_sp_turing_gptq_teal25 $T --quantization w4a16-gptq --activation-sparsity 0.25
step final_sp_turing_gptq_teal40 $T --quantization w4a16-gptq --activation-sparsity 0.4
step final_sp_vllm_fp16 $VPY scripts/benchmark_e2e_generation.py --engine vllm --model $M $SP
step final_sp_vllm_gptq $VPY scripts/benchmark_e2e_generation.py --engine vllm --model Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4 $SP
step final_sp_llamacpp python3 scripts/benchmark_llamacpp.py bench --gguf $GGUF --batches 1,8,64 --out results_4bit_speed_final_cuda.json
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --out results_4bit_quality_turing_final_cuda.json"
step final_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --batch-size 32
step final_gsm_gptq_head python3 -m turing.cli eval-accuracy $GSM --batch-size 32 --quantize-lm-head
step final_gsm_gptq_teal25 python3 -m turing.cli eval-accuracy $GSM --batch-size 1 --activation-sparsity 0.25
step final_gsm_gptq_teal40 python3 -m turing.cli eval-accuracy $GSM --batch-size 1 --activation-sparsity 0.4
# phase 12: SmolLM2 batch sweep, both engines
E="--context 1024 --new-tokens 256 --batches 1,8,32,64 --out results_e2e_generation_final_cuda.json"  # phase 12's settings
for m in HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B; do
  step final_e2e_turing_$(basename $m) python3 scripts/benchmark_e2e_generation.py --engine turing --model $m $E
  step final_e2e_vllm_$(basename $m) $VPY scripts/benchmark_e2e_generation.py --engine vllm --model $m $E
done
# phase 8: 16K long context, vLLM first (its KV budget is given to Turing)
LC="--model $M --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_final_cuda.json"
step final_lc_vllm_fp16 $VPY scripts/benchmark_e2e_generation.py --engine vllm $LC --kv-cache-dtype native
step final_lc_vllm_fp8 $VPY scripts/benchmark_e2e_generation.py --engine vllm $LC --kv-cache-dtype fp8
TUR="python3 scripts/benchmark_e2e_generation.py --engine turing $LC --kv-pool-gib-from results_longctx_final_cuda.json"
step final_lc_turing_fp16 $TUR --kv-cache-dtype native
step final_lc_turing_int8 $TUR --kv-cache-dtype int8
step final_lc_turing_int4 $TUR --kv-cache-dtype int4
step final_niah python3 -m turing.cli eval-niah --model $M --device cuda --dtype fp16 --context-len 4096 16384 --out results_niah_final_speed_cuda.json
step final_cold_fill python3 scripts/benchmark_cold_start.py --models $M --formats safetensors,tgate4 --caches warm --runs 1 --out /tmp/cold_fill.json
step final_cold_turing python3 scripts/benchmark_cold_start.py --models $M --formats safetensors,tgate4 --out results_cold_start_vs_turing_final_cuda.json
step final_tests python3 -W error -m pytest -q -p no:cacheprovider
echo "FINAL DONE $(date +%T)" >> ~/l4_status.txt
