#!/bin/bash
# Speed plan P8.4, second pass: the final session's Turing rows again after packed prompts share one FlashInfer
# plan per forward (the final session re-planned per request and layer: TTFT 1.25 -> 1.68 s at batch 64). Was: the final head-to-head in one L4 session, after checkpoints 1-8 (Marlin built in this checkout). The
# phases behind the docs tables (8, 10, 12) again, Turing, vLLM 0.31 and llama.cpp side by side, written to *_final_*
# files that replace the canonical ones on the Mac after review; phase 13 (vLLM's own KV formats) is unchanged code and
# is not rerun. FlashInfer (D10 fallback for native prefill) is installed into Turing's Python first. From ~/jflux-speed2.
cd ~/jflux-speed2; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
echo "--- final2" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; timeout 10800 "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
M=Qwen/Qwen2.5-1.5B-Instruct
VPY="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python"
GGUF=$HOME/gguf/qwen2.5-1.5b-instruct-q4_k_m.gguf
step final2_parity python3 -W error -m pytest -q -p no:cacheprovider tests/test_flashinfer_prefill.py tests/test_triton_kv_prefill.py tests/test_engine_generation.py tests/test_hf_parity.py
step final2_ttft python3 scripts/profile_engine_ttft.py --model $M --out results_engine_ttft_final2_cuda.json
SP="--context 512 --new-tokens 256 --batches 1,8,64 --warmup-context 512 --out results_4bit_speed_final2_cuda.json"
T="python3 scripts/benchmark_e2e_generation.py --engine turing --model $M $SP"
step final2_sp_turing_fp16 $T
step final2_sp_turing_gptq $T --quantization w4a16-gptq
step final2_sp_turing_gptq_head $T --quantization w4a16-gptq --quantize-lm-head
step final2_sp_turing_gptq_teal25 $T --quantization w4a16-gptq --activation-sparsity 0.25
step final2_sp_turing_gptq_teal40 $T --quantization w4a16-gptq --activation-sparsity 0.4
GSM="--model $M --samples 500 --device cuda --dtype fp16 --quantization w4a16-gptq --out results_4bit_quality_turing_final2_cuda.json"
step final2_gsm_gptq python3 -m turing.cli eval-accuracy $GSM --batch-size 32
step final2_gsm_gptq_head python3 -m turing.cli eval-accuracy $GSM --batch-size 32 --quantize-lm-head
E="--context 1024 --new-tokens 256 --batches 1,8,32,64 --out results_e2e_generation_final2_cuda.json"
for m in HuggingFaceTB/SmolLM2-360M HuggingFaceTB/SmolLM2-1.7B; do
  step final2_e2e_turing_$(basename $m) python3 scripts/benchmark_e2e_generation.py --engine turing --model $m $E
done
LC="--model $M --context 16384 --new-tokens 256 --batches 16,32,64,128 --warmup-context 512 --out results_longctx_final2_cuda.json"
TUR="python3 scripts/benchmark_e2e_generation.py --engine turing $LC --kv-pool-gib-from results_longctx_final_cuda.json"
step final2_lc_turing_fp16 $TUR --kv-cache-dtype native
step final2_lc_turing_int8 $TUR --kv-cache-dtype int8
step final2_lc_turing_int4 $TUR --kv-cache-dtype int4
step final2_niah python3 -m turing.cli eval-niah --model $M --device cuda --dtype fp16 --context-len 4096 16384 --out results_niah_final2_speed_cuda.json
echo "FINAL2 DONE $(date +%T)" >> ~/l4_status.txt
