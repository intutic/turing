#!/bin/bash
# Low-bit KV safety across model families (fresh g2-standard-16 Spot L4). For each instruct model: kv-layer-guard picks
# the layers vLLM's fp8 cache breaks; vLLM runs bf16, fp8 (default) and fp8 with those layers skipped, on GSM8K (500),
# pass-key retrieval and WikiText-2 NLL; Turing runs its int8 and int4 caches (layers chosen by its own calibration)
# on GSM8K and pass-key retrieval. Setup steps are part of the queue so a preemption resumes them too.
cd ~/jflux; export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
python() { python3 "$@"; }
echo "--- validation" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null; }
mkdir -p ~/l4_logs
step v_setup_apt sudo apt-get install -y -q build-essential python3.10-dev
step v_setup_torchaudio sudo pip uninstall -y torchaudio
step v_setup_turing pip install -q --user -e '.[dev]'
step v_setup_guard pip install -q --user -e packages/kv-layer-guard --no-deps
step v_setup_venvpkg sudo apt-get install -y -q python3.10-venv  # the image lacks it; vLLM gets its own venv
step v_setup_vllm bash -c "python3 -m venv ~/vllm-venv && ~/vllm-venv/bin/pip install -q vllm datasets && ~/vllm-venv/bin/pip install -q -e packages/kv-layer-guard --no-deps"
VQ="env PATH=$HOME/vllm-venv/bin:$PATH $HOME/vllm-venv/bin/python scripts/eval_vllm_quality.py --out results_quality_vllm_validation_cuda.json"
run_model() {  # tag, HF id, pass-key context lengths
  local tag=$1 m=$2 ctx=$3
  step v_${tag}_guard bash -c "python3 -m kv_layer_guard $m --kv-cache-dtype fp8 > results_guard_${tag}_cuda.json"
  local skip; skip=$(python3 -c "import json; print(','.join(json.load(open('results_guard_${tag}_cuda.json'))['kv_cache_dtype_skip_layers']))" 2>/dev/null)
  step v_${tag}_vllm_bf16 $VQ --model $m --context-len $ctx --kv-cache-dtype auto
  step v_${tag}_vllm_fp8 $VQ --model $m --context-len $ctx --kv-cache-dtype fp8
  if [ -n "$skip" ]; then step v_${tag}_vllm_fp8_guard $VQ --model $m --context-len $ctx --kv-cache-dtype fp8 --kv-skip-layers "$skip"; fi
  for kv in int8 int4; do
    step v_${tag}_turing_gsm_$kv turing eval-accuracy --model $m --samples 500 --batch-size 32 --device cuda --dtype bf16 --kv-cache-dtype $kv --out results_gsm8k_validation_cuda.json
    step v_${tag}_turing_niah_$kv turing eval-niah --model $m --context-len $ctx --depths 0 0.25 0.5 0.75 1 --device cuda --dtype bf16 --kv-cache-dtype $kv --out results_niah_validation_cuda.json
  done
}
run_model qwen15 Qwen/Qwen2.5-1.5B-Instruct "2048 8192 16384 32768"
run_model llama3b unsloth/Llama-3.2-3B-Instruct "2048 8192 16384 32768"
run_model gemma2 unsloth/gemma-2-2b-it "2048 4096 8192"
run_model phi35 microsoft/Phi-3.5-mini-instruct "2048 8192 16384 32768"
run_model mistral7b unsloth/mistral-7b-instruct-v0.3 "2048 8192 16384 32768"
run_model qwen7b Qwen/Qwen2.5-7B-Instruct "2048 8192 16384 32768"
echo "VALIDATION DONE $(date +%T)" >> ~/l4_status.txt
