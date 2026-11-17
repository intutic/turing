#!/bin/bash
# The CUDA kernel test fixed after phase 11's suite (it compared a full-precision layer's never-written low-bit slots).
cd ~/jflux; export PATH="$HOME/.local/bin:$PATH"
until grep -q "PHASE13 DONE" ~/l4_status.txt; do sleep 60; done
echo "--- phase14" >> ~/l4_status.txt
step() { n=$1; shift; grep -q "end $n exit=0" ~/l4_status.txt && return 0; echo "start $n $(date +%T)" >> ~/l4_status.txt; "$@" > ~/l4_logs/$n.log 2>&1; echo "end $n exit=$? $(date +%T)" >> ~/l4_status.txt; }
step p14_kv_decode_tests python3 -W error -m pytest -q tests/test_triton_kv_decode.py tests/test_kv_cache.py tests/test_kv_transfer.py
echo "PHASE14 DONE $(date +%T)" >> ~/l4_status.txt
