#!/bin/bash
# Swaps a running queue for its edited script at a step boundary: waits until step $2 has ended, stops the queue
# $1 and whatever step it just started, and relaunches $1 (finished steps are skipped by its step function).
# Usage: l4_swap_queue.sh l4_session_validation.sh v_phi35_16k_turing_niah_int4
until grep -q "end $2 exit=" ~/l4_status.txt; do sleep 15; done
for p in $(pgrep -f "^bash scripts/$1"); do pkill -P "$p"; kill "$p"; done
sleep 3
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
echo "--- relaunched $1 after $2 (edited queue)" >> ~/l4_status.txt
cd ~/jflux && setsid nohup bash "scripts/$1" > ~/l4_logs/$1.out 2>&1 < /dev/null &
