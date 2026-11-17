#!/bin/bash
# Fail-safe on the VM: once the last queue's DONE marker appears, power off after a grace period (time for the Mac to
# copy the results and delete the VM). Usage: l4_poweroff_when_done.sh "PHASE13 DONE" [grace minutes, default 120]
until grep -q "$1" ~/l4_status.txt; do sleep 60; done
echo "--- all queues done; powering off in ${2:-120} min unless deleted first" >> ~/l4_status.txt
sudo shutdown -h +"${2:-120}"
