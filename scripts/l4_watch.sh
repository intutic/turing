#!/bin/zsh
# Watches the L4 run queues from the Mac. After a Spot preemption it restarts the VM and relaunches only the queues
# whose DONE marker is missing (each queue skips steps that already finished). Exits on a new failed step or when the
# last queue's marker appears. Usage: scripts/l4_watch.sh "PHASE10 DONE" l4_session_phase9.sh:PHASE9 l4_session_phase10.sh:PHASE10
Z=us-central1-a; V=turing-bench-l4
last=$1; shift
vm() { timeout 90 gcloud compute ssh ishangupta@$V --zone $Z --tunnel-through-iap --ssh-key-file $HOME/.ssh/google_compute_engine --quiet --command "$1" 2>/dev/null; }
relaunch="cd ~/jflux && echo '--- resumed after preemption (auto)' >> ~/l4_status.txt; run() { (setsid nohup bash scripts/\$1 > ~/l4_logs/\$1.out 2>&1 < /dev/null &); }"
for q in "$@"; do relaunch="$relaunch; grep -q '${q#*:} DONE' ~/l4_status.txt || run ${q%%:*}"; done
seen=$(vm "grep -cE 'exit=[1-9]' ~/l4_status.txt"); echo "failures already seen: $seen"
restarts=0
while true; do
  s=$(gcloud compute instances describe $V --zone $Z --format='value(status)' 2>/dev/null)
  if [ "$s" = "TERMINATED" ]; then
    restarts=$((restarts+1)); echo "$(date -u +%T) preempted, restart $restarts"
    [ $restarts -gt 8 ] && { echo "too many preemptions"; exit 1; }
    until gcloud compute instances start $V --zone $Z >/dev/null 2>&1; do sleep 60; done
    until vm "true"; do sleep 20; done
    vm "$relaunch"
    continue
  fi
  n=$(vm "grep -cE 'exit=[1-9]' ~/l4_status.txt")
  if [ -n "$n" ] && { [ "$n" -gt "$seen" ] || vm "grep -q '$last' ~/l4_status.txt"; }; then
    vm "grep -E 'exit=|DONE|^---' ~/l4_status.txt | tail -40"; exit 0
  fi
  sleep 180
done
