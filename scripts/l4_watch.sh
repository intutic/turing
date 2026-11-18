#!/bin/zsh
# Watches the L4 run queues from the Mac. After a Spot preemption it restarts the VM and relaunches only the queues
# whose DONE marker is missing (each queue skips steps that already finished). Exits on a new failed step or when the
# last queue's marker appears. Usage: scripts/l4_watch.sh "PHASE10 DONE" l4_session_phase9.sh:PHASE9 l4_session_phase10.sh:PHASE10
# With L4_FINISH=1, at the last marker it copies the results and logs back, checks every results file arrived, and
# deletes the VM (the VM also powers itself off 120 min after the marker: scripts/l4_poweroff_when_done.sh).
# With L4_SYNC=1 it copies this checkout to the VM (scripts/l4_sync.sh) before relaunching after a preemption.
Z=${L4_ZONE:-us-central1-a}; V=turing-bench-l4  # L4_ZONE: the zone the VM lives in now
last=$1; shift
repo=${0:A:h:h}  # top level: inside a zsh function $0 is the function's name
vm() { timeout 90 gcloud compute ssh ishangupta@$V --zone $Z --tunnel-through-iap --ssh-key-file $HOME/.ssh/google_compute_engine --quiet --command "$1" 2>/dev/null; }
relaunch="cd ~/jflux && echo '--- resumed after preemption (auto)' >> ~/l4_status.txt; run() { (setsid nohup bash scripts/\$1 > ~/l4_logs/\$1.out 2>&1 < /dev/null &); }"
for q in "$@"; do relaunch="$relaunch; grep -q '${q#*:} DONE' ~/l4_status.txt || run ${q%%:*}"; done
relaunch="$relaunch; (setsid nohup bash scripts/l4_poweroff_when_done.sh '$last' 120 > ~/l4_logs/poweroff.out 2>&1 < /dev/null &)"
finish() {  # copy everything back, verify, then delete the VM
  local RS="ssh -i $HOME/.ssh/google_compute_engine -o StrictHostKeyChecking=accept-new -o HostKeyAlias=$V -o ProxyCommand='gcloud compute start-iap-tunnel $V 22 --listen-on-stdin --zone $Z --verbosity=warning'"
  local logs=$HOME/jflux-l4-logs/$(date +%Y-%m-%d)
  mkdir -p $logs
  # -u: a newer local file wins; the VM's results_ingest_ladder.json is its Linux run under the Mac file's name
  rsync -azu -e "$RS" --include='results_*.json' --exclude='results_ingest_ladder.json' --exclude='*' \
    ishangupta@$V:~/jflux/ $repo/ || return 1
  rsync -az -e "$RS" ishangupta@$V:'~/l4_status.txt ~/l4_logs ~/diag_*.json' $logs/ || return 1
  local missing=$(vm "cd ~/jflux && ls results_*.json" | grep -v '^results_ingest_ladder.json$' | while read f; do [ -f "$repo/$f" ] || echo $f; done)
  [ -n "$missing" ] && { echo "not copied: $missing"; return 1; }
  echo "results in $repo, logs in $logs; deleting $V"
  gcloud compute instances delete $V --zone $Z --quiet
}
seen=""  # failed steps already in the log, read once the VM answers (empty would make an old failure look new)
restarts=0
while true; do
  s=$(gcloud compute instances describe $V --zone $Z --format='value(status)' 2>/dev/null)
  if [ "$s" = "TERMINATED" ]; then
    restarts=$((restarts+1)); echo "$(date -u +%T) preempted, restart $restarts"
    [ $restarts -gt 8 ] && { echo "too many preemptions"; exit 1; }
    until gcloud compute instances start $V --zone $Z >/dev/null 2>&1; do sleep 60; done
    # a preemption while waiting for SSH would otherwise leave this loop waiting forever
    until vm "true"; do sleep 20; [ "$(gcloud compute instances describe $V --zone $Z --format='value(status)' 2>/dev/null)" = TERMINATED ] && continue 2; done
    [ "${L4_SYNC:-0}" = 1 ] && $repo/scripts/l4_sync.sh >/dev/null  # L4_SYNC=1: this checkout's tracked files first
    vm "$relaunch"
    continue
  fi
  [ -z "$seen" ] && { seen=$(vm "grep -cE 'exit=[1-9]' ~/l4_status.txt") && echo "failures already seen: $seen"; sleep 20; continue; }
  n=$(vm "grep -cE 'exit=[1-9]' ~/l4_status.txt")
  if [ -n "$n" ] && { [ "$n" -gt "$seen" ] || vm "grep -q '$last' ~/l4_status.txt"; }; then
    vm "grep -E 'exit=|DONE|^---' ~/l4_status.txt | tail -40"
    if [ "${L4_FINISH:-0}" = 1 ] && vm "grep -q '$last' ~/l4_status.txt"; then finish || { echo "finish failed: VM kept"; exit 1; }; fi
    exit 0
  fi
  sleep 180
done
