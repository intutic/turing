#!/bin/bash
# Copies every git-tracked file (and nothing else: no results, no builds, no local .so) to the L4 VM, so the VM runs
# exactly this checkout. Hand-picked file lists missed tests/test_measured_differentiators.py once. Pass -n for a dry run
# that lists what differs.
# shortcut: files deleted from git stay on the VM (rsync --files-from cannot delete); remove those by hand.
set -eu
cd "$(dirname "$0")/.."
RS="ssh -i $HOME/.ssh/google_compute_engine -o StrictHostKeyChecking=accept-new -o HostKeyAlias=turing-bench-l4 -o ProxyCommand='gcloud compute start-iap-tunnel turing-bench-l4 22 --listen-on-stdin --zone ${L4_ZONE:-us-central1-a} --verbosity=warning'"
git ls-files -z | grep -zv '^results_' | rsync -azc --from0 --files-from=- -e "$RS" ${1:+"$1"} --itemize-changes \
  ./ ishangupta@turing-bench-l4:~/jflux/
