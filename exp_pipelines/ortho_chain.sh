#!/usr/bin/env bash
# Chain waiter: block until ortho_v1 finishes, then auto-run ortho_v2 (cleanlab).
# Run as a tracked background task so the harness notifies on full completion.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
echo "[chain] waiting for ortho_v1 to finish..."
while ! grep -aq "\[ortho_v1\] ALL DONE" exp_pipelines/ortho_v1.log 2>/dev/null; do sleep 120; done
echo "[chain] ortho_v1 done -> launching ortho_v2 (cleanlab)"
bash exp_pipelines/ortho_v2.sh
echo "[chain] ALL DONE (ortho_v1 + ortho_v2)"