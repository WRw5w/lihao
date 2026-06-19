#!/usr/bin/env bash
# Chain waiter: block until ortho_v4 finishes, then run ortho_v5 (cleanlab escalation).
# Run as a tracked background task -> notifies on completion. Does NOT edit the
# running v4 script (unsafe to modify a script mid-execution).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
echo "[chain-v5] waiting for ortho_v4 to finish..."
while ! grep -aq "\[ortho_v4\] ALL DONE" exp_pipelines/ortho_v4.log 2>/dev/null; do sleep 120; done
echo "[chain-v5] ortho_v4 done -> launching ortho_v5 (cleanlab soup escalation)"
bash exp_pipelines/ortho_v5.sh
echo "[chain-v5] ALL DONE (ortho_v5)"