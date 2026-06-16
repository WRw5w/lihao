#!/usr/bin/env bash
# Wait for gen_single_candidates.sh to finish, then commit+push the candidate
# submissions and print each one's check_submission RESULT.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
LOG=exp_pipelines/single_candidates_runner.log
for i in $(seq 1 100); do
  grep -q "SINGLE CANDIDATES DONE" "$LOG" 2>/dev/null && break
  sleep 60
done
echo "=== gen runner final tail ==="
tail -20 "$LOG" 2>/dev/null | tr -d '\r'

# stage only existing candidate artifacts
for n in c448_gce keep85 keep95 aug06 ema9995 drecall; do
  for ext in zip csv; do
    f="submissions/pred_results_${n}_tta_balanced.${ext}"
    [ -f "$f" ] && git add "$f"
  done
done
for d in breadth_gce breadth_keep85 breadth_keep95 breadth_aug06 breadth_ema9995 auto_c448_drecall; do
  [ -f "exp_pipelines/$d/single_infer.log" ] && git add "exp_pipelines/$d/single_infer.log"
done
git add "$LOG" exp_pipelines/commit_single_on_done.sh 2>/dev/null

if git diff --cached --quiet; then
  echo "=== nothing to commit (candidates may have failed) ==="
else
  git commit -q -m "$(cat <<'MSG'
Single-model breadth candidates: full-inference submissions (guide 9-14)

gce/keep85/keep95/aug06/ema9995/drecall best.pt -> full TTA(448/512/576)+balance1.0.
Filenames match SUBMIT_GUIDE.md orders 9-14 for codex queue pickup.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
MSG
)"
  git push 2>&1 | tail -3
  echo "=== committed: $(git log --oneline -1) ==="
fi
echo "[monitor] COMMIT-ON-DONE FINISHED"
