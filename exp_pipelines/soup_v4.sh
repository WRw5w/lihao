#!/usr/bin/env bash
# soup_v4: val-gated "bigger soup of STRONG models". Expand soup_lab candidate
# pool with the 5 fresh breadth models (+drecall); val (90/10) auto-picks top-10.
# If the resulting uniform soup beats the champion's val (0.9267), emit a full
# TTA(448/512/576)+balance1.0 submission for a leaderboard A/B vs soup_uniform=77.69.
# Output dirs are outputs_soup_v4_* (champion outputs_soup_uniform untouched).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
CHAMP_VAL=0.9267

echo "[soupv4] === 1. val-gated soup over expanded pool ==="
$PY -u tools/soup_lab_v4.py > exp_pipelines/soup_v4.log 2>&1
tail -25 exp_pipelines/soup_v4.log | tr -d '\r'

UNI=$(grep -aE 'Uniform Soup' exp_pipelines/soup_v4.log | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "[soupv4] uniform val=$UNI  champ_val=$CHAMP_VAL"

if [ -z "$UNI" ]; then echo "[soupv4] FAILED: no val parsed"; exit 1; fi

BETTER=$($PY -c "print(1 if float('$UNI') >= float('$CHAMP_VAL') else 0)")
if [ "$BETTER" = "1" ]; then
  echo "[soupv4] === 2. val>=champ -> generate full-inference submission ==="
  $PY -u tools/tta_predict.py --work-dir outputs_soup_v4_uniform \
    --out-prefix submissions/pred_results_soup_v4 --scales 448,512,576 --balance-strength 1.0 \
    --num-workers 2 --no-pin >> exp_pipelines/soup_v4.log 2>&1
  $PY check_submission.py --csv submissions/pred_results_soup_v4_tta_balanced.csv \
    --zip submissions/pred_results_soup_v4_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'
  git add submissions/pred_results_soup_v4_tta_balanced.zip submissions/pred_results_soup_v4_tta_balanced.csv \
    tools/soup_lab_v4.py exp_pipelines/soup_v4.sh exp_pipelines/soup_v4.log 2>/dev/null
  git commit -q -m "soup_v4: val-gated bigger soup (uniform val=$UNI >= champ $CHAMP_VAL)

Expanded soup_lab pool with 5 fresh breadth models (+drecall); val picks top-10.
Uniform soup val $UNI vs champion soup_uniform val $CHAMP_VAL. Full TTA+balance1.0
submission ready for leaderboard A/B vs soup_uniform=77.69.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  git push 2>&1 | tail -2
  echo "[soupv4] SUBMISSION READY: pred_results_soup_v4_tta_balanced.zip (val=$UNI)"
else
  echo "[soupv4] === val<champ: breadth did NOT improve soup, NOT submitting ==="
  git add tools/soup_lab_v4.py exp_pipelines/soup_v4.sh exp_pipelines/soup_v4.log 2>/dev/null
  git commit -q -m "soup_v4 negative: val-gated bigger soup did not beat champion (val=$UNI < $CHAMP_VAL)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  git push 2>&1 | tail -2
fi
echo "[soupv4] SOUP V4 DONE"
