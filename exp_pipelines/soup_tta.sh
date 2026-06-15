#!/usr/bin/env bash
# Generate tta_balanced submissions for the uniform & greedy soups from soup_lab.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
for tag in uniform greedy; do
  echo "[soup-tta] === $tag ==="
  $PY -u tools/tta_predict.py --work-dir outputs_soup_$tag \
    --out-prefix submissions/pred_results_soup_$tag --scales 448,512,576 --num-workers 2 --no-pin
  $PY check_submission.py --csv submissions/pred_results_soup_${tag}_tta_balanced.csv \
    --zip submissions/pred_results_soup_${tag}_tta_balanced.zip 2>&1 | grep RESULT
done
echo "SOUP TTA DONE"
