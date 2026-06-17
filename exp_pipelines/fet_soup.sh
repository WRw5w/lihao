#!/usr/bin/env bash
# FET soup: uniform weight-average of the strongest same-arch FET checkpoints
# (iter1/iter2/elr8 best + late snapshots). Soup usually >= best single (LoRA
# soup gave +1.5pt on leaderboard). Compliant (same-arch weight avg, single
# model inference). Then full-recipe inference -> candidate.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe

echo "[fet-soup] === 1. uniform soup of top FET checkpoints ==="
$PY tools/swa_soup.py --out outputs_fet_soup/lora/full.pt --checkpoints \
  outputs_fet_iter1/lora/best.pt outputs_fet_iter1/lora/best_ep07.pt \
  outputs_fet_iter2/lora/best.pt outputs_fet_iter2/lora/best_ep06.pt \
  outputs_fet_c448_elr8/lora/best.pt > exp_pipelines/fet_soup.log 2>&1
[ -f outputs_fet_soup/lora/full.pt ] || { echo "[fet-soup] soup failed"; cat exp_pipelines/fet_soup.log | tail -5; exit 1; }
echo "[fet-soup] souped 5 FET checkpoints -> outputs_fet_soup/lora/full.pt"

echo "[fet-soup] === 2. full-recipe inference (checkpoint=full) ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_soup --checkpoint full \
  --out-prefix submissions/pred_results_fet_soup --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> exp_pipelines/fet_soup.log 2>&1
$PY check_submission.py --csv submissions/pred_results_fet_soup_tta_balanced.csv \
  --zip submissions/pred_results_fet_soup_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

git add exp_pipelines/fet_soup.sh exp_pipelines/fet_soup.log \
  submissions/pred_results_fet_soup_tta_balanced.zip submissions/pred_results_fet_soup_tta_balanced.csv 2>/dev/null
git commit -q -m "FET soup: 同架构平均 iter1/iter2/elr8 强checkpoint -> 满推理候选

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[fet-soup] FET SOUP DONE"
