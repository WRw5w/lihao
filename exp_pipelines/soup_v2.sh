#!/usr/bin/env bash
# Soup v2: a MORE DIVERSE model soup than soup_uniform (which was run60-heavy).
# Average the best checkpoint from distinct runs spanning keep0.75/0.85/0.90 +
# the 5-signal run + the best run60 epoch -> diversity is the key lever for soups.
# Then full TTA (448/512/576) + full balance (1.0) -- the winning inference.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
R=exp_pipelines

echo "[soupv2] === 1. diverse soup (keep0.75/0.85/0.90 + 5sig + run60) ==="
$PY tools/swa_soup.py --out outputs_soup_v2/lora/full.pt --checkpoints \
  $R/auto_c448_rank32/lora/best.pt \
  $R/auto_c448_dr_rank32/lora/best.pt \
  $R/auto_c448_dr_rank32_keep90/lora/best.pt \
  $R/five_signal/lora/last.pt \
  $R/run60_c448_dr_rank32_keep90/lora/best_ep30.pt
if [ ! -f outputs_soup_v2/lora/full.pt ]; then echo "[soupv2] SOUP FAILED"; exit 1; fi

echo "[soupv2] === 2. full TTA + full balance predict ==="
$PY -u tools/tta_predict.py --work-dir outputs_soup_v2 \
  --out-prefix submissions/pred_results_soup_v2 --scales 448,512,576 --balance-strength 1.0 \
  --num-workers 2 --no-pin

echo "[soupv2] === 3. validate ==="
$PY check_submission.py --csv submissions/pred_results_soup_v2_tta_balanced.csv \
  --zip submissions/pred_results_soup_v2_tta_balanced.zip 2>&1 | grep RESULT
echo "[soupv2] SOUP V2 DONE"
