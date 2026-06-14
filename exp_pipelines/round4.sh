#!/usr/bin/env bash
# Round 4: target the REAL metric (overall Top-1 accuracy on a balanced test set).
# Part A = inference levers on the existing champion (fast, no retrain).
# Part B = noise-robust training (GCE) full retrain -> submission.
# All produce leaderboard-testable submissions (we can't rank them on the noisy
# val, so the user A/Bs them on the platform).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
WD=outputs_full_c448_dr_rank32_keep90
PFX=submissions/pred_results_c448_dr_rank32_keep90
INF="--num-workers 2 --no-pin"

echo "[r4] === Part A: inference levers on champion ==="
echo "[r4] A1 multi-view TTA (scales 448,512,576 x flip)"
$PY -u tools/tta_predict.py --work-dir $WD --out-prefix $PFX --scales 448,512,576 $INF
echo "[r4] A2 balanced strength 0.5 (milder)"
$PY -u tools/balanced_predict.py --work-dir $WD --output-csv ${PFX}_balanced_s05.csv --strength 0.5 $INF

echo "[r4] === Part B: GCE robust training (champion config + GCE q=0.7) ==="
GWD=outputs_full_c448_gce
mkdir -p $GWD/lora
$PY -u finetune_lora.py --full --epochs 4 --img-size 448 --batch-size 64 \
  --lora-target attn_mlp --ema-decay 0.999 --randaug --keep-ratio 0.90 \
  --pseudo-thresh 0.6 --pseudo-margin 0.05 --lora-rank 32 --lora-alpha 64 \
  --robust-loss gce --gce-q 0.7 $INF --work-dir $GWD --cache-dir outputs/cache \
  > $GWD/train_full.log 2>&1
echo "[r4] B predict GCE model (plain + balanced)"
$PY -u finetune_lora.py --predict --checkpoint full --work-dir $GWD --cache-dir outputs/cache --batch-size 64 $INF --output-csv submissions/pred_results_c448_gce.csv
$PY -u tools/balanced_predict.py --work-dir $GWD --output-csv submissions/pred_results_c448_gce_balanced.csv --strength 1.0 $INF

echo "[r4] === validate all new submissions ==="
for f in ${PFX}_tta ${PFX}_tta_balanced ${PFX}_balanced_s05 submissions/pred_results_c448_gce submissions/pred_results_c448_gce_balanced; do
  echo "--- $(basename $f) ---"
  $PY check_submission.py --csv ${f}.csv --zip ${f}.zip 2>&1 | grep RESULT
done
echo "[r4] ROUND4 QUEUE COMPLETE"
