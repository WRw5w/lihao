#!/usr/bin/env bash
# Full-data SWA on the champion config (c448_dr_rank32_keep90):
#   1. full retrain saving per-epoch snapshots (batch 32 for 8GB headroom)
#   2. weight-average the peak-region checkpoints (ep4/5/6) into one model (SWA)
#   3. multi-scale TTA + balanced inference -> leaderboard candidate
# Single model, single inference pipeline (compliant).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
SRC=outputs_full_swa_src
SOUP=outputs_swa_champion

echo "[swa] === 1. full retrain (champion config, batch32, 6ep, per-epoch snapshots) ==="
rm -rf $SRC; mkdir -p $SRC/lora
$PY -u finetune_lora.py --full --epochs 6 --save-every 1 --img-size 448 --batch-size 32 \
  --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 --ema-decay 0.999 --randaug \
  --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --num-workers 2 --no-pin --work-dir $SRC --cache-dir outputs/cache > $SRC/train.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f $SRC/lora/ep06.pt ]; then
  echo "[swa] FAILED: retrain rc=$rc or ep06.pt missing (check $SRC/train.log)"; exit 1
fi

echo "[swa] === 2. SWA weight-average ep4/5/6 ==="
$PY tools/swa_soup.py --checkpoints $SRC/lora/ep04.pt $SRC/lora/ep05.pt $SRC/lora/ep06.pt \
  --out $SOUP/lora/full.pt

echo "[swa] === 3. multi-scale TTA + balanced ==="
$PY -u tools/tta_predict.py --work-dir $SOUP \
  --out-prefix submissions/pred_results_swa_champion --scales 448,512,576 --num-workers 2 --no-pin

echo "[swa] === 4. validate ==="
$PY check_submission.py --csv submissions/pred_results_swa_champion_tta_balanced.csv \
  --zip submissions/pred_results_swa_champion_tta_balanced.zip 2>&1 | grep RESULT
echo "[swa] SWA CHAMPION DONE"
