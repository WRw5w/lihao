#!/usr/bin/env bash
# Full-data 5-signal model: composite reliability (kNN x teacher x margin x
# visual-prototype x aug-consistency) on 100% data, 6 epochs (val peak), then
# lean TTA (448+flip) + balance 0.5. The strongest single 5-signal candidate.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
WD=outputs_full_5sig

echo "[5sig-full] === 1. 满数据 5合一重训 (6轮, batch32) ==="
rm -rf $WD; mkdir -p $WD/lora
$PY -u finetune_lora.py --full --epochs 6 --img-size 448 --batch-size 32 \
  --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 --ema-decay 0.999 --randaug \
  --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --proto-power 0.5 --aug-consist-power 0.5 \
  --num-workers 2 --no-pin --work-dir $WD --cache-dir outputs/cache > $WD/train.log 2>&1
rc=$?
if [ $rc -ne 0 ] || [ ! -f $WD/lora/full.pt ]; then echo "[5sig-full] TRAIN FAILED rc=$rc"; exit 1; fi

echo "[5sig-full] === 2. predict (精简TTA 448+翻转 + balance0.5) ==="
$PY -u tools/tta_predict.py --work-dir $WD \
  --out-prefix submissions/pred_results_5sig_full --scales 448 --balance-strength 0.5 \
  --num-workers 2 --no-pin

echo "[5sig-full] === 3. 校验 ==="
$PY check_submission.py --csv submissions/pred_results_5sig_full_tta_balanced.csv \
  --zip submissions/pred_results_5sig_full_tta_balanced.zip 2>&1 | grep RESULT
echo "[5sig-full] 5SIG FULL DONE"
