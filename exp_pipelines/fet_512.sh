#!/usr/bin/env bash
# FET @ 512 + iterative-relabel clean labels: combine two orthogonal levers --
# finer spatial resolution (512->16x16=256 tokens, the local branch benefits)
# AND the round-2 cleaned labels (fet_preds_r2.pt). batch 4x6=24 fits 8GB
# (512 smoke: 6.1GB). Inference scales 512/576/640 to match the 512-trained model.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
PREDS=outputs/cache/fet_preds_r2.pt   # strongest relabel teacher so far

echo "[fet512] === train FET+ELR @512 + relabel (batch 4x6=24, 8ep) ==="
$PY -u finetune_fet.py \
  --work-dir outputs_fet_512 --cache-dir outputs/cache --img-size 512 \
  --epochs 8 --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 6 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --elr-lambda 1.0 --elr-beta 0.7 --teacher-preds-path "$PREDS" \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 4 > exp_pipelines/fet_512_train.log 2>&1
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_512_train.log | sort -t= -k2 -rn | head -1)
echo "[fet512] train rc=$?  best mid: $BEST  (448 iter1=0.9289, soup=0.9267)"
[ -f outputs_fet_512/lora/best.pt ] || { echo "[fet512] NO best.pt"; tail -5 exp_pipelines/fet_512_train.log; exit 1; }

echo "[fet512] === full-recipe inference (scales 512/576/640) ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_512 --checkpoint best \
  --out-prefix submissions/pred_results_fet_512 --scales 512,576,640 --balance-strength 1.0 \
  --batch-size 64 --num-workers 2 --no-pin >> exp_pipelines/fet_512_train.log 2>&1
$PY check_submission.py --csv submissions/pred_results_fet_512_tta_balanced.csv \
  --zip submissions/pred_results_fet_512_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

git add exp_pipelines/fet_512.sh exp_pipelines/fet_512_train.log \
  submissions/pred_results_fet_512_tta_balanced.zip submissions/pred_results_fet_512_tta_balanced.csv 2>/dev/null
git commit -q -m "FET@512 + 迭代重打标干净标签: val $BEST (448 iter1=0.9289)

正交杠杆叠加: 512->256token局部分支更细 + round2清洗标签. batch24 fit 8GB.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[fet512] FET512 DONE -- val=$BEST"
