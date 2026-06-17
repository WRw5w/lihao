#!/usr/bin/env bash
# FET+ELR longer (8 epochs): ELR(lambda=1) flipped the late-epoch dip into a
# rise (6ep: 0.9166->0.9214 still climbing) -> ELR enables beneficial longer
# training (it suppresses noise memorization). Test if 8ep climbs past LoRA 0.9244.
# Snapshots (snapshot-after 3) become FET-soup ingredients. Isolated dir.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
LAM=1.0

echo "[fet-elr8] === train FET+ELR(lambda=$LAM) 90/10 8ep ==="
$PY -u finetune_fet.py \
  --work-dir outputs_fet_c448_elr8 --cache-dir outputs/cache --img-size 448 \
  --epochs 8 --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 8 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --elr-lambda $LAM --elr-beta 0.7 \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 4 > exp_pipelines/fet_elr8_train.log 2>&1
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_elr8_train.log | sort -t= -k2 -rn | head -1)
echo "[fet-elr8] train rc=$?  best mid: $BEST  (6ep FET+ELR=0.9214, LoRA=0.9244)"
[ -f outputs_fet_c448_elr8/lora/best.pt ] || { echo "[fet-elr8] NO best.pt"; exit 1; }

echo "[fet-elr8] === full-recipe inference ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_c448_elr8 --checkpoint best \
  --out-prefix submissions/pred_results_fet_elr8 --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> exp_pipelines/fet_elr8_train.log 2>&1
$PY check_submission.py --csv submissions/pred_results_fet_elr8_tta_balanced.csv \
  --zip submissions/pred_results_fet_elr8_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

git add exp_pipelines/fet_elr_long.sh exp_pipelines/fet_elr8_train.log \
  submissions/pred_results_fet_elr8_tta_balanced.zip submissions/pred_results_fet_elr8_tta_balanced.csv 2>/dev/null
git commit -q -m "FET+ELR 8ep: val $BEST (6ep was 0.9214, LoRA 0.9244) -- test longer ELR training

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[fet-elr8] FET+ELR-8 DONE -- val=$BEST"
