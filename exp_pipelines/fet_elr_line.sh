#!/usr/bin/env bash
# FET + ELR A/B: same FET config as fet_line.sh but with Early-Learning
# Regularization (--elr-lambda 1) to fight the late-epoch noise memorization
# seen in the FET baseline (loss down but noisy_all up, mid ep5->ep6 dip).
# Isolated outputs_fet_c448_elr/ -- FET baseline (outputs_fet_c448) untouched.
# A/B target: beat FET baseline val mid 0.9184 (and ideally LoRA 0.9244).
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
LAM=1.0

echo "[fet-elr] === 1. train FET+ELR(lambda=$LAM) 90/10 seed42 6ep ==="
$PY -u finetune_fet.py \
  --work-dir outputs_fet_c448_elr --cache-dir outputs/cache --img-size 448 \
  --epochs 6 --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 8 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --elr-lambda $LAM --elr-beta 0.7 \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 3 > exp_pipelines/fet_elr_train.log 2>&1
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_elr_train.log | sort -t= -k2 -rn | head -1)
echo "[fet-elr] train rc=$?  best mid: $BEST  (FET baseline was 0.9184, LoRA 0.9244)"

if [ ! -f outputs_fet_c448_elr/lora/best.pt ]; then echo "[fet-elr] NO best.pt -- failed"; exit 1; fi

echo "[fet-elr] === 2. full-recipe inference (448/512/576 + balance 1.0) ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_c448_elr --checkpoint best \
  --out-prefix submissions/pred_results_fet_elr --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> exp_pipelines/fet_elr_train.log 2>&1

echo "[fet-elr] === 3. validate ==="
$PY check_submission.py --csv submissions/pred_results_fet_elr_tta_balanced.csv \
  --zip submissions/pred_results_fet_elr_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

echo "[fet-elr] === 4. commit+push ==="
git add exp_pipelines/fet_elr_line.sh exp_pipelines/fet_elr_train.log \
  submissions/pred_results_fet_elr_tta_balanced.zip submissions/pred_results_fet_elr_tta_balanced.csv 2>/dev/null
git commit -q -m "FET+ELR(lambda=$LAM) A/B: val $BEST (vs FET baseline 0.9184 / LoRA 0.9244)

ELR fights late-epoch noise memorization seen in FET baseline. Same FET config,
+Early-Learning Regularization. Full TTA+balance candidate. Isolated outputs_fet_c448_elr.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -2
echo "[fet-elr] FET+ELR DONE -- val=$BEST ; candidate=pred_results_fet_elr_tta_balanced.zip"
