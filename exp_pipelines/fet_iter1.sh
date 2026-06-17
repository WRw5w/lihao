#!/usr/bin/env bash
# Iterative relabel ROUND 1 (self-distillation):
#  1. extract the best FET+ELR model's predictions over the full train set
#     (a strong label cleaner, vs the weak linear teacher head)
#  2. retrain FET+ELR using those preds as the pseudo-label teacher
#     (--teacher-preds-path) -> recovers/relabels more of the ~37% noise
#  3. full-recipe inference -> A/B candidate
# Confirmation-bias guard: relabel/recover only where model preds AGREE with
# kNN-majority (independent frozen-feature signal), already enforced in prepare_targets.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
TEACHER=outputs_fet_c448_elr8          # best FET+ELR so far (8ep); falls back below
[ -f "$TEACHER/lora/best.pt" ] || TEACHER=outputs_fet_c448_elr
[ -f "$TEACHER/lora/best.pt" ] || { echo "[iter1] no teacher checkpoint"; exit 1; }
PREDS=outputs/cache/fet_preds_round0.pt

echo "[iter1] === 1. extract teacher preds from $TEACHER ==="
$PY -u tools/extract_model_preds.py --work-dir "$TEACHER" --checkpoint best \
  --out "$PREDS" --batch-size 128 --num-workers 2 --no-pin > exp_pipelines/fet_iter1.log 2>&1
grep -aE 'saved|mean_conf' exp_pipelines/fet_iter1.log | tr -d '\r' | tail -1
[ -f "$PREDS" ] || { echo "[iter1] extract failed"; exit 1; }

echo "[iter1] === 2. retrain FET+ELR with iterative-relabel teacher (8ep) ==="
$PY -u finetune_fet.py \
  --work-dir outputs_fet_iter1 --cache-dir outputs/cache --img-size 448 \
  --epochs 8 --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 8 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --elr-lambda 1.0 --elr-beta 0.7 --teacher-preds-path "$PREDS" \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 4 >> exp_pipelines/fet_iter1.log 2>&1
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' exp_pipelines/fet_iter1.log | sort -t= -k2 -rn | head -1)
REC=$(grep -aoE 'would recover [0-9]+' exp_pipelines/fet_iter1.log | head -1)
echo "[iter1] retrain best mid: $BEST ; $REC (vs FET+ELR 0.9214, LoRA 0.9244)"
[ -f outputs_fet_iter1/lora/best.pt ] || { echo "[iter1] retrain failed"; exit 1; }

echo "[iter1] === 3. full-recipe inference ==="
$PY -u tools/tta_predict_fet.py --work-dir outputs_fet_iter1 --checkpoint best \
  --out-prefix submissions/pred_results_fet_iter1 --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> exp_pipelines/fet_iter1.log 2>&1
$PY check_submission.py --csv submissions/pred_results_fet_iter1_tta_balanced.csv \
  --zip submissions/pred_results_fet_iter1_tta_balanced.zip 2>&1 | grep -E 'RESULT|ERROR'

git add exp_pipelines/fet_iter1.sh exp_pipelines/fet_iter1.log tools/extract_model_preds.py \
  submissions/pred_results_fet_iter1_tta_balanced.zip submissions/pred_results_fet_iter1_tta_balanced.csv 2>/dev/null
git commit -q -m "迭代重打标 round1: FET+ELR当老师重洗标签重训, val $BEST ($REC)

强模型预测替线性老师做伪标签决策, 回收更多37%噪声. 防确认偏差: 要求与kNN共识.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[iter1] ITER1 DONE -- val=$BEST"
