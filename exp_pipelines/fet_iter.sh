#!/usr/bin/env bash
# Generic iterative-relabel round: extract a teacher model's train-set preds,
# retrain FET+ELR using them as the pseudo-label teacher (data-level denoising),
# full-recipe inference, commit. Compounds: each round's stronger model cleans
# labels better. Confirmation-bias guard = kNN-consensus requirement in prepare_targets.
#
# Usage: bash exp_pipelines/fet_iter.sh <round> <teacher_workdir> [epochs]
#   bash exp_pipelines/fet_iter.sh 2 outputs_fet_iter1 8
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
R=${1:?round}; TEACHER=${2:?teacher workdir}; EP=${3:-8}
[ -f "$TEACHER/lora/best.pt" ] || { echo "[iter$R] no teacher $TEACHER/lora/best.pt"; exit 1; }
PREDS=outputs/cache/fet_preds_r${R}.pt
WD=outputs_fet_iter${R}
OUT=submissions/pred_results_fet_iter${R}
LOG=exp_pipelines/fet_iter${R}.log

echo "[iter$R] === 1. extract teacher preds from $TEACHER ==="
$PY -u tools/extract_model_preds.py --work-dir "$TEACHER" --checkpoint best \
  --out "$PREDS" --batch-size 128 --num-workers 2 --no-pin > "$LOG" 2>&1
grep -aE 'saved|mean_conf' "$LOG" | tr -d '\r' | tail -1
[ -f "$PREDS" ] || { echo "[iter$R] extract failed"; exit 1; }

echo "[iter$R] === 2. retrain FET+ELR with round-$R relabel teacher (${EP}ep) ==="
$PY -u finetune_fet.py \
  --work-dir "$WD" --cache-dir outputs/cache --img-size 448 \
  --epochs "$EP" --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
  --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 \
  --pfi-weight 0.5 --pfi-classes 4 --pfi-images 8 \
  --local-depth 2 --num-parts 8 --part-channels 16 --local-scale 0.5 --gaussian-ksize 15 \
  --elr-lambda 1.0 --elr-beta 0.7 --teacher-preds-path "$PREDS" \
  --num-workers 2 --no-pin --save-every 1 --snapshot-after 4 >> "$LOG" 2>&1
BEST=$(grep -aoE 'mid_03_06=[0-9.]+' "$LOG" | sort -t= -k2 -rn | head -1)
REC=$(grep -aoE 'would recover [0-9]+' "$LOG" | head -1)
echo "[iter$R] retrain best mid: $BEST ; $REC"
[ -f "$WD/lora/best.pt" ] || { echo "[iter$R] retrain failed"; exit 1; }

echo "[iter$R] === 3. full-recipe inference ==="
$PY -u tools/tta_predict_fet.py --work-dir "$WD" --checkpoint best \
  --out-prefix "$OUT" --scales 448,512,576 --balance-strength 1.0 \
  --batch-size 96 --num-workers 2 --no-pin >> "$LOG" 2>&1
$PY check_submission.py --csv "${OUT}_tta_balanced.csv" --zip "${OUT}_tta_balanced.zip" 2>&1 | grep -E 'RESULT|ERROR'

git add exp_pipelines/fet_iter.sh "$LOG" "${OUT}_tta_balanced.zip" "${OUT}_tta_balanced.csv" 2>/dev/null
git commit -q -m "迭代重打标 round$R (teacher=$TEACHER): val $BEST, $REC

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
git push 2>&1 | tail -1
echo "[iter$R] ITER$R DONE -- val=$BEST"
