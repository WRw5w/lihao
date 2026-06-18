#!/usr/bin/env bash
# Orthogonal-mechanism probe queue ("做两发试试水").
# Each mechanism = champion recipe (448, rank32/a64 attn_mlp 12blk, keep0.90,
# ema0.999, randaug, ls0.1, pseudo 0.6/0.05) + ONE change. 90/10 val for a quick
# read, then best.pt->full.pt + full 3-scale TTA + balance 1.0 -> submittable zip
# (same methodology as the keep95 baseline = val mid 0.9233 -> leaderboard 76.14).
#
# Reference line to beat: single ~76.1, soup champion 77.73. A mechanism is
# "interesting" only if its single clearly clears ~76.5 on the leaderboard;
# then escalate to a soup of its snapshots.
#
# Two发 per 方案:
#   robust-loss:  sce, apl          (two distinct robust losses)
#   mixup:        a0.2, a0.4
#   dora:         rank32, rank16
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v1.log
: > "$MASTER"

run_one() {
  name=$1; batch=$2; shift 2; extra="$*"
  wd="outputs_ortho_$name"
  log="exp_pipelines/ortho_${name}.log"
  echo "[$name] ===== TRAIN (batch $batch) extra: $extra =====" | tee -a "$MASTER"
  $PY -u finetune_lora.py \
    --work-dir "$wd" --cache-dir outputs/cache --img-size 448 --epochs 6 --batch-size "$batch" \
    --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
    --keep-ratio 0.90 --ema-decay 0.999 --randaug --pseudo-thresh 0.6 --pseudo-margin 0.05 \
    --label-smoothing 0.1 --num-workers 2 --no-pin --snapshot-after 3 \
    $extra > "$log" 2>&1
  rc=$?
  best=$(grep -aoE 'mid_03_06=[0-9.]+' "$log" | sort -t= -k2 -rn | head -1)
  echo "[$name] train rc=$rc  best $best  (ref keep95=0.9233 -> 76.14)" | tee -a "$MASTER"
  if [ ! -f "$wd/lora/best.pt" ]; then echo "[$name] NO best.pt -- skip infer"; tail -5 "$log" | tee -a "$MASTER"; return; fi
  cp -f "$wd/lora/best.pt" "$wd/lora/full.pt"
  echo "[$name] ===== full TTA(448/512/576) + balance 1.0 =====" | tee -a "$MASTER"
  $PY -u tools/tta_predict.py --work-dir "$wd" \
    --out-prefix "submissions/pred_results_ortho_$name" --scales 448,512,576 \
    --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$log" 2>&1
  $PY check_submission.py --csv "submissions/pred_results_ortho_${name}_tta_balanced.csv" \
    --zip "submissions/pred_results_ortho_${name}_tta_balanced.zip" 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
  git add "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
    "submissions/pred_results_ortho_${name}_tta_balanced.csv" "$log" 2>/dev/null
  git commit -q -m "ortho probe $name: val $best (ref 0.9233->76.14) [extra: $extra]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
  echo "[$name] DONE  $best" | tee -a "$MASTER"
}

# wave 1: one shot each mechanism (broad coverage first)
run_one sce     32 --robust-loss sce --sce-alpha 0.1 --sce-beta 1.0
run_one apl     32 --robust-loss apl --apl-alpha 1.0 --apl-beta 1.0
run_one mixup02 32 --mixup-alpha 0.2
run_one dora    24 --peft dora
# wave 2: second shot each mechanism
run_one mixup04 32 --mixup-alpha 0.4
run_one dora16  24 --peft dora --lora-rank 16 --lora-alpha 32

echo "[ortho_v1] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1