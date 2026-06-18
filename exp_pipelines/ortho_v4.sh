#!/usr/bin/env bash
# Orthogonal probe wave 4 (breadth +3, 2026-06-18): three NEW families distinct
# from waves 1-3 (loss/mixup/dora/cleanlab/fuse).
#   attnpool   : attention-pool last-block patch tokens (learned query) vs CLS (#42, feature-usage)
#   mmixup     : Manifold Mixup -- mix features not inputs (#31, feature-space reg)
#   curriculum : reliability curriculum, clean->noisy staging across epochs (#34)
# Each run self-smokes with its exact flags on GPU first (waves 1-3 done -> GPU free)
# and is skipped if the smoke fails, so one broken path can't kill the others.
# Champion recipe + full TTA + balance; auto-drop to next_queue.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v4.log
: > "$MASTER"

run_one() {
  name=$1; shift; extra="$*"
  wd="outputs_ortho_$name"; log="exp_pipelines/ortho_${name}.log"
  echo "[$name] smoke-guard ($extra)..." | tee -a "$MASTER"
  $PY -u finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache \
    --num-workers 2 --no-pin --batch-size 64 $extra > "exp_pipelines/ortho_${name}_smoke.log" 2>&1
  if [ $? -ne 0 ]; then echo "[$name] SMOKE FAILED -- skip"; tail -5 "exp_pipelines/ortho_${name}_smoke.log" | tee -a "$MASTER"; return; fi
  echo "[$name] ===== TRAIN extra: $extra =====" | tee -a "$MASTER"
  $PY -u finetune_lora.py \
    --work-dir "$wd" --cache-dir outputs/cache --img-size 448 --epochs 6 --batch-size 32 \
    --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
    --keep-ratio 0.90 --ema-decay 0.999 --randaug --pseudo-thresh 0.6 --pseudo-margin 0.05 \
    --label-smoothing 0.1 --num-workers 2 --no-pin --snapshot-after 3 \
    $extra > "$log" 2>&1
  best=$(grep -aoE 'mid_03_06=[0-9.]+' "$log" | sort -t= -k2 -rn | head -1)
  echo "[$name] train best $best  (ref keep95=0.9233 -> 76.14)" | tee -a "$MASTER"
  [ -f "$wd/lora/best.pt" ] || { echo "[$name] NO best.pt"; tail -5 "$log" | tee -a "$MASTER"; return; }
  cp -f "$wd/lora/best.pt" "$wd/lora/full.pt"
  $PY -u tools/tta_predict.py --work-dir "$wd" \
    --out-prefix "submissions/pred_results_ortho_$name" --scales 448,512,576 \
    --balance-strength 1.0 --batch-size 64 --num-workers 2 --no-pin >> "$log" 2>&1
  $PY check_submission.py --csv "submissions/pred_results_ortho_${name}_tta_balanced.csv" \
    --zip "submissions/pred_results_ortho_${name}_tta_balanced.zip" 2>&1 | grep -aE 'RESULT|ERROR' | tee -a "$MASTER"
  git add "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
    "submissions/pred_results_ortho_${name}_tta_balanced.csv" "$log" 2>/dev/null
  git commit -q -m "ortho probe $name: val $best [extra: $extra]

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" 2>&1 | tail -1
  cp -f "submissions/pred_results_ortho_${name}_tta_balanced.zip" \
        "submissions/pred_results_ortho_${name}_tta_balanced.csv" submissions/next_queue/ 2>/dev/null
  echo "[$name] DONE $best  (-> next_queue)" | tee -a "$MASTER"
}

run_one attnpool   --attn-pool
run_one mmixup     --manifold-mixup 0.2
run_one curriculum --curriculum --curriculum-start 0.5
echo "[ortho_v4] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1