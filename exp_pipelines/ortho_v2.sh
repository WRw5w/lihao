#!/usr/bin/env bash
# Orthogonal probe wave 2: Confident Learning (cleanlab #18) clean-set selection.
# Replaces the kNN-agreement keep with a per-class confidence-threshold keep
# derived from the teacher's full predictive distribution. Two发:
#   cleanlab      : CL keep alone
#   cleanlab_knn  : CL keep INTERSECT kNN keep (stricter clean set)
# Same champion recipe + full TTA + balance as ortho_v1. Launch only after the
# GPU is free (ortho_v1 done). First runs a --smoke guard; aborts if it fails.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
MASTER=exp_pipelines/ortho_v2.log
: > "$MASTER"

echo "[v2] smoke-guard cleanlab path..." | tee -a "$MASTER"
$PY -u finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache \
  --num-workers 2 --no-pin --batch-size 64 --denoise cleanlab > exp_pipelines/ortho_v2_smoke.log 2>&1
if [ $? -ne 0 ]; then echo "[v2] SMOKE FAILED -- abort"; tail -6 exp_pipelines/ortho_v2_smoke.log | tee -a "$MASTER"; exit 1; fi
echo "[v2] smoke ok ($(grep -aoE 'cleanlab\[[a-z_]+\] keeps [0-9/]+ \([0-9.%]+\)' exp_pipelines/ortho_v2_smoke.log | head -1))" | tee -a "$MASTER"
# v1 已全部完成(本脚本由链等待器在 v1 done 后启动) -> 把 v1 所有 ortho 候选也补投 next_queue
cp -f submissions/pred_results_ortho_*_tta_balanced.zip submissions/pred_results_ortho_*_tta_balanced.csv submissions/next_queue/ 2>/dev/null
echo "[v2] swept v1 ortho candidates into next_queue" | tee -a "$MASTER"

run_one() {
  name=$1; shift; extra="$*"
  wd="outputs_ortho_$name"; log="exp_pipelines/ortho_${name}.log"
  echo "[$name] ===== TRAIN extra: $extra =====" | tee -a "$MASTER"
  $PY -u finetune_lora.py \
    --work-dir "$wd" --cache-dir outputs/cache --img-size 448 --epochs 6 --batch-size 32 \
    --lora-rank 32 --lora-alpha 64 --lora-target attn_mlp --lora-blocks 12 \
    --keep-ratio 0.90 --ema-decay 0.999 --randaug --pseudo-thresh 0.6 --pseudo-margin 0.05 \
    --label-smoothing 0.1 --num-workers 2 --no-pin --snapshot-after 3 \
    $extra > "$log" 2>&1
  best=$(grep -aoE 'mid_03_06=[0-9.]+' "$log" | sort -t= -k2 -rn | head -1)
  keepln=$(grep -aoE 'cleanlab\[[a-z_]+\] keeps [0-9/]+ \([0-9.%]+\)' "$log" | head -1)
  echo "[$name] train best $best  ($keepln)  (ref keep95=0.9233 -> 76.14)" | tee -a "$MASTER"
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

run_one cleanlab     --denoise cleanlab
run_one cleanlabknn  --denoise cleanlab_knn
echo "[ortho_v2] ALL DONE" | tee -a "$MASTER"
git push 2>&1 | tail -1
# chain wave 3 (feature fusion) once the GPU frees
bash exp_pipelines/ortho_v3.sh