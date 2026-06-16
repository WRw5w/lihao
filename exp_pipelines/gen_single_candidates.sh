#!/usr/bin/env bash
# Generate FULL-INFERENCE submissions for the single-model breadth candidates
# (orders 9-14 in submissions/SUBMIT_GUIDE.md). Each: best.pt -> full.pt, then
# full TTA (448/512/576) + full balance (1.0) -- the proven winning inference.
# Names match the guide exactly so codex's queue picks them up.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe

# name | source work-dir | out-prefix (guide filename minus _tta_balanced)
gen() {
  name=$1; wd=$2; prefix=$3
  src="$wd/lora/best.pt"
  if [ ! -f "$src" ]; then echo "[single] SKIP $name (no best.pt at $src)"; return; fi
  cp -f "$src" "$wd/lora/full.pt"
  echo "[single] START $name :: $wd -> $prefix"
  $PY -u tools/tta_predict.py --work-dir "$wd" \
    --out-prefix "$prefix" --scales 448,512,576 --balance-strength 1.0 \
    --num-workers 2 --no-pin > "$wd/single_infer.log" 2>&1
  $PY check_submission.py --csv "${prefix}_tta_balanced.csv" \
    --zip "${prefix}_tta_balanced.zip" 2>&1 | grep -E 'RESULT|OK|ERROR' | sed "s/^/[single $name] /"
}

gen gce      exp_pipelines/breadth_gce      submissions/pred_results_c448_gce
gen keep85   exp_pipelines/breadth_keep85   submissions/pred_results_keep85
gen keep95   exp_pipelines/breadth_keep95   submissions/pred_results_keep95
gen aug06    exp_pipelines/breadth_aug06    submissions/pred_results_aug06
gen ema9995  exp_pipelines/breadth_ema9995  submissions/pred_results_ema9995
gen drecall  exp_pipelines/auto_c448_drecall submissions/pred_results_drecall

echo "[single] SINGLE CANDIDATES DONE"
