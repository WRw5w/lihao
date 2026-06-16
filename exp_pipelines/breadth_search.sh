#!/usr/bin/env bash
# Phase 1 breadth search: diverse SAME-ARCH (rank32/attn_mlp/448, seed42) models
# varying loss / keep / augmentation / regularization -> all soup-compatible, so
# they fuse into a stronger, more diverse soup (the proven 76.7 winning approach).
# Each is a 90/10 val run (best.pt by mid_03_06). Then soup the diverse bests +
# strong existing ingredients -> soup_v3 -> full-inference submission.
set -u
cd /d/02_Projects/ML/jinyinsai || exit 1
PY=/d/04_Tools/Python/python.exe
COMMON="--epochs 8 --img-size 448 --batch-size 32 --lora-target attn_mlp --lora-rank 32 --lora-alpha 64 --ema-decay 0.999 --randaug --keep-ratio 0.90 --pseudo-thresh 0.6 --pseudo-margin 0.05 --num-workers 2 --no-pin --cache-dir outputs/cache --early-stop-patience 0"

run() {
  name=$1; shift
  wd=exp_pipelines/breadth_${name}
  rm -rf "$wd"; mkdir -p "$wd/lora"
  echo "[breadth] START ${name} :: $*"
  $PY -u finetune_lora.py $COMMON "$@" --work-dir "$wd" > "$wd/train.log" 2>&1
  echo "[breadth] ${name} done (best mid: $(grep -oE 'mid_03_06=[0-9.]+' $wd/train.log | sort -t= -k2 -rn | head -1))"
}

run gce      --robust-loss gce --gce-q 0.7
run keep95   --keep-ratio 0.95
run keep85   --keep-ratio 0.85
run aug06    --crop-min-scale 0.6
run ema9995  --ema-decay 0.9995

echo "[breadth] === soup the diverse bests + strong existing ingredients -> soup_v3 ==="
R=exp_pipelines
$PY tools/swa_soup.py --out outputs_soup_v3/lora/full.pt --checkpoints \
  $R/breadth_gce/lora/best.pt $R/breadth_keep95/lora/best.pt $R/breadth_keep85/lora/best.pt \
  $R/breadth_aug06/lora/best.pt $R/breadth_ema9995/lora/best.pt \
  $R/auto_c448_dr_rank32_keep90/lora/best.pt $R/run60_c448_dr_rank32_keep90/lora/best_ep30.pt
if [ -f outputs_soup_v3/lora/full.pt ]; then
  $PY -u tools/tta_predict.py --work-dir outputs_soup_v3 \
    --out-prefix submissions/pred_results_soup_v3 --scales 448,512,576 --balance-strength 1.0 \
    --num-workers 2 --no-pin
  $PY check_submission.py --csv submissions/pred_results_soup_v3_tta_balanced.csv \
    --zip submissions/pred_results_soup_v3_tta_balanced.zip 2>&1 | grep RESULT
fi
echo "[breadth] BREADTH PHASE1 DONE"
