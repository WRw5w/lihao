#!/usr/bin/env bash
# Train ONE arm -> same-trajectory SWA -> multi-scale TTA + balanced submission.
#   usage: bash run_arm.sh <name> <img_size> <pos_resample> <batch> <scales>
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

name="$1"; img="$2"; pos="$3"; bs="$4"; scales="$5"
WD="$OUT_ROOT/$name"
mkdir -p "$WD"
echo "============================================================"
echo "[$name] TRAIN  img=$img  pos=$pos  bs=$bs  epochs=$EPOCHS"
echo "============================================================"

( cd "$CODE" && $PY -u finetune_lora.py \
    --work-dir "$WD" --cache-dir "$CACHE_DIR" \
    --train-dir "$DATA_DIR/train" --test-dir "$DATA_DIR/test" \
    --img-size "$img" --pos-resample "$pos" --batch-size "$bs" \
    --epochs "$EPOCHS" --num-workers "$NUM_WORKERS" --seed "$SEED" \
    $COMMON ) 2>&1 | tee "$WD/train.log"

# ---- same-trajectory SWA: average ep<SWA_START>..ep<EPOCHS> into full.pt (single model, compliant) ----
ckpts=""
for e in $(seq "$SWA_START" "$EPOCHS"); do
  f=$(printf "%s/lora/ep%02d.pt" "$WD" "$e")
  [ -f "$f" ] && ckpts="$ckpts $f"
done
if [ -z "$ckpts" ]; then echo "[$name] !! no ep snapshots found, abort"; exit 1; fi
echo "[$name] SWA over:$ckpts"
( cd "$CODE" && $PY -u tools/swa_soup.py --checkpoints $ckpts --out "$WD/lora/full.pt" )

# ---- TTA (multi-scale + h-flip) + balanced. ONLY submit *_tta_balanced.zip ----
( cd "$CODE" && $PY -u tools/tta_predict.py \
    --work-dir "$WD" --test-dir "$DATA_DIR/test" \
    --out-prefix "$OUT_ROOT/submissions/pred_$name" \
    --scales "$scales" --no-pin )

echo "[$name] DONE -> $OUT_ROOT/submissions/pred_${name}_tta_balanced.zip"
