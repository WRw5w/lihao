#!/usr/bin/env bash
# Build the frozen 224 CLIP feature cache (train+test) used for cleanlab denoising
# and the teacher head. It is RESOLUTION-INDEPENDENT, so all arms share one cache.
# A prebuilt cache is bundled in ./cache — this only runs if it is missing.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

if [ -f "$CACHE_DIR/train_features.pt" ] && [ -f "$CACHE_DIR/test_features.pt" ]; then
  echo "[cache] present at $CACHE_DIR (bundled) — skipping build"
  exit 0
fi

echo "[cache] building via main.py (one-time, frozen backbone over train+test)…"
mkdir -p "$CACHE_DIR"
( cd "$CODE" && $PY -u main.py --work-dir "$OUT_ROOT" \
    --train-dir "$DATA_DIR/train" --test-dir "$DATA_DIR/test" \
    --num-workers "$NUM_WORKERS" )
# main.py writes to $OUT_ROOT/cache; mirror into CACHE_DIR if they differ
if [ "$OUT_ROOT/cache" != "$CACHE_DIR" ]; then
  cp -f "$OUT_ROOT/cache/train_features.pt" "$OUT_ROOT/cache/test_features.pt" "$CACHE_DIR/"
fi
echo "[cache] done -> $CACHE_DIR"
