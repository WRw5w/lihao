#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
PKG=/root/remote_posembed_exp
DATA_ZIP=/root/cloud/jinyinsai/data.zip
OUT_ROOT=/root/cloud/jinyinsai/remote_posembed_runs
cd "$PKG"

echo "=== remote core run start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "PKG=$PKG"
echo "DATA_ZIP=$DATA_ZIP"
echo "OUT_ROOT=$OUT_ROOT"

echo "=== stop gpu heartbeat before training ==="
pkill -f '/root/server_ops/gpu_load_heartbeat.py' || true
pkill -f '/root/server_ops/gpu_heartbeat.py' || true

mkdir -p "$OUT_ROOT" "$OUT_ROOT/submissions"

echo "=== install/check deps ==="
python - <<'PY'
import importlib.util
mods=['torch','torchvision','timm','numpy','PIL','tqdm','pandas','sklearn','cleanlab']
for m in mods:
    print(m, bool(importlib.util.find_spec(m)))
PY
python -m pip install -q -r requirements.txt pandas scikit-learn cleanlab

echo "=== unzip data if needed ==="
if [ ! -d "$PKG/data/train" ] || [ ! -d "$PKG/data/test" ]; then
  rm -rf "$PKG/data"
  mkdir -p "$PKG"
  unzip -q "$DATA_ZIP" -d "$PKG"
else
  echo "data dirs already exist, skip unzip"
fi

echo "=== data sanity ==="
find "$PKG/data" -maxdepth 2 -type d | sed -n '1,30p'
echo "train class dirs: $(find "$PKG/data/train" -mindepth 1 -maxdepth 1 -type d | wc -l)"
echo "test images: $(find "$PKG/data/test" -maxdepth 1 -type f | wc -l)"
du -sh "$PKG/data" || true

echo "=== smoke ==="
PY=python DATA_DIR="$PKG/data" OUT_ROOT="$OUT_ROOT" NUM_WORKERS=4 bash "$PKG/00_smoke.sh"

echo "=== run core arms ==="
PY=python DATA_DIR="$PKG/data" OUT_ROOT="$OUT_ROOT" NUM_WORKERS=4 bash "$PKG/run_all.sh" core

echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
ls -lh "$OUT_ROOT/submissions" || true
