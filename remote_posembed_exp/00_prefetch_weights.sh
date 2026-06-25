#!/usr/bin/env bash
# RUN THIS ON THE LOGIN NODE (it has internet). Compute nodes usually do NOT.
# Downloads the CLIP ViT-B/32 weights into the package-local HF cache so the
# SLURM job can load them offline.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

echo "[prefetch] HF_HOME=$HF_HOME"
( cd "$CODE" && $PY - <<'PYEOF'
import timm
m = timm.create_model("vit_base_patch32_clip_224.openai", pretrained=True, num_classes=0)
print("[prefetch] OK — CLIP ViT-B/32 cached, params:", sum(p.numel() for p in m.parameters()))
PYEOF
)
echo "[prefetch] done. The SLURM job can now run with HF_HUB_OFFLINE=1."
