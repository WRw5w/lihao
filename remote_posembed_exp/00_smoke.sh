#!/usr/bin/env bash
# Fast sanity check BEFORE the long runs: downloads/loads CLIP, builds every arm's
# model and does one forward pass, and prints anchor fidelity for the odd grids.
# Expect: aligned odd-grid arms show anchor_err=0.0000; all out=(2,500) finite=True.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"

( cd "$CODE" && $PY - <<'PYEOF'
import torch, timm
from robustft.models import build_lora_model

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", dev, "| torch", torch.__version__, "| timm", timm.__version__)
g7 = timm.create_model("vit_base_patch32_clip_224.openai", pretrained=True, num_classes=0,
                       img_size=224).pos_embed.detach()[:, 1:, :].reshape(1, 7, 7, 768)

arms = [("b224_native", 224, "timm"), ("b448_default", 448, "timm"),
        ("r416_default", 416, "timm"), ("r416_aligned", 416, "aligned"),
        ("b448_aligned", 448, "aligned"), ("r608_aligned", 608, "aligned")]
ok = True
for name, img, pos in arms:
    m = build_lora_model(500, 32, 64.0, 0.05, None, dev, lora_blocks=12,
                         lora_target="attn_mlp", img_size=img, pos_resample=pos).eval()
    pe = m.backbone.pos_embed.detach().cpu()
    side = int((pe.shape[1] - 1) ** 0.5)
    err = ""
    if side % 2 == 1 and (side - 1) % 6 == 0:          # odd grid that contains the 7x7 anchors
        step = (side - 1) // 6
        g = pe[:, 1:, :].reshape(1, side, side, 768)
        err = f"  anchor_err={(g[:, ::step, ::step, :] - g7).abs().max().item():.4f}"
    x = torch.randn(2, 3, img, img, device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
        o = m(x)
    fin = torch.isfinite(o).all().item()
    ok = ok and fin and tuple(o.shape) == (2, 500)
    print(f"[{name:13s}] grid={side}x{side} tokens={pe.shape[1]-1:4d} out={tuple(o.shape)} finite={fin}{err}")
print("SMOKE OK" if ok else "SMOKE FAILED")
PYEOF
)
