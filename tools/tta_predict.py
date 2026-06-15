"""Multi-view TTA inference: average logits over {resize-scales} x {h-flip}.

Single model, single inference pipeline with test-time augmentation (compliant:
not a multi-model ensemble). The backbone position-embeds are fixed at the
trained img_size, so we vary *scale* by resizing the shorter side to different
sizes then center-cropping back to img_size (the object appears at different
scales while the network input stays img_size). Optionally also fits a uniform
class-marginal bias (balanced inference) on the averaged logits.

Usage:
  python tools/tta_predict.py --work-dir outputs_full_c448_dr_rank32_keep90 \
      --out-prefix submissions/pred_results_c448_dr_rank32_keep90 [--scales 448,512,576]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from timm.data import resolve_data_config
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import IndexedImageDataset
from robustft.models import build_lora_model
from robustft.robust_utils import choose_checkpoint
from robustft.submission import save_predictions, zip_submission

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BICUBIC = transforms.InterpolationMode.BICUBIC


def scale_tf(backbone, img_size, scale):
    cfg = resolve_data_config(model=backbone)
    return transforms.Compose([
        transforms.Resize(scale, interpolation=BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])


@torch.inference_mode()
def collect_tta_logits(args, device):
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", "full")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    t = ckpt.get("args", {})
    img_size = t.get("img_size", 224)
    model = build_lora_model(len(class_names), t.get("lora_rank", 16), t.get("lora_alpha", 32.0),
                             0.0, None, device, lora_blocks=t.get("lora_blocks", 12),
                             lora_target=t.get("lora_target", "attn"), img_size=img_size)
    model.load_state_dict(ckpt["model"])
    model.eval()
    scales = [int(s) for s in args.scales.split(",")]
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')}), img_size={img_size}, scales={scales} x flip", flush=True)

    paths = sorted(str(p) for p in Path(args.test_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    names = [Path(p).name for p in paths]
    logits = torch.zeros(len(paths), len(class_names), dtype=torch.float32)
    for scale in scales:
        loader = DataLoader(IndexedImageDataset(paths, scale_tf(model.backbone, img_size, scale)),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=not args.no_pin)
        for images, idx in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(images) + model(torch.flip(images, dims=[3]))
            logits[idx] += out.float().cpu()
        print(f"  scale {scale} done", flush=True)
    return logits, names, class_names


def fit_uniform_bias(logits, iters=200, step=1.0):
    n, c = logits.shape
    target = n / c
    b = torch.zeros(c)
    for _ in range(iters):
        col = torch.softmax(logits + b, dim=1).sum(0)
        b = b - step * torch.log((col / target).clamp_min(1e-8))
    return b


def dist_stats(preds, c):
    counts = np.bincount(preds, minlength=c)
    return f"min={counts.min()} max={counts.max()} std={counts.std():.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--out-prefix", required=True, help="writes <prefix>_tta.csv and <prefix>_tta_balanced.csv")
    p.add_argument("--scales", default="448,512,576")
    p.add_argument("--balance-strength", type=float, default=1.0,
                   help="0=no balancing, 1=full match to uniform prior (adjustable)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    logits, names, class_names = collect_tta_logits(args, device)
    c = len(class_names)

    tta = logits.argmax(1).numpy()
    b = fit_uniform_bias(logits) * args.balance_strength
    tta_bal = (logits + b).argmax(1).numpy()
    print(f"tta          dist: {dist_stats(tta, c)}")
    print(f"tta_balanced dist: {dist_stats(tta_bal, c)}  (changed {int((tta != tta_bal).sum())})")

    for tag, preds in [("tta", tta), ("tta_balanced", tta_bal)]:
        out_csv = Path(f"{args.out_prefix}_{tag}.csv")
        save_predictions(out_csv, names, preds.tolist(), class_names)
        z = zip_submission(out_csv)
        print(f"saved {out_csv} and {z}")


if __name__ == "__main__":
    main()
