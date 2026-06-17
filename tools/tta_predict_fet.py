"""Multi-view TTA inference for the FET-enhanced LoRA model (FETLoraClassifier).

Identical inference recipe to tools/tta_predict.py (multi-scale resize+center-crop
x h-flip logit averaging + optional uniform-marginal balance), but builds
build_fet_model so it can load FET checkpoints (outputs_fet_*/lora/full.pt).
At inference FETLoraClassifier.forward(images) returns plain logits (PFI is
train-only), so the TTA loop is unchanged. Single model, single inference.

Usage:
  python tools/tta_predict_fet.py --work-dir outputs_fet_c448_full \
      --out-prefix submissions/pred_results_fet --scales 448,512,576 --balance-strength 1.0
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
from robustft.fet_model import build_fet_model
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
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    t = ckpt.get("args", {})
    img_size = t.get("img_size", 448)
    model = build_fet_model(
        len(class_names), t.get("lora_rank", 32), t.get("lora_alpha", 64.0),
        0.0, None, device,
        lora_blocks=t.get("lora_blocks", 12), lora_target=t.get("lora_target", "attn_mlp"),
        img_size=img_size,
        num_parts=t.get("num_parts", 8), part_channels=t.get("part_channels", 16),
        local_depth=t.get("local_depth", 2), local_scale=t.get("local_scale", 0.5),
        use_pfi=False, gaussian_ksize=t.get("gaussian_ksize", 15),
    )
    # checkpoint's EMA model was built with use_pfi=True -> has pfi_* params;
    # we build use_pfi=False (PFI is train-only). strict=False drops the unused
    # pfi_* keys; assert nothing else is missing so weights aren't silently wrong.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [k for k in missing if not k.startswith("pfi_")]
    bad_unexpected = [k for k in unexpected if not k.startswith("pfi_")]
    assert not bad_missing and not bad_unexpected, \
        f"unexpected state_dict mismatch: missing={bad_missing} unexpected={bad_unexpected}"
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
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="full")
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--out-prefix", required=True, help="writes <prefix>_tta.csv and <prefix>_tta_balanced.csv")
    p.add_argument("--scales", default="448,512,576")
    p.add_argument("--balance-strength", default="1.0",
                   help="comma-separated strengths to emit (e.g. 0.25,0.5,0.75,1.0); 0=none, 1=full uniform")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    logits, names, class_names = collect_tta_logits(args, device)
    c = len(class_names)

    tta = logits.argmax(1).numpy()
    b = fit_uniform_bias(logits)
    strengths = [float(x) for x in str(args.balance_strength).split(",")]
    print(f"tta          dist: {dist_stats(tta, c)}")
    outputs = [("tta", tta)]
    for lam in strengths:
        bal = (logits + lam * b).argmax(1).numpy()
        tag = "tta_balanced" if len(strengths) == 1 else f"tta_balanced_s{lam:g}"
        print(f"{tag:18s} dist: {dist_stats(bal, c)}  (changed {int((tta != bal).sum())})")
        outputs.append((tag, bal))
    for tag, preds in outputs:
        out_csv = Path(f"{args.out_prefix}_{tag}.csv")
        save_predictions(out_csv, names, preds.tolist(), class_names)
        z = zip_submission(out_csv)
        print(f"saved {out_csv} and {z}")


if __name__ == "__main__":
    main()
