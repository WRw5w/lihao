"""Balanced / logit-adjusted inference for a known class-uniform test set.

The competition test set is class-balanced (~N/C per class), but the raw model
predictions are skewed (some classes over-predicted, others starved). We fit a
per-class log-bias b so that the predicted class marginal matches the known
uniform target, then argmax(logits + b). This is inference-time prior
correction (single model, single forward pass + flip TTA) -- it does NOT train
on the test data, only calibrates to the *given* uniform target distribution.

Usage:
  python tools/balanced_predict.py --work-dir outputs_full_c448_dr_rank32_keep90 \
      --output-csv submissions/pred_results_<name>_balanced.csv [--strength 1.0]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.models import build_lora_model
from robustft.robust_utils import choose_checkpoint
from robustft.submission import save_predictions, zip_submission

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@torch.inference_mode()
def collect_logits(args, device):
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", "full")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    t = ckpt.get("args", {})
    model = build_lora_model(len(class_names), t.get("lora_rank", 16), t.get("lora_alpha", 32.0),
                             0.0, None, device, lora_blocks=t.get("lora_blocks", 12),
                             lora_target=t.get("lora_target", "attn"), img_size=t.get("img_size", 224))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')}), img_size={t.get('img_size', 224)}", flush=True)

    _, eval_tf = build_finetune_transforms(model.backbone, t.get("crop_min_scale", 0.8),
                                           img_size=t.get("img_size", 224))
    paths = sorted(str(p) for p in Path(args.test_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXT)
    loader = DataLoader(IndexedImageDataset(paths, eval_tf), batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers, pin_memory=not args.no_pin)
    logits = torch.empty(len(paths), len(class_names), dtype=torch.float32)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(images)
            if not args.no_flip_tta:
                out = out + model(torch.flip(images, dims=[3]))
        logits[idx] = out.float().cpu()
    return logits, [Path(p).name for p in paths], class_names


def fit_uniform_bias(logits: torch.Tensor, iters: int, step: float) -> torch.Tensor:
    """Per-class log-bias b s.t. softmax(logits+b) has a uniform column marginal."""
    n, c = logits.shape
    target = n / c
    b = torch.zeros(c)
    for _ in range(iters):
        col = torch.softmax(logits + b, dim=1).sum(0)
        b = b - step * torch.log((col / target).clamp_min(1e-8))
    return b


def dist_stats(preds: np.ndarray, c: int) -> str:
    counts = np.bincount(preds, minlength=c)
    return (f"min={counts.min()} max={counts.max()} std={counts.std():.1f} "
            f">2x={int((counts > 2 * len(preds) / c).sum())} <0.5x={int((counts < 0.5 * len(preds) / c).sum())}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", required=True)
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--output-csv", required=True)
    p.add_argument("--strength", type=float, default=1.0, help="0=no change, 1=full match to uniform")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--no-flip-tta", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda")
    logits, names, class_names = collect_logits(args, device)
    c = len(class_names)

    plain = logits.argmax(1).numpy()
    b = fit_uniform_bias(logits, args.iters, args.step) * args.strength
    balanced = (logits + b).argmax(1).numpy()
    changed = int((plain != balanced).sum())

    print(f"plain    dist: {dist_stats(plain, c)}")
    print(f"balanced dist: {dist_stats(balanced, c)}  (strength={args.strength})")
    print(f"predictions changed by balancing: {changed}/{len(plain)} ({changed/len(plain):.1%})")

    out_csv = Path(args.output_csv)
    save_predictions(out_csv, names, balanced.tolist(), class_names)
    zip_path = zip_submission(out_csv)
    print(f"saved {out_csv} and {zip_path} ({len(names)} predictions)")


if __name__ == "__main__":
    main()
