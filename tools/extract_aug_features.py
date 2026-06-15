"""Extract augmentation-consistency for the training set.

For each train image we extract frozen-backbone features under K=2 random
augmented views (RandomResizedCrop + RandAugment, same 224 setup as the standard
cache) and measure how stable the feature is, as a noise signal:

  consistency_i = mean pairwise cosine over { f_std_i, f_aug1_i, f_aug2_i }.

Clean / canonical samples are stable under augmentation (high consistency);
noisy / ambiguous ones drift (low). Saved to outputs/cache/aug_consistency.pt
for use as the 5th factor in the composite reliability weight.

Run: python tools/extract_aug_features.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import LabeledImageDataset, build_finetune_transforms, load_train_samples
from robustft.features import extract_features
from robustft.models import build_frozen_backbone

CACHE = Path("outputs/cache")


def main():
    device = torch.device("cuda")
    backbone = build_frozen_backbone(device)
    samples, _ = load_train_samples(config.TRAIN_DIR, None, 42)
    names = [os.path.basename(p) for p, _ in samples]
    # stronger augmentation than training to probe stability; 224 to match the cache
    train_tf, _ = build_finetune_transforms(backbone, crop_min_scale=0.7, img_size=224, randaug=True)

    views = []
    for v in range(2):
        ds = LabeledImageDataset(samples, train_tf)
        feats, _ = extract_features(backbone, ds, device, batch_size=256, num_workers=2,
                                    tta_flip=False, desc=f"aug-view-{v + 1}")
        views.append(feats.float())  # (N, D), L2-normalised

    std = torch.load(CACHE / "train_features.pt", map_location="cpu")["features"].float()
    assert std.size(0) == views[0].size(0), "aug/cache sample count mismatch"
    v1, v2 = views
    consistency = ((std * v1).sum(1) + (std * v2).sum(1) + (v1 * v2).sum(1)) / 3.0
    consistency = consistency.clamp(0, 1)

    torch.save({"consistency": consistency, "image_names": names}, CACHE / "aug_consistency.pt")
    print(f"[aug-consist] saved {len(consistency)} samples to {CACHE/'aug_consistency.pt'}", flush=True)
    print(f"[aug-consist] mean={consistency.mean():.4f} min={consistency.min():.4f} "
          f"p10={consistency.quantile(0.1):.4f} max={consistency.max():.4f}", flush=True)


if __name__ == "__main__":
    main()
