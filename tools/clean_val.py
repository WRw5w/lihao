"""Pseudo Clean-val experiment.

Problem: the noisy 10% val rewards noise-fitting (misleads selection), while the
trusted-val (agree>=0.7 & teacher-consensus) is too clean -> saturates -> useless.

Idea: build a PSEUDO CLEAN-VAL by CLIP-feature auto-filtering -- keep val samples
whose given label is corroborated by their kNN neighbours (kNN majority == label).
This removes the likely-mislabelled val samples (cleaner accuracy estimate) while
keeping a broad difficulty range (more discriminative than a high-agreement gate).

We then evaluate several 90/10 checkpoints on different val definitions and compare
which one is (a) clean (trustworthy) and (b) discriminative (varies across models).

Run: python tools/clean_val.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.denoise import knn_agreement, knn_majority_prediction
from robustft.engine import stratified_split
from robustft.models import build_lora_model

CKPTS = [
    ("auto_dr_rank32_keep90(champ90/10)", "exp_pipelines/auto_c448_dr_rank32_keep90/lora/best.pt"),
    ("5sig_best(ep3)",                    "exp_pipelines/five_signal/lora/best.pt"),
    ("5sig_last(ep8)",                    "exp_pipelines/five_signal/lora/last.pt"),
    ("run60_ep30",                        "exp_pipelines/run60_c448_dr_rank32_keep90/lora/ep30.pt"),
    ("run60_ep60",                        "exp_pipelines/run60_c448_dr_rank32_keep90/lora/ep60.pt"),
    ("auto_drecall(rank16)",              "exp_pipelines/auto_c448_drecall/lora/best.pt"),
]


@torch.inference_mode()
def per_sample_correct(model, loader, labels_gpu, device):
    correct = torch.zeros_like(labels_gpu, dtype=torch.bool)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        correct[idx.to(device)] = logits.argmax(1) == labels_gpu[idx.to(device)]
    return correct


@torch.inference_mode()
def main():
    device = torch.device("cuda")
    cache = torch.load("outputs/cache/train_features.pt", map_location="cpu")
    labels, class_names, feats = cache["labels"], cache["class_names"], cache["features"]
    num_classes = len(class_names)
    tr_idx, va_idx = stratified_split(labels, 0.1, 42)

    f16 = feats.to(device)
    y = labels.to(device)
    tr, va = tr_idx.to(device), va_idx.to(device)
    val_y = y[va]
    agree = knn_agreement(f16[va], val_y, f16[tr], y[tr], k=16, exclude_self=False).to(device)
    knn_maj = knn_majority_prediction(f16[va], f16[tr], y[tr], 16, num_classes, exclude_self=False).to(device)

    clean = knn_maj == val_y  # CLIP auto-filter: neighbours corroborate the label
    masks = {
        "noisy_all": torch.ones_like(val_y, dtype=torch.bool),
        "clean_knn": clean,
        "clean_hard": clean & (agree >= 0.3) & (agree < 0.6),
        "mid_03_06": (agree >= 0.3) & (agree < 0.6),
        "easy_ge07": agree >= 0.7,
    }
    n = val_y.numel()
    print(f"=== 验证集 {n} 样本，各定义大小 ===", flush=True)
    for k, m in masks.items():
        print(f"  {k:12s}: {int(m.sum()):5d} ({int(m.sum())/n:.1%})", flush=True)
    print(f"  -> Pseudo-clean 剔除了 {int((~clean).sum())} 个'邻居不认'的疑似错标 ({int((~clean).sum())/n:.1%})\n", flush=True)

    paths = [p for p, _ in ImageFolder(str(config.TRAIN_DIR)).samples]
    val_paths = [paths[i] for i in va_idx.tolist()]

    header = f"{'checkpoint':32s} " + " ".join(f"{k:>11s}" for k in masks)
    print(header, flush=True)
    print("-" * len(header), flush=True)
    eval_tf = loader = None
    for name, cp in CKPTS:
        if not Path(cp).exists():
            print(f"{name:32s}  (缺文件)", flush=True); continue
        ck = torch.load(cp, map_location="cpu")
        a = ck.get("args", {})
        model = build_lora_model(num_classes, a.get("lora_rank", 16), a.get("lora_alpha", 32.0), 0.0,
                                 None, device, lora_blocks=a.get("lora_blocks", 12),
                                 lora_target=a.get("lora_target", "attn"), img_size=a.get("img_size", 224))
        model.load_state_dict(ck["model"]); model.eval()
        if eval_tf is None:
            _, eval_tf = build_finetune_transforms(model.backbone, a.get("crop_min_scale", 0.8),
                                                   img_size=a.get("img_size", 224))
            loader = DataLoader(IndexedImageDataset(val_paths, eval_tf), batch_size=128,
                                shuffle=False, num_workers=2, pin_memory=False)
        correct = per_sample_correct(model, loader, val_y, device)
        row = f"{name:32s} " + " ".join(
            f"{(correct & m).sum().item()/max(1,int(m.sum())):>11.4f}" for m in masks.values())
        print(row, flush=True)
        del model; torch.cuda.empty_cache()
    print("\nCLEAN VAL DONE", flush=True)


if __name__ == "__main__":
    main()
