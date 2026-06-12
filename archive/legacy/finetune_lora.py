"""LoRA fine-tuning of CLIP ViT-B/32 on the kNN-denoised training set.

Pipeline (fully scripted, reproducible):
  1. Load cached frozen features -> kNN agreement -> per-class top-75% keep mask
     + high-agreement floor (same recipe as the head-level winner).
  2. Train a frozen-feature teacher head, pseudo-label the dropped samples.
  3. Inject LoRA (rank r) into every attention qkv/proj Linear of the ViT;
     train LoRA + cosine head (warm-started from the teacher) on images with
     augmentation, soft targets and sample weights.
  4. Evaluate per epoch on a stratified noisy val split (banded metrics).
  5. Save checkpoints; --predict generates pred_results_lora.csv from the best
     checkpoint.

Run:  python finetune_lora.py            (train, 90/10 split)
      python finetune_lora.py --full     (train on 100% of the data)
      python finetune_lora.py --predict  (inference from best checkpoint)
      python finetune_lora.py --smoke    (2 quick epochs on a subset)
"""

from __future__ import annotations

import argparse
import csv
import json
import config
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from exp_head import (
    CosineClassifier,
    knn_agreement,
    make_soft_targets_with_pseudo,
    per_class_topk_keep,
    per_sample_stats,
    seed_everything,
    stratified_split,
    train_head,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
MODEL_NAME = "vit_base_patch32_clip_224.openai"


# ------------------------------------------------------------------ LoRA


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        out = out + self.dropout(x) @ self.lora_a.t() @ self.lora_b.t() * self.scaling
        return out


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float) -> list[str]:
    replaced = []
    for blk_i, blk in enumerate(model.blocks):
        attn = blk.attn
        attn.qkv = LoRALinear(attn.qkv, rank, alpha, dropout)
        attn.proj = LoRALinear(attn.proj, rank, alpha, dropout)
        replaced += [f"blocks.{blk_i}.attn.qkv", f"blocks.{blk_i}.attn.proj"]
    return replaced


# ------------------------------------------------------------------ data


class TrainImageDataset(Dataset):
    """Returns (augmented image, soft target vector index, sample weight)."""

    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), idx


class EvalImageDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        return self.transform(img), idx


def build_transforms(model: nn.Module):
    cfg = resolve_data_config(model=model)
    mean, std = cfg["mean"], cfg["std"]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


# ------------------------------------------------------------------ model


class LoraClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: CosineClassifier):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = F.normalize(feat.float(), dim=-1)
        return self.head(feat)


def build_model(num_classes: int, rank: int, alpha: float, lora_dropout: float,
                head_state: dict | None, device: torch.device) -> LoraClassifier:
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank, alpha, lora_dropout)
    head = CosineClassifier(backbone.num_features, num_classes, dropout=0.0)
    if head_state is not None:
        head.load_state_dict(head_state)
    model = LoraClassifier(backbone, head).to(device)
    return model


# ------------------------------------------------------------------ pipeline


def prepare_targets(args, device) -> dict:
    """Frozen-feature stage: kNN keep mask + teacher + soft targets/weights.

    Returns everything index-aligned with ImageFolder ordering.
    """
    work = Path(args.work_dir)
    cache = torch.load(work / "cache" / "train_features.pt", map_location="cpu")
    feats, labels, class_names = cache["features"], cache["labels"], cache["class_names"]
    image_names = cache["image_names"]
    num_classes = len(class_names)
    f16 = feats.to(device)
    f32 = feats.to(device, dtype=torch.float32)
    y = labels.to(device)

    seed_everything(args.seed)
    agree = knn_agreement(f16, y, f16, y, k=args.knn_k, exclude_self=True).to(device)
    keep = per_class_topk_keep(agree, y, num_classes, keep_ratio=args.keep_ratio)
    keep |= agree >= 0.7
    idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
    print(f"kNN filter keeps {idx_keep.numel()}/{y.numel()} ({idx_keep.numel() / y.numel():.2%})")

    teacher = train_head(f32, y, num_classes, idx_keep, smoothing=0.1, epochs=20,
                         batch_size=8192, device=device)
    _, _, preds_t, pmax_t = per_sample_stats(teacher, f32, y)
    targets, weights, _ = make_soft_targets_with_pseudo(
        y, keep, preds_t, pmax_t, num_classes, smoothing=0.1, pseudo_thresh=args.pseudo_thresh)
    print(f"targets ready: {int((weights > 0).sum())} usable samples "
          f"({int(((weights > 0) & ~keep).sum())} pseudo-labelled)")
    return {
        "targets": targets.cpu(), "weights": weights.cpu(), "labels": labels,
        "class_names": class_names, "image_names": image_names,
        "teacher_head_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
        "agree": agree.cpu(),
    }


@torch.inference_mode()
def evaluate_images(model, loader, labels_gpu, agree_gpu, device) -> dict:
    model.eval()
    correct = torch.zeros_like(labels_gpu, dtype=torch.bool)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
        correct[idx.to(device)] = logits.argmax(1) == labels_gpu[idx.to(device)]
    bands = {
        "noisy_all": torch.ones_like(labels_gpu, dtype=torch.bool),
        "low_lt03": agree_gpu < 0.3,
        "mid_03_06": (agree_gpu >= 0.3) & (agree_gpu < 0.6),
        "high_ge06": agree_gpu >= 0.6,
    }
    return {k: round((correct & m).sum().item() / max(1, int(m.sum())), 4) for k, m in bands.items()}


def train(args) -> None:
    device = torch.device("cuda")
    work = Path(args.work_dir)
    ckpt_dir = work / "lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare_targets(args, device)
    class_names = prep["class_names"]
    num_classes = len(class_names)
    targets_all = prep["targets"].to(device)
    weights_all = prep["weights"].to(device)
    labels_all = prep["labels"].to(device)
    agree_all = prep["agree"].to(device)

    base = ImageFolder(args.train_dir)
    paths = [p for p, _ in base.samples]
    names_check = [Path(p).name for p in paths]
    assert names_check == prep["image_names"], "ImageFolder order mismatch with feature cache"

    if args.full:
        tr_idx = torch.arange(len(paths))
        va_idx = torch.arange(0)
    else:
        tr_idx, va_idx = stratified_split(prep["labels"], args.val_frac, args.seed)
    if args.smoke:
        tr_idx = tr_idx[torch.randperm(tr_idx.numel())[:5000]].sort().values
        va_idx = va_idx[torch.randperm(va_idx.numel())[:2000]].sort().values if va_idx.numel() else va_idx
        args.epochs = 2

    usable = (weights_all[tr_idx.to(device)] > 0).cpu()
    tr_idx = tr_idx[usable]
    print(f"train images {tr_idx.numel()}  val images {va_idx.numel()}")

    model = build_model(num_classes, args.lora_rank, args.lora_alpha, args.lora_dropout,
                        prep["teacher_head_state"], device)
    train_tf, eval_tf = build_transforms(model.backbone)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.2f}M")

    train_ds = TrainImageDataset([paths[i] for i in tr_idx.tolist()], train_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = None
    if va_idx.numel():
        val_ds = EvalImageDataset([paths[i] for i in va_idx.tolist()], eval_tf)
        # Windows: workers are full processes (~1GB each); keep val pool small and
        # persistent so it is spawned once instead of every epoch.
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=True, persistent_workers=True)

    tr_idx_gpu = tr_idx.to(device)
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n]
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lora_lr, "weight_decay": 0.0},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-2},
    ])
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lora_lr, args.head_lr], total_steps=args.epochs * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda")

    va_labels = labels_all[va_idx.to(device)] if va_idx.numel() else None
    va_agree = agree_all[va_idx.to(device)] if va_idx.numel() else None
    history = []
    best_mid = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_seen = 0.0, 0
        for images, idx in train_loader:
            images = images.to(device, non_blocking=True)
            gidx = tr_idx_gpu[idx.to(device)]
            tb = targets_all[gidx]
            wb = weights_all[gidx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss_vec = -(tb * logits.log_softmax(1)).sum(1)
                loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.item() * images.size(0)
            tot_seen += images.size(0)
        msg = f"epoch {epoch:02d}/{args.epochs} loss={tot_loss / max(1, tot_seen):.4f} time={time.time() - t0:.0f}s"
        entry = {"epoch": epoch, "loss": round(tot_loss / max(1, tot_seen), 4)}
        if val_loader is not None:
            bands = evaluate_images(model, val_loader, va_labels, va_agree, device)
            entry.update(bands)
            msg += "  " + "  ".join(f"{k}={v}" for k, v in bands.items())
            if bands["mid_03_06"] > best_mid:
                best_mid = bands["mid_03_06"]
                torch.save({"model": model.state_dict(), "class_names": class_names,
                            "epoch": epoch, "bands": bands, "args": vars(args)},
                           ckpt_dir / "best.pt")
                msg += "  *best*"
        print(msg, flush=True)
        history.append(entry)
        torch.save({"model": model.state_dict(), "class_names": class_names,
                    "epoch": epoch, "args": vars(args)}, ckpt_dir / "last.pt")
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
    print("done")


@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda")
    work = Path(args.work_dir)
    ckpt_path = work / "lora" / ("best.pt" if (work / "lora" / "best.pt").exists() else "last.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    targs = ckpt.get("args", {})
    model = build_model(len(class_names), targs.get("lora_rank", args.lora_rank),
                        targs.get("lora_alpha", args.lora_alpha), 0.0, None, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    _, eval_tf = build_transforms(model.backbone)
    test_dir = Path(args.test_dir)
    paths = sorted(str(p) for p in test_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ds = EvalImageDataset(paths, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    preds = np.empty(len(paths), dtype=np.int64)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            logits = logits + model(torch.flip(images, dims=[3]))
        preds[idx.numpy()] = logits.argmax(1).cpu().numpy()

    out_csv = Path(args.output_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for p, pr in zip(paths, preds):
            writer.writerow([Path(p).name, class_names[pr]])
    zip_path = out_csv.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname="pred_results.csv")
    print(f"saved {out_csv} and {zip_path} ({len(paths)} predictions)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=str(config.TRAIN_DIR))
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    p.add_argument("--output-csv", default="pred_results_lora.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--keep-ratio", type=float, default=0.75)
    p.add_argument("--pseudo-thresh", type=float, default=0.7)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--full", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--predict", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
