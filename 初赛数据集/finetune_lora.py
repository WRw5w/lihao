"""LoRA fine-tuning of CLIP ViT-B/32 on the kNN-denoised training set.

Pipeline (fully scripted, reproducible):
  1. Load cached frozen features -> kNN agreement -> per-class top-75% keep mask
     + high-agreement floor (same recipe as the head-level winner).
  2. Train a frozen-feature teacher head, pseudo-label the dropped samples.
  3. Inject LoRA (rank r) into every attention qkv/proj Linear of the ViT;
     train LoRA + cosine head (warm-started from the teacher) on images with
     augmentation, compact smoothed-label targets and sample weights.
  4. Evaluate per epoch on a stratified noisy val split (banded metrics).
  5. Save resumable checkpoints; --predict requires an explicit checkpoint
     policy.

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
    per_class_topk_keep,
    seed_everything,
    stratified_split,
    train_head,
)
from robust_utils import choose_checkpoint, validate_disjoint_split

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


def inject_lora(model: nn.Module, rank: int, alpha: float, dropout: float, last_blocks: int = 12) -> list[str]:
    replaced = []
    first_block = max(0, len(model.blocks) - last_blocks)
    for blk_i, blk in enumerate(model.blocks):
        if blk_i < first_block:
            continue
        attn = blk.attn
        attn.qkv = LoRALinear(attn.qkv, rank, alpha, dropout)
        attn.proj = LoRALinear(attn.proj, rank, alpha, dropout)
        replaced += [f"blocks.{blk_i}.attn.qkv", f"blocks.{blk_i}.attn.proj"]
    return replaced


# ------------------------------------------------------------------ data


class TrainImageDataset(Dataset):
    """Returns an augmented image and its local dataset index."""

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


def build_transforms(model: nn.Module, crop_min_scale: float = 0.8):
    cfg = resolve_data_config(model=model)
    mean, std = cfg["mean"], cfg["std"]
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(crop_min_scale, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
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
                head_state: dict | None, device: torch.device, lora_blocks: int = 12) -> LoraClassifier:
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad_(False)
    inject_lora(backbone, rank, alpha, lora_dropout, last_blocks=lora_blocks)
    head = CosineClassifier(backbone.num_features, num_classes, dropout=0.0)
    if head_state is not None:
        head.load_state_dict(head_state)
    model = LoraClassifier(backbone, head).to(device)
    return model


# ------------------------------------------------------------------ pipeline


@torch.inference_mode()
def knn_majority_prediction(
    query: torch.Tensor,
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int,
    num_classes: int,
    *,
    exclude_self: bool,
    chunk: int = 2048,
) -> torch.Tensor:
    predictions = torch.empty(query.size(0), dtype=torch.long, device=query.device)
    gallery_t = gallery.t().contiguous()
    for start in range(0, query.size(0), chunk):
        end = min(start + chunk, query.size(0))
        sim = query[start:end] @ gallery_t
        if exclude_self:
            rows = torch.arange(start, end, device=sim.device)
            sim[torch.arange(end - start, device=sim.device), rows] = -2.0
        topv, topi = sim.topk(k, dim=1)
        votes = torch.zeros((end - start, num_classes), device=sim.device)
        votes.scatter_add_(1, gallery_labels[topi], topv.float().clamp_min(0))
        predictions[start:end] = votes.argmax(1)
    return predictions


@torch.inference_mode()
def teacher_stats(model, features: torch.Tensor, labels: torch.Tensor, batch: int = 16384):
    model.eval()
    p_label, preds, pmax, margins = [], [], [], []
    for start in range(0, features.size(0), batch):
        logits = model(features[start : start + batch])
        probs = logits.softmax(1)
        top2 = probs.topk(2, dim=1)
        yb = labels[start : start + batch]
        p_label.append(probs.gather(1, yb[:, None]).squeeze(1))
        preds.append(top2.indices[:, 0])
        pmax.append(top2.values[:, 0])
        margins.append(top2.values[:, 0] - top2.values[:, 1])
    return tuple(torch.cat(x) for x in (p_label, preds, pmax, margins))


def prepare_targets(args, device, train_idx: torch.Tensor, val_idx: torch.Tensor) -> dict:
    """Frozen-feature stage: kNN keep mask + teacher + compact targets/weights.

    Returns everything index-aligned with ImageFolder ordering.
    """
    work = Path(args.work_dir)
    cache = torch.load(work / "cache" / "train_features.pt", map_location="cpu")
    feats, labels, class_names = cache["features"], cache["labels"], cache["class_names"]
    image_names = cache["image_names"]
    num_classes = len(class_names)
    validate_disjoint_split(train_idx.tolist(), val_idx.tolist())
    f16 = feats.to(device)
    f32 = feats.to(device, dtype=torch.float32)
    y = labels.to(device)
    tr = train_idx.to(device)
    va = val_idx.to(device)
    ftr, ytr = f16[tr], y[tr]

    seed_everything(args.seed)
    agree = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    agree[tr] = knn_agreement(ftr, ytr, ftr, ytr, k=args.knn_k, exclude_self=True).to(device)
    if va.numel():
        agree[va] = knn_agreement(f16[va], y[va], ftr, ytr, k=args.knn_k, exclude_self=False).to(device)
    keep = torch.zeros_like(y, dtype=torch.bool)
    keep_tr = per_class_topk_keep(agree[tr], ytr, num_classes, keep_ratio=args.keep_ratio)
    keep_tr |= agree[tr] >= args.high_agreement_floor
    keep[tr] = keep_tr
    idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
    print(f"kNN filter keeps {idx_keep.numel()}/{y.numel()} ({idx_keep.numel() / y.numel():.2%})")

    teacher = train_head(f32, y, num_classes, idx_keep, smoothing=0.1, epochs=args.teacher_epochs,
                         batch_size=8192, device=device)
    p_label, preds_t, pmax_t, margins_t = teacher_stats(teacher, f32, y)
    knn_preds_tr = knn_majority_prediction(
        ftr, ftr, ytr, args.knn_k, num_classes, exclude_self=True)

    target_labels = y.clone()
    weights = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    reliability = (
        agree[tr].clamp(0, 1).pow(args.agreement_power)
        * p_label[tr].clamp(0, 1).pow(args.confidence_power)
        * margins_t[tr].clamp(0, 1).pow(args.margin_power)
    )
    weights[tr[keep_tr]] = args.min_sample_weight + (1.0 - args.min_sample_weight) * reliability[keep_tr]
    pseudo_tr = (
        (~keep_tr)
        & (preds_t[tr] == knn_preds_tr)
        & (pmax_t[tr] >= args.pseudo_thresh)
        & (margins_t[tr] >= args.pseudo_margin)
    )
    if pseudo_tr.any():
        pseudo_idx = tr[pseudo_tr]
        target_labels[pseudo_idx] = preds_t[pseudo_idx]
        weights[pseudo_idx] = pmax_t[pseudo_idx] * margins_t[pseudo_idx].sqrt()
    pseudo_count = int(pseudo_tr.sum())
    print(f"targets ready: {int((weights > 0).sum())} usable samples "
          f"({pseudo_count} consensus pseudo-labelled)")
    return {
        "target_labels": target_labels.cpu(), "weights": weights.cpu(), "labels": labels,
        "class_names": class_names, "image_names": image_names,
        "teacher_head_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
        "agree": agree.cpu(),
        "target_stats": {
            "kept": int(keep.sum()), "pseudo": pseudo_count,
            "mean_kept_weight": round(float(weights[keep].mean()), 6),
        },
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

    cache = torch.load(work / "cache" / "train_features.pt", map_location="cpu")
    labels = cache["labels"]
    if args.full:
        tr_idx = torch.arange(labels.numel())
        va_idx = torch.arange(0)
    else:
        tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=generator)[:5000]].sort().values
        va_idx = va_idx[torch.randperm(va_idx.numel(), generator=generator)[:2000]].sort().values if va_idx.numel() else va_idx
        args.epochs = 2

    prep = prepare_targets(args, device, tr_idx, va_idx)
    class_names = prep["class_names"]
    num_classes = len(class_names)
    target_labels_all = prep["target_labels"].to(device)
    weights_all = prep["weights"].to(device)
    labels_all = prep["labels"].to(device)
    agree_all = prep["agree"].to(device)

    base = ImageFolder(args.train_dir)
    paths = [p for p, _ in base.samples]
    names_check = [Path(p).name for p in paths]
    assert names_check == prep["image_names"], "ImageFolder order mismatch with feature cache"

    usable = (weights_all[tr_idx.to(device)] > 0).cpu()
    tr_idx = tr_idx[usable]
    print(f"train images {tr_idx.numel()}  val images {va_idx.numel()}")

    model = build_model(num_classes, args.lora_rank, args.lora_alpha, args.lora_dropout,
                        prep["teacher_head_state"], device, lora_blocks=args.lora_blocks)
    train_tf, eval_tf = build_transforms(model.backbone, args.crop_min_scale)
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
    start_epoch = 1
    if args.resume:
        resume = torch.load(Path(args.resume), map_location=device)
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        print(f"resuming {args.resume} from epoch {start_epoch}")

    def checkpoint_payload(epoch: int, bands: dict | None = None) -> dict:
        return {
            "model": model.state_dict(), "class_names": class_names,
            "epoch": epoch, "bands": bands, "args": vars(args),
            "target_stats": prep["target_stats"],
            "train_idx": tr_idx, "val_idx": va_idx,
            "model_name": MODEL_NAME, "single_model": True,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
        }

    va_labels = labels_all[va_idx.to(device)] if va_idx.numel() else None
    va_agree = agree_all[va_idx.to(device)] if va_idx.numel() else None
    history = []
    best_mid = -1.0
    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_seen = 0.0, 0
        for images, idx in train_loader:
            images = images.to(device, non_blocking=True)
            gidx = tr_idx_gpu[idx.to(device)]
            tb = target_labels_all[gidx]
            wb = weights_all[gidx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                loss_vec = F.cross_entropy(
                    logits, tb, reduction="none", label_smoothing=args.label_smoothing)
                loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                stale_epochs = 0
                torch.save(checkpoint_payload(epoch, bands), ckpt_dir / "best.pt")
                msg += "  *best*"
            else:
                stale_epochs += 1
        print(msg, flush=True)
        history.append(entry)
        torch.save(checkpoint_payload(epoch), ckpt_dir / ("full.pt" if args.full else "last.pt"))
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        if val_loader is not None and args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f"early stopping after {stale_epochs} stale epochs")
            break
    print("done")


@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda")
    work = Path(args.work_dir)
    ckpt_path = choose_checkpoint(work / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    targs = ckpt.get("args", {})
    model = build_model(len(class_names), targs.get("lora_rank", args.lora_rank),
                        targs.get("lora_alpha", args.lora_alpha), 0.0, None, device,
                        lora_blocks=targs.get("lora_blocks", args.lora_blocks))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    _, eval_tf = build_transforms(model.backbone, targs.get("crop_min_scale", args.crop_min_scale))
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
            if not args.no_flip_tta:
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
    p.add_argument("--pseudo-margin", type=float, default=0.2)
    p.add_argument("--high-agreement-floor", type=float, default=0.7)
    p.add_argument("--min-sample-weight", type=float, default=0.2)
    p.add_argument("--agreement-power", type=float, default=1.0)
    p.add_argument("--confidence-power", type=float, default=0.5)
    p.add_argument("--margin-power", type=float, default=0.5)
    p.add_argument("--teacher-epochs", type=int, default=20)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-blocks", type=int, default=12, choices=(4, 6, 12))
    p.add_argument("--crop-min-scale", type=float, default=0.8)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="best")
    p.add_argument("--resume", default=None, help="resume from a training checkpoint using the same schedule")
    p.add_argument("--no-flip-tta", action="store_true")
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
