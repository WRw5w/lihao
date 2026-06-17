"""FET-FGVC + LoRA fine-tuning of CLIP ViT-B/32 (thin entry).

New on top of finetune_lora.py:
  - FETLoraClassifier: CLIP LoRA backbone + LocalBranch (CBAM + PartTransformer)
  - PFI auxiliary loss (pair feature interaction, --pfi-weight to control)
  - Balanced batch sampler (--pfi-classes / --pfi-images)
  - Separate LR group for local branch & head vs. LoRA backbone

Pipeline:
  1. Strict split-first: kNN/teacher/pseudo-labels from train partition only.
  2. FET local branch operates on patch tokens; decision_mask = all-ones
     (CLIP ViT-B/32 has no dynamic token pruning).
  3. At training time, PFI pairs are drawn per-batch; intra/inter CE auxiliary
     loss is added with weight --pfi-weight (default 0.5).
  4. Val / predict paths are identical to finetune_lora.py (FETLoraClassifier
     forward without targets returns plain logits).

Run:
  python finetune_fet.py --epochs 6 --img-size 448
  python finetune_fet.py --full --epochs 6 --img-size 448
  python finetune_fet.py --predict --checkpoint best --work-dir outputs_fet_c448
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision.datasets import ImageFolder

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.denoise import knn_agreement, knn_majority_prediction, per_class_topk_keep
from robustft.engine import (
    seed_everything,
    stratified_split,
    stratified_three_way_split,
    teacher_stats,
    train_head,
)
from robustft.fet_model import FETLoraClassifier, build_fet_model
from robustft.robust_utils import choose_checkpoint, validate_disjoint_split
from robustft.submission import save_predictions, zip_submission


# ---------------------------------------------------------------------------
# Balanced batch sampler (FET requirement for PFI)
# ---------------------------------------------------------------------------

class BalancedBatchSampler(Sampler):
    """Yield mini-batches with exactly n_classes classes, n_images each."""

    def __init__(self, labels: list[int], n_classes: int, n_images: int, seed: int = 42):
        self.n_classes = n_classes
        self.n_images = n_images
        self.batch_size = n_classes * n_images
        self.rng = random.Random(seed)

        from collections import defaultdict
        self.label2idx: dict[int, list[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            self.label2idx[lbl].append(i)
        self.classes = list(self.label2idx.keys())

    def __iter__(self):
        indices_by_class = {c: list(v) for c, v in self.label2idx.items()}
        for v in indices_by_class.values():
            self.rng.shuffle(v)
        ptr = {c: 0 for c in self.classes}
        # yield EXACTLY len(self) batches so steps_per_epoch matches the LR
        # scheduler's total_steps (OneCycleLR errors if stepped past the end).
        for _ in range(len(self)):
            chosen = self.rng.sample(self.classes, min(self.n_classes, len(self.classes)))
            batch = []
            for c in chosen:
                pool = indices_by_class[c]
                if ptr[c] + self.n_images > len(pool):
                    self.rng.shuffle(pool)
                    ptr[c] = 0
                batch.extend(pool[ptr[c]:ptr[c] + self.n_images])
                ptr[c] += self.n_images
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.label2idx.values())
        return total // self.batch_size


# ---------------------------------------------------------------------------
# Target preparation (identical to finetune_lora.py)
# ---------------------------------------------------------------------------

def prepare_targets(args, device, train_idx: torch.Tensor, val_idx: torch.Tensor) -> dict:
    cache = torch.load(Path(args.cache_dir) / "train_features.pt", map_location="cpu")
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
    knn_preds_tr = knn_majority_prediction(ftr, ftr, ytr, args.knn_k, num_classes, exclude_self=True)

    # Iterative relabeling: replace the WEAK linear-teacher stats with a strong
    # trained model's predictions (a far better label cleaner) -> the same
    # consensus/pseudo rules below then recover/relabel more of the noisy labels.
    # Off by default (no path) so baseline is unaffected. (teacher head kept for
    # head warm-start; only the pseudo-decision stats are overridden.)
    if args.teacher_preds_path and Path(args.teacher_preds_path).exists():
        mp = torch.load(args.teacher_preds_path, map_location="cpu")
        assert mp["image_names"] == image_names, "model-preds image order != feature cache"
        probs = mp["probs"].to(device).float()
        preds_t = probs.argmax(1)
        pmax_t = probs.max(1).values
        top2 = probs.topk(2, dim=1).values
        margins_t = top2[:, 0] - top2[:, 1]
        p_label = probs.gather(1, y.view(-1, 1)).squeeze(1)
        would = ((~keep[tr]) & (preds_t[tr] == knn_preds_tr)
                 & (pmax_t[tr] >= args.pseudo_thresh) & (margins_t[tr] >= args.pseudo_margin))
        print(f"[iter-relabel] model preds from {Path(args.teacher_preds_path).name}: "
              f"mean_conf={pmax_t.mean():.3f}, would recover {int(would.sum())} discarded "
              f"(linear-teacher recovered ~84)")

    target_labels = y.clone()
    weights = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    sig_terms = [
        (agree[tr].clamp(0, 1), args.agreement_power),
        (p_label[tr].clamp(0, 1), args.confidence_power),
        (margins_t[tr].clamp(0, 1), args.margin_power),
    ]
    if args.reliability_mode == "sum":
        num = sum(w * s for s, w in sig_terms)
        den = sum(w for _, w in sig_terms)
        reliability = (num / max(den, 1e-6)).clamp(0, 1)
    else:
        reliability = torch.ones(tr.numel(), device=device)
        for s, w in sig_terms:
            reliability = reliability * s.pow(w)
    weights[tr[keep_tr]] = args.min_sample_weight + (1.0 - args.min_sample_weight) * reliability[keep_tr]

    soft_alpha = torch.zeros(y.numel(), dtype=torch.float32, device=device)
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
        soft_alpha[pseudo_idx] = args.pseudo_soft_alpha
    pseudo_count = int(pseudo_tr.sum())
    print(f"targets ready: {int((weights > 0).sum())} usable samples ({pseudo_count} pseudo-labelled)")
    return {
        "target_labels": target_labels.cpu(), "weights": weights.cpu(), "labels": labels,
        "class_names": class_names, "image_names": image_names,
        "teacher_head_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
        "agree": agree.cpu(), "teacher_preds": preds_t.cpu(), "soft_alpha": soft_alpha.cpu(),
        "target_stats": {"kept": int(keep.sum()), "pseudo": pseudo_count},
    }


# ---------------------------------------------------------------------------
# Evaluation (identical to finetune_lora.py)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_images(model, loader, labels_gpu, agree_gpu, device) -> dict:
    model.eval()
    correct = torch.zeros_like(labels_gpu, dtype=torch.bool)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)   # FETLoraClassifier returns plain logits without targets
        correct[idx.to(device)] = logits.argmax(1) == labels_gpu[idx.to(device)]
    bands = {
        "noisy_all":  torch.ones_like(labels_gpu, dtype=torch.bool),
        "low_lt03":   agree_gpu < 0.3,
        "mid_03_06":  (agree_gpu >= 0.3) & (agree_gpu < 0.6),
        "high_ge06":  agree_gpu >= 0.6,
    }
    return {k: round((correct & m).sum().item() / max(1, int(m.sum())), 4) for k, m in bands.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device("cuda")
    ckpt_dir = Path(args.work_dir) / "lora"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cache = torch.load(Path(args.cache_dir) / "train_features.pt", map_location="cpu")
    labels = cache["labels"]
    ho_idx = torch.arange(0)
    if args.full:
        tr_idx = torch.arange(labels.numel())
        va_idx = torch.arange(0)
    else:
        tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)

    if args.smoke:
        g = torch.Generator().manual_seed(args.seed)
        tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=g)[:5000]].sort().values
        va_idx = va_idx[torch.randperm(va_idx.numel(), generator=g)[:2000]].sort().values if va_idx.numel() else va_idx
        args.epochs = 2

    prep = prepare_targets(args, device, tr_idx, va_idx)
    class_names = prep["class_names"]
    num_classes = len(class_names)
    target_labels_all = prep["target_labels"].to(device)
    weights_all = prep["weights"].to(device)
    labels_all = prep["labels"].to(device)
    agree_all = prep["agree"].to(device)
    soft_alpha_all = prep["soft_alpha"].to(device)

    # ELR (Early-Learning Regularization): per-sample temporal-ensemble target.
    # Off by default (elr_lambda=0) so the FET baseline is unaffected (clean A/B).
    elr_targets = (torch.zeros(labels_all.numel(), num_classes, device=device)
                   if args.elr_lambda > 0 else None)
    if elr_targets is not None:
        print(f"ELR on: lambda={args.elr_lambda} beta={args.elr_beta} "
              f"buffer={tuple(elr_targets.shape)} ({elr_targets.numel() * 4 / 1e6:.0f}MB)")

    base = ImageFolder(args.train_dir)
    paths = [p for p, _ in base.samples]
    names_check = [Path(p).name for p in paths]
    assert names_check == prep["image_names"], "ImageFolder order mismatch with feature cache"

    usable = (weights_all[tr_idx.to(device)] > 0).cpu()
    tr_idx = tr_idx[usable]
    print(f"train images {tr_idx.numel()}  val images {va_idx.numel()}")

    model = build_fet_model(
        num_classes=num_classes,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        head_state=prep["teacher_head_state"],
        device=device,
        lora_blocks=args.lora_blocks,
        lora_target=args.lora_target,
        img_size=args.img_size,
        num_parts=args.num_parts,
        part_channels=args.part_channels,
        local_depth=args.local_depth,
        local_scale=args.local_scale,
        use_pfi=args.pfi_weight > 0,
        gaussian_ksize=args.gaussian_ksize,
    )

    train_tf, eval_tf = build_finetune_transforms(model.backbone, args.crop_min_scale,
                                                  img_size=args.img_size, randaug=args.randaug)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.2f}M")

    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    pin = not args.no_pin
    tr_paths = [paths[i] for i in tr_idx.tolist()]
    tr_labels = labels_all[tr_idx.to(device)].cpu().tolist()
    train_ds = IndexedImageDataset(tr_paths, train_tf)

    if args.pfi_weight > 0 and args.pfi_classes > 1:
        sampler = BalancedBatchSampler(tr_labels, args.pfi_classes, args.pfi_images, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=args.num_workers > 0)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=pin,
                                  persistent_workers=args.num_workers > 0, drop_last=True)

    val_loader = None
    if va_idx.numel():
        val_ds = IndexedImageDataset([paths[i] for i in va_idx.tolist()], eval_tf)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=pin, persistent_workers=True)

    tr_idx_gpu = tr_idx.to(device)

    lora_params  = [p for n, p in model.named_parameters() if p.requires_grad and "head" not in n and "local_branch" not in n and "local_proj" not in n and "pfi_" not in n]
    local_params = [p for n, p in model.named_parameters() if p.requires_grad and ("local_branch" in n or "local_proj" in n or "pfi_" in n)]
    head_params  = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {"params": lora_params,  "lr": args.lora_lr,  "weight_decay": 0.0},
        {"params": local_params, "lr": args.local_lr, "weight_decay": 1e-4},
        {"params": head_params,  "lr": args.head_lr,  "weight_decay": 1e-2},
    ])
    steps_per_epoch = len(train_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[args.lora_lr, args.local_lr, args.head_lr],
        total_steps=args.epochs * steps_per_epoch, pct_start=0.1, anneal_strategy="cos")
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if args.resume:
        resume = torch.load(Path(args.resume), map_location=device)
        model.load_state_dict(resume.get("raw_model", resume["model"]))
        if ema_model is not None:
            ema_model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        print(f"resuming from epoch {start_epoch}")

    def checkpoint_payload(epoch, bands=None):
        eval_model = ema_model if ema_model is not None else model
        extra = {"raw_model": model.state_dict()} if ema_model is not None else {}
        return {
            "model": eval_model.state_dict(), "class_names": class_names,
            **extra,
            "epoch": epoch, "bands": bands, "args": vars(args),
            "target_stats": prep["target_stats"],
            "train_idx": tr_idx, "val_idx": va_idx,
            "model_name": "fet_lora", "single_model": True,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
        }

    va_labels = labels_all[va_idx.to(device)] if va_idx.numel() else None
    va_agree  = agree_all[va_idx.to(device)]  if va_idx.numel() else None
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
                out = model(images, tb)

                if isinstance(out, tuple):
                    logits, pfi_logits, pfi_targets = out
                else:
                    logits = out
                    pfi_logits = pfi_targets = None

                loss_vec = F.cross_entropy(logits, tb, reduction="none",
                                           label_smoothing=args.label_smoothing)
                if args.pseudo_soft_alpha > 0:
                    ab = soft_alpha_all[gidx]
                    ce_orig = F.cross_entropy(logits, labels_all[gidx], reduction="none",
                                              label_smoothing=args.label_smoothing)
                    loss_vec = (1.0 - ab) * loss_vec + ab * ce_orig
                main_loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)

                loss = main_loss
                if pfi_logits is not None and args.pfi_weight > 0:
                    pfi_loss = F.cross_entropy(pfi_logits, pfi_targets,
                                               label_smoothing=args.label_smoothing)
                    loss = loss + args.pfi_weight * pfi_loss

                if elr_targets is not None:
                    # temporal-ensemble target t = EMA of softmax; reg log(1-<p,t>)
                    # pulls current p toward the early-learned (clean) target.
                    prob = logits.float().softmax(1)
                    with torch.no_grad():
                        t = elr_targets[gidx].mul_(args.elr_beta).add_((1 - args.elr_beta) * prob.detach())
                        elr_targets[gidx] = t
                    inner = (prob * t).sum(1).clamp(max=1 - 1e-4)
                    loss = loss + args.elr_lambda * torch.log(1.0 - inner).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()

            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        if pm.requires_grad:
                            pe.mul_(args.ema_decay).add_(pm, alpha=1.0 - args.ema_decay)

            tot_loss += loss.item() * images.size(0)
            tot_seen += images.size(0)

        msg = f"epoch {epoch:02d}/{args.epochs} loss={tot_loss / max(1, tot_seen):.4f} time={time.time() - t0:.0f}s"
        entry = {"epoch": epoch, "loss": round(tot_loss / max(1, tot_seen), 4)}

        if val_loader is not None:
            bands = evaluate_images(
                ema_model if ema_model is not None else model,
                val_loader, va_labels, va_agree, device)
            entry.update(bands)
            msg += "  " + "  ".join(f"{k}={v}" for k, v in bands.items())
            if bands["mid_03_06"] > best_mid:
                best_mid = bands["mid_03_06"]
                stale_epochs = 0
                torch.save(checkpoint_payload(epoch, bands), ckpt_dir / "best.pt")
                msg += "  *best*"
                if args.snapshot_after > 0 and epoch >= args.snapshot_after:
                    torch.save(checkpoint_payload(epoch, bands), ckpt_dir / f"best_ep{epoch:02d}.pt")
                    msg += "  [snap]"
            else:
                stale_epochs += 1
        print(msg, flush=True)
        history.append(entry)
        torch.save(checkpoint_payload(epoch), ckpt_dir / ("full.pt" if args.full else "last.pt"))
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(checkpoint_payload(epoch), ckpt_dir / f"ep{epoch:02d}.pt")
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
        if val_loader is not None and args.early_stop_patience > 0 and stale_epochs >= args.early_stop_patience:
            print(f"early stopping after {stale_epochs} stale epochs")
            break
    print("done")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda")
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    targs = ckpt.get("args", {})

    model = build_fet_model(
        num_classes=len(class_names),
        rank=targs.get("lora_rank", args.lora_rank),
        alpha=targs.get("lora_alpha", args.lora_alpha),
        lora_dropout=0.0,
        head_state=None,
        device=device,
        lora_blocks=targs.get("lora_blocks", args.lora_blocks),
        lora_target=targs.get("lora_target", args.lora_target),
        img_size=targs.get("img_size", args.img_size),
        num_parts=targs.get("num_parts", args.num_parts),
        part_channels=targs.get("part_channels", args.part_channels),
        local_depth=targs.get("local_depth", args.local_depth),
        local_scale=targs.get("local_scale", args.local_scale),
        use_pfi=False,
        gaussian_ksize=targs.get("gaussian_ksize", args.gaussian_ksize),
    )
    # checkpoint EMA model has train-only pfi_* params; we build use_pfi=False.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    _, eval_tf = build_finetune_transforms(model.backbone,
                                           targs.get("crop_min_scale", args.crop_min_scale),
                                           img_size=targs.get("img_size", args.img_size))
    paths = sorted(str(p) for p in Path(args.test_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    ds = IndexedImageDataset(paths, eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=args.num_workers, pin_memory=not args.no_pin)
    preds = np.empty(len(paths), dtype=np.int64)
    for images, idx in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(images)
            if not args.no_flip_tta:
                logits = logits + model(torch.flip(images, dims=[3]))
        preds[idx.numpy()] = logits.argmax(1).cpu().numpy()

    out_csv = Path(args.output_csv)
    save_predictions(out_csv, [Path(p).name for p in paths], preds.tolist(), class_names)
    zip_path = zip_submission(out_csv)
    print(f"saved {out_csv} and {zip_path} ({len(paths)} predictions)")


# ---------------------------------------------------------------------------
# Arg parser
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    # paths
    p.add_argument("--train-dir",   default=str(config.TRAIN_DIR))
    p.add_argument("--test-dir",    default=str(config.TEST_DIR))
    p.add_argument("--work-dir",    default="outputs_fet_c448")
    p.add_argument("--cache-dir",   default="outputs/cache",
                   help="feature cache dir (shared with finetune_lora; default: outputs/cache)")
    p.add_argument("--output-csv",  default=str(config.SUBMISSIONS_DIR / "pred_results_fet.csv"))
    # split / denoising
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--val-frac",          type=float, default=0.1)
    p.add_argument("--knn-k",             type=int,   default=16)
    p.add_argument("--keep-ratio",        type=float, default=0.90)
    p.add_argument("--pseudo-thresh",     type=float, default=0.6)
    p.add_argument("--pseudo-margin",     type=float, default=0.05)
    p.add_argument("--high-agreement-floor", type=float, default=0.7)
    p.add_argument("--min-sample-weight", type=float, default=0.2)
    p.add_argument("--agreement-power",   type=float, default=1.0)
    p.add_argument("--confidence-power",  type=float, default=0.5)
    p.add_argument("--margin-power",      type=float, default=0.5)
    p.add_argument("--reliability-mode",  choices=("mul", "sum"), default="mul")
    p.add_argument("--pseudo-soft-alpha", type=float, default=0.05)
    p.add_argument("--teacher-epochs",    type=int,   default=20)
    # training
    p.add_argument("--epochs",          type=int,   default=6)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--num-workers",     type=int,   default=2)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--save-every",      type=int,   default=0)
    p.add_argument("--snapshot-after",  type=int,   default=0)
    # ELR (Early-Learning Regularization) -- noise-robust reg, off by default
    p.add_argument("--elr-lambda", type=float, default=0.0,
                   help="ELR strength (0=off). Counteracts memorizing noisy labels; try 1-3.")
    p.add_argument("--elr-beta",   type=float, default=0.7, help="ELR temporal-ensemble EMA decay")
    p.add_argument("--teacher-preds-path", default="",
                   help="iterative relabel: .pt of a trained model's train-set probs "
                        "(from tools/extract_model_preds.py) to use as the pseudo-label teacher")
    # LoRA
    p.add_argument("--lora-rank",    type=int,   default=32)
    p.add_argument("--lora-alpha",   type=float, default=64.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-blocks",  type=int,   default=12, choices=(4, 6, 12))
    p.add_argument("--lora-target",  choices=("attn", "attn_mlp"), default="attn_mlp")
    # image / augmentation
    p.add_argument("--img-size",        type=int,   default=448)
    p.add_argument("--crop-min-scale",  type=float, default=0.8)
    p.add_argument("--randaug",         action="store_true", default=True)
    p.add_argument("--no-randaug",      dest="randaug", action="store_false")
    # FET local branch
    p.add_argument("--num-parts",      type=int,   default=8)
    p.add_argument("--part-channels",  type=int,   default=16)
    p.add_argument("--local-depth",    type=int,   default=2)
    p.add_argument("--local-scale",    type=float, default=0.5)
    p.add_argument("--gaussian-ksize", type=int,   default=15)
    # PFI
    p.add_argument("--pfi-weight",   type=float, default=0.5,
                   help="PFI auxiliary CE loss weight (0=disable PFI)")
    p.add_argument("--pfi-classes",  type=int,   default=4,
                   help="classes per mini-batch for balanced sampler (PFI)")
    p.add_argument("--pfi-images",   type=int,   default=16,
                   help="images per class per mini-batch (PFI). batch_size = pfi-classes * pfi-images")
    # optimizer / scheduler
    p.add_argument("--lora-lr",  type=float, default=2e-4)
    p.add_argument("--local-lr", type=float, default=5e-4)
    p.add_argument("--head-lr",  type=float, default=1e-3)
    p.add_argument("--ema-decay",type=float, default=0.999)
    # EMA / checkpointing
    p.add_argument("--resume",     default=None)
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="best")
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--no-pin",      action="store_true")
    p.add_argument("--full",        action="store_true")
    p.add_argument("--smoke",       action="store_true")
    p.add_argument("--predict",     action="store_true")
    args = p.parse_args()
    # When using balanced sampler, actual batch_size = pfi_classes * pfi_images
    if args.pfi_weight > 0 and args.pfi_classes > 1:
        args.batch_size = args.pfi_classes * args.pfi_images
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
