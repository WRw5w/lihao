"""LoRA fine-tuning of CLIP ViT-B/32 on the kNN-denoised training set (thin entry).

Pipeline (fully scripted, reproducible, strict validation):
  1. Create the stratified train/val split BEFORE any label-driven step.
  2. kNN agreement, keep mask and teacher training use the training partition
     only; validation samples only query the training gallery.
  3. Recover dropped samples solely via teacher-kNN consensus pseudo-labels;
     kept samples get continuous reliability weights.
  4. Inject LoRA into the last N attention blocks of the ViT; train LoRA +
     cosine head (warm-started from the teacher) on images with augmentation,
     compact smoothed-label targets and sample weights.
  5. Evaluate per epoch on the held-out noisy val split (banded metrics).
  6. Save resumable checkpoints; --predict requires an explicit checkpoint
     policy (best / last / full).

Run:  python finetune_lora.py --epochs 15          (train, 90/10 split)
      python finetune_lora.py --epochs 15 --holdout-frac 0.1
                                                   (8:1:1; final unbiased holdout report)
      python finetune_lora.py --full --epochs 15   (train on 100% -> full.pt)
      python finetune_lora.py --predict --checkpoint full
      python finetune_lora.py --smoke --work-dir outputs_tmp --cache-dir outputs/cache
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

import config
from robustft.data import IndexedImageDataset, build_finetune_transforms
from robustft.denoise import (
    confident_learning_keep,
    fit_gmm_1d,
    knn_agreement,
    knn_majority_prediction,
    label_propagation,
    per_class_topk_keep,
    sinkhorn_balance,
)
from robustft.engine import (
    seed_everything,
    stratified_split,
    stratified_three_way_split,
    teacher_stats,
    train_head,
)
from robustft.models import MODEL_NAME, build_lora_model
from robustft.robust_utils import choose_checkpoint, validate_disjoint_split
from robustft.submission import save_predictions, zip_submission


def _loss_vec(logits: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    """Per-sample loss for the chosen robust objective (no soft-alpha; used by the
    mixup wrapper and the gce/sce/apl branches). Computed in fp32 for log stability."""
    if args.robust_loss == "gce":
        # Generalized Cross-Entropy (Zhang & Sabuncu 2018): L_q=(1-p_y^q)/q.
        p_y = logits.float().softmax(1).gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-6)
        return (1.0 - p_y.pow(args.gce_q)) / args.gce_q
    if args.robust_loss == "sce":
        # Symmetric Cross-Entropy (Wang 2019): alpha*CE + beta*RCE; RCE is robust.
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=args.label_smoothing)
        p = logits.float().softmax(1).clamp(1e-7, 1.0)
        oh = F.one_hot(target, logits.size(1)).float().clamp_min(1e-4)
        rce = -(p * oh.log()).sum(1)
        return args.sce_alpha * ce + args.sce_beta * rce
    if args.robust_loss == "apl":
        # Active Passive Loss (Ma 2020): normalized CE (active) + RCE (passive),
        # theoretically robust to symmetric noise.
        lp = logits.float().log_softmax(1)
        nce = (-lp.gather(1, target.unsqueeze(1)).squeeze(1)) / (-lp.sum(1)).clamp_min(1e-6)
        p = lp.exp().clamp(1e-7, 1.0)
        oh = F.one_hot(target, logits.size(1)).float().clamp_min(1e-4)
        rce = -(p * oh.log()).sum(1)
        return args.apl_alpha * nce + args.apl_beta * rce
    return F.cross_entropy(logits, target, reduction="none", label_smoothing=args.label_smoothing)


def prepare_targets(args, device, train_idx: torch.Tensor, val_idx: torch.Tensor) -> dict:
    """Frozen-feature stage: kNN keep mask + teacher + compact targets/weights.

    Split-first: every label-driven statistic is computed from the training
    partition only. Returns everything index-aligned with ImageFolder ordering.
    """
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
    knn_preds_tr = knn_majority_prediction(
        ftr, ftr, ytr, args.knn_k, num_classes, exclude_self=True)

    cl_relabel_info = None
    if args.denoise in ("cleanlab", "cleanlab_knn", "cleanlab_relabel", "ot", "otknn"):
        # Confident Learning on teacher probs. "ot": first Sinkhorn-balance the
        # probs to the uniform test prior (exploit balanced-test) before selection.
        P = teacher(f32[tr]).softmax(1)
        if args.denoise in ("ot", "otknn"):
            P = sinkhorn_balance(P, iters=args.ot_iters)
        ck, cp, cc, ca = confident_learning_keep(P, ytr, num_classes)
        if args.denoise in ("cleanlab_knn", "otknn"):
            ck = ck & keep_tr  # stricter: agree AND confident
        if args.denoise in ("cleanlab_relabel", "ot", "otknn"):
            cl_relabel_info = (cp, cc, ca & (cp != ytr))  # relabel confident-wrong
        keep_tr = ck
    elif args.denoise == "labelprop":
        # propagate labels on the feature kNN graph from high-agreement anchors;
        # keep where propagation agrees with the label, relabel confident-disagree.
        anchors = torch.nonzero(agree[tr] >= args.lp_anchor_agree, as_tuple=False).squeeze(1)
        Z = label_propagation(F.normalize(f32[tr], dim=1), ytr, anchors, num_classes,
                              k=args.knn_k, alpha=args.lp_alpha)
        zp, zc = Z.argmax(1), Z.max(1).values
        has = Z.sum(1) > 1e-6
        keep_tr = ((zp == ytr) & has) | (~has)
        cl_relabel_info = (zp, zc, has & (zp != ytr))
    elif args.denoise == "divmix":
        # DivideMix-lite: GMM on per-sample teacher loss splits clean/noisy; noisy
        # samples are relabelled to the teacher prediction (semi-supervised co-refine).
        P = teacher(f32[tr]).softmax(1)
        ce = -(P.gather(1, ytr[:, None]).squeeze(1).clamp_min(1e-8).log())
        clean_prob = fit_gmm_1d(ce)
        keep_tr = clean_prob >= args.divmix_thresh
        tp, tc = P.argmax(1), P.max(1).values
        cl_relabel_info = (tp, tc * (1 - clean_prob), (clean_prob < args.divmix_thresh) & (tp != ytr))
    if args.denoise != "knn":
        keep[tr] = keep_tr
        idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
        print(f"denoise[{args.denoise}] keeps {int(keep_tr.sum())}/{tr.numel()} "
              f"({keep_tr.float().mean():.2%})")

    target_labels = y.clone()
    weights = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    # Composite reliability signals: (signal, weight). Combined either by product
    # (mul, aggressive: any low factor -> ~0 weight) or weighted average (sum,
    # conservative: no single factor can zero a sample out).
    sig_terms = [
        (agree[tr].clamp(0, 1), args.agreement_power),
        (p_label[tr].clamp(0, 1), args.confidence_power),
        (margins_t[tr].clamp(0, 1), args.margin_power),
    ]
    if args.proto_power > 0:
        # per-class visual prototype = centroid of kept samples; similarity is a
        # global per-class signal complementing local kNN agreement.
        ftr32 = f32[tr]
        kept_pos = torch.nonzero(keep_tr, as_tuple=False).squeeze(1)
        proto = torch.zeros(num_classes, ftr32.size(1), device=device)
        proto.index_add_(0, ytr[kept_pos], ftr32[kept_pos])
        cnt = torch.bincount(ytr[kept_pos], minlength=num_classes).clamp_min(1).unsqueeze(1)
        proto = F.normalize(proto / cnt, dim=1)
        proto_sim = (F.normalize(ftr32, dim=1) * proto[ytr]).sum(1).clamp(0, 1)
        sig_terms.append((proto_sim, args.proto_power))
    if args.aug_consist_power > 0 and Path(args.aug_consist_path).exists():
        ac = torch.load(args.aug_consist_path, map_location="cpu")["consistency"].to(device)
        sig_terms.append((ac[tr].clamp(0, 1), args.aug_consist_power))
    if args.reliability_mode == "sum":
        num = sum(w * s for s, w in sig_terms)
        den = sum(w for _, w in sig_terms)
        reliability = (num / max(den, 1e-6)).clamp(0, 1)
    else:
        reliability = torch.ones(tr.numel(), device=device)
        for s, w in sig_terms:
            reliability = reliability * s.pow(w)
    weights[tr[keep_tr]] = args.min_sample_weight + (1.0 - args.min_sample_weight) * reliability[keep_tr]
    pseudo_tr = (
        (~keep_tr)
        & (preds_t[tr] == knn_preds_tr)
        & (pmax_t[tr] >= args.pseudo_thresh)
        & (margins_t[tr] >= args.pseudo_margin)
    )
    soft_alpha = torch.zeros(y.numel(), dtype=torch.float32, device=device)
    if pseudo_tr.any():
        pseudo_idx = tr[pseudo_tr]
        target_labels[pseudo_idx] = preds_t[pseudo_idx]
        weights[pseudo_idx] = pmax_t[pseudo_idx] * margins_t[pseudo_idx].sqrt()
        # soft fusion: keep alpha of the original label instead of hard-replacing
        # (conservative: don't fully trust the pseudo-label).
        soft_alpha[pseudo_idx] = args.pseudo_soft_alpha
    relabel_count = 0
    if cl_relabel_info is not None:
        # Confident-Learning relabel: confident-wrong samples get the CL-predicted
        # class with a confidence weight (overrides the consensus-pseudo path above
        # for these samples). Recovers corrected data instead of discarding it.
        cl_pred_tr, cl_conf_tr, relabel_tr = cl_relabel_info
        if relabel_tr.any():
            ridx = tr[relabel_tr]
            target_labels[ridx] = cl_pred_tr[relabel_tr]
            weights[ridx] = cl_conf_tr[relabel_tr].clamp(0, 1)
            soft_alpha[ridx] = 0.0  # hard-use the corrected label
            relabel_count = int(relabel_tr.sum())
    if args.class_balance:
        # counter noise-induced class imbalance: scale weights by inverse class
        # frequency (of the final used set) to the cb_power.
        used_m = weights > 0
        cnt = torch.bincount(target_labels[used_m], minlength=num_classes).clamp_min(1).float()
        cls_w = (cnt.mean() / cnt).pow(args.cb_power)
        weights = weights * cls_w[target_labels]
        print(f"class-balance(power={args.cb_power}): weight ratio max/min = "
              f"{cls_w.max() / cls_w.min():.2f}")
    pseudo_count = int(pseudo_tr.sum())
    print(f"targets ready: {int((weights > 0).sum())} usable samples "
          f"({pseudo_count} consensus pseudo-labelled, {relabel_count} CL-relabelled)")
    return {
        "target_labels": target_labels.cpu(), "weights": weights.cpu(), "labels": labels,
        "class_names": class_names, "image_names": image_names,
        "teacher_head_state": {k: v.cpu() for k, v in teacher.state_dict().items()},
        "agree": agree.cpu(), "teacher_preds": preds_t.cpu(), "soft_alpha": soft_alpha.cpu(),
        "target_stats": {
            "kept": int(keep.sum()), "pseudo": pseudo_count,
            "mean_kept_weight": round(float(weights[keep].mean()), 6),
        },
    }


@torch.inference_mode()
def evaluate_images(model, loader, labels_gpu, agree_gpu, device, trusted_mask=None) -> dict:
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
    if trusted_mask is not None:
        bands["trusted"] = trusted_mask  # high-agreement & teacher-consensus subset
    return {k: round((correct & m).sum().item() / max(1, int(m.sum())), 4) for k, m in bands.items()}


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
    elif args.holdout_frac > 0:
        tr_idx, va_idx, ho_idx = stratified_three_way_split(
            labels, args.val_frac, args.holdout_frac, args.seed)
    else:
        tr_idx, va_idx = stratified_split(labels, args.val_frac, args.seed)
    if args.smoke:
        generator = torch.Generator().manual_seed(args.seed)
        tr_idx = tr_idx[torch.randperm(tr_idx.numel(), generator=generator)[:5000]].sort().values
        va_idx = va_idx[torch.randperm(va_idx.numel(), generator=generator)[:2000]].sort().values if va_idx.numel() else va_idx
        ho_idx = ho_idx[torch.randperm(ho_idx.numel(), generator=generator)[:1000]].sort().values if ho_idx.numel() else ho_idx
        args.epochs = 2

    # holdout samples are query-only, exactly like val: never in gallery/teacher
    prep = prepare_targets(args, device, tr_idx, torch.cat([va_idx, ho_idx]))
    class_names = prep["class_names"]
    num_classes = len(class_names)
    target_labels_all = prep["target_labels"].to(device)
    weights_all = prep["weights"].to(device)
    labels_all = prep["labels"].to(device)
    agree_all = prep["agree"].to(device)
    teacher_preds_all = prep["teacher_preds"].to(device)
    soft_alpha_all = prep["soft_alpha"].to(device)

    base = ImageFolder(args.train_dir)
    paths = [p for p, _ in base.samples]
    names_check = [Path(p).name for p in paths]
    assert names_check == prep["image_names"], "ImageFolder order mismatch with feature cache"

    usable = (weights_all[tr_idx.to(device)] > 0).cpu()
    tr_idx = tr_idx[usable]
    print(f"train images {tr_idx.numel()}  val images {va_idx.numel()}")

    model = build_lora_model(num_classes, args.lora_rank, args.lora_alpha, args.lora_dropout,
                             prep["teacher_head_state"], device, lora_blocks=args.lora_blocks,
                             lora_target=args.lora_target, img_size=args.img_size, peft=args.peft,
                             feat_fuse=args.feat_fuse, attn_pool=args.attn_pool)
    train_tf, eval_tf = build_finetune_transforms(model.backbone, args.crop_min_scale,
                                                  img_size=args.img_size, randaug=args.randaug)
    if args.random_erasing:
        # RandomErasing (Zhong 2020): occlude a random rectangle -> occlusion-robust.
        train_tf = transforms.Compose([train_tf, transforms.RandomErasing(p=0.25)])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable / 1e6:.2f}M")
    ema_model = None
    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for p in ema_model.parameters():
            p.requires_grad_(False)

    pin = not args.no_pin
    train_ds = IndexedImageDataset([paths[i] for i in tr_idx.tolist()], train_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              persistent_workers=args.num_workers > 0, drop_last=True)
    val_loader = None
    if va_idx.numel():
        val_ds = IndexedImageDataset([paths[i] for i in va_idx.tolist()], eval_tf)
        # Windows: workers are full processes (~1GB each); keep val pool small and
        # persistent so it is spawned once instead of every epoch.
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                                num_workers=2, pin_memory=pin, persistent_workers=True)

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
        model.load_state_dict(resume.get("raw_model", resume["model"]))
        if ema_model is not None:
            ema_model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
        print(f"resuming {args.resume} from epoch {start_epoch}")

    def checkpoint_payload(epoch: int, bands: dict | None = None) -> dict:
        eval_model = ema_model if ema_model is not None else model
        payload_extra = {"raw_model": model.state_dict()} if ema_model is not None else {}
        return {
            "model": eval_model.state_dict(), "class_names": class_names,
            **payload_extra,
            "epoch": epoch, "bands": bands, "args": vars(args),
            "target_stats": prep["target_stats"],
            "train_idx": tr_idx, "val_idx": va_idx, "holdout_idx": ho_idx,
            "model_name": MODEL_NAME, "single_model": True,
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
        }

    va_labels = labels_all[va_idx.to(device)] if va_idx.numel() else None
    va_agree = agree_all[va_idx.to(device)] if va_idx.numel() else None
    trusted_va = None
    if args.trusted_agree > 0 and va_idx.numel():
        # trustworthy mini-val: high kNN agreement AND teacher prediction == given
        # label -> labels here are very likely correct, a cleaner proxy for the
        # clean test set than the full noisy val.
        tp_va = teacher_preds_all[va_idx.to(device)]
        trusted_va = (va_agree >= args.trusted_agree) & (tp_va == va_labels)
    history = []
    best_mid = -1.0
    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot_loss, tot_seen = 0.0, 0
        # reliability curriculum: per-epoch threshold drops from a high quantile
        # (train only on the most reliable samples first) to 0 (use all by the end).
        thr_e = 0.0
        if args.curriculum:
            pos_w = weights_all[tr_idx_gpu]
            pos_w = pos_w[pos_w > 0]
            frac = args.curriculum_start * (1.0 - (epoch - 1) / max(1, args.epochs - 1))
            thr_e = float(torch.quantile(pos_w, frac)) if frac > 1e-6 else 0.0
        for images, idx in train_loader:
            images = images.to(device, non_blocking=True)
            gidx = tr_idx_gpu[idx.to(device)]
            tb = target_labels_all[gidx]
            wb = weights_all[gidx]
            if thr_e > 0:
                wb = wb * (wb >= thr_e).to(wb.dtype)
            opt.zero_grad(set_to_none=True)
            perm = None
            if args.mixup_alpha > 0 and args.manifold_mixup == 0:
                # Mixup (Zhang 2018): convex-combine inputs + targets -> a strong
                # label-noise smoother, orthogonal to the kNN denoising pipeline.
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1.0 - lam) * images[perm]
            elif args.cutmix > 0 and args.manifold_mixup == 0:
                # CutMix (Yun 2019): paste a random patch from another image; the
                # label mix is the patch area fraction.
                lam = float(np.random.beta(args.cutmix, args.cutmix))
                perm = torch.randperm(images.size(0), device=device)
                H, W = images.shape[-2:]
                r = math.sqrt(1.0 - lam)
                rh, rw = int(H * r), int(W * r)
                cy, cx = np.random.randint(H), np.random.randint(W)
                y1, y2 = max(0, cy - rh // 2), min(H, cy + rh // 2)
                x1, x2 = max(0, cx - rw // 2), min(W, cx + rw // 2)
                images[:, :, y1:y2, x1:x2] = images[perm][:, :, y1:y2, x1:x2]
                lam = 1.0 - ((y2 - y1) * (x2 - x1) / (H * W))
            with torch.autocast("cuda", dtype=torch.float16):
                if args.manifold_mixup > 0:
                    # Manifold Mixup (Verma 2019): mix the *feature* representation
                    # instead of the input -> flatter, noise-robust feature manifold.
                    lam = float(np.random.beta(args.manifold_mixup, args.manifold_mixup))
                    mperm = torch.randperm(images.size(0), device=device)
                    feat = model.extract_feat(images)
                    feat = lam * feat + (1.0 - lam) * feat[mperm]
                    logits = model.head(F.normalize(feat.float(), dim=-1))
                    loss_vec = lam * _loss_vec(logits, tb, args) + (1.0 - lam) * _loss_vec(logits, tb[mperm], args)
                    wb = lam * wb + (1.0 - lam) * wb[mperm]
                else:
                    logits = model(images)
                    if perm is not None:
                        loss_vec = lam * _loss_vec(logits, tb, args) + (1.0 - lam) * _loss_vec(logits, tb[perm], args)
                        wb = lam * wb + (1.0 - lam) * wb[perm]
                    elif args.robust_loss != "ce":
                        loss_vec = _loss_vec(logits, tb, args)
                    else:
                        loss_vec = F.cross_entropy(
                            logits, tb, reduction="none", label_smoothing=args.label_smoothing)
                        if args.pseudo_soft_alpha > 0:
                            ab = soft_alpha_all[gidx]
                            ce_orig = F.cross_entropy(
                                logits, labels_all[gidx], reduction="none", label_smoothing=args.label_smoothing)
                            loss_vec = (1.0 - ab) * loss_vec + ab * ce_orig
                loss = (loss_vec * wb).sum() / wb.sum().clamp_min(1e-6)
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
            bands = evaluate_images(ema_model if ema_model is not None else model,
                                    val_loader, va_labels, va_agree, device, trusted_mask=trusted_va)
            entry.update(bands)
            msg += "  " + "  ".join(f"{k}={v}" for k, v in bands.items())
            sel_key = "trusted" if "trusted" in bands else "mid_03_06"
            if bands[sel_key] > best_mid:
                best_mid = bands[sel_key]
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

    if ho_idx.numel():
        # free the persistent train/val worker processes before spawning new ones
        del train_loader, val_loader
        ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        ho_ds = IndexedImageDataset([paths[i] for i in ho_idx.tolist()], eval_tf)
        ho_loader = DataLoader(ho_ds, batch_size=args.batch_size * 2, shuffle=False,
                               num_workers=2, pin_memory=not args.no_pin)
        ho_labels = labels_all[ho_idx.to(device)]
        ho_agree = agree_all[ho_idx.to(device)]
        bands = evaluate_images(model, ho_loader, ho_labels, ho_agree, device)
        print(f"holdout eval (untouched {args.holdout_frac:.0%}, best epoch {ckpt.get('epoch')}): "
              + "  ".join(f"{k}={v}" for k, v in bands.items()), flush=True)
        history.append({"holdout_best_epoch": ckpt.get("epoch"),
                        **{f"holdout_{k}": v for k, v in bands.items()}})
        with (ckpt_dir / "history.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)
    print("done")


@torch.inference_mode()
def predict(args) -> None:
    device = torch.device("cuda")
    ckpt_path = choose_checkpoint(Path(args.work_dir) / "lora", args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    class_names = ckpt["class_names"]
    targs = ckpt.get("args", {})
    model = build_lora_model(len(class_names), targs.get("lora_rank", args.lora_rank),
                             targs.get("lora_alpha", args.lora_alpha), 0.0, None, device,
                             lora_blocks=targs.get("lora_blocks", args.lora_blocks),
                             lora_target=targs.get("lora_target", args.lora_target),
                             img_size=targs.get("img_size", args.img_size),
                             peft=targs.get("peft", args.peft),
                             feat_fuse=targs.get("feat_fuse", args.feat_fuse),
                             attn_pool=targs.get("attn_pool", args.attn_pool))
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {ckpt_path} (epoch {ckpt.get('epoch')})")

    _, eval_tf = build_finetune_transforms(model.backbone, targs.get("crop_min_scale", args.crop_min_scale),
                                           img_size=targs.get("img_size", args.img_size))
    test_dir = Path(args.test_dir)
    paths = sorted(str(p) for p in test_dir.iterdir()
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=str(config.TRAIN_DIR))
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    p.add_argument("--cache-dir", default=None,
                   help="feature cache dir; defaults to <work-dir>/cache. Set explicitly "
                        "when using an isolated --work-dir for tests.")
    p.add_argument("--output-csv", default=str(config.DEFAULT_LORA_OUTPUT_CSV))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--holdout-frac", type=float, default=0.0,
                   help="carve out an untouched stratified test partition before val "
                        "(0.1 -> 8:1:1 split with an unbiased final report)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--keep-ratio", type=float, default=0.75)
    p.add_argument("--denoise", default="knn",
                   choices=("knn", "cleanlab", "cleanlab_knn", "cleanlab_relabel",
                            "ot", "otknn", "labelprop", "divmix"),
                   help="clean-set selection: knn; cleanlab(*); ot=Sinkhorn-balance probs then CL; "
                        "otknn=ot∩knn; labelprop=feature-graph propagation; divmix=GMM loss-split co-refine")
    p.add_argument("--ot-iters", type=int, default=30, help="Sinkhorn iterations for ot/otknn")
    p.add_argument("--lp-alpha", type=float, default=0.8, help="label-propagation diffusion weight")
    p.add_argument("--lp-anchor-agree", type=float, default=0.6, help="kNN-agreement floor to be a propagation anchor")
    p.add_argument("--divmix-thresh", type=float, default=0.5, help="GMM clean-posterior threshold for divmix keep")
    p.add_argument("--class-balance", action="store_true", help="scale sample weights by inverse class frequency")
    p.add_argument("--cb-power", type=float, default=0.5, help="class-balance exponent (0=off..1=full inverse-freq)")
    p.add_argument("--cutmix", type=float, default=0.0, help="CutMix Beta(a,a) strength (0=off; mutually exclusive w/ mixup)")
    p.add_argument("--random-erasing", action="store_true", help="append RandomErasing(p=0.25) to train augmentation")
    p.add_argument("--pseudo-thresh", type=float, default=0.7)
    p.add_argument("--pseudo-margin", type=float, default=0.2)
    p.add_argument("--high-agreement-floor", type=float, default=0.7)
    p.add_argument("--min-sample-weight", type=float, default=0.2)
    p.add_argument("--agreement-power", type=float, default=1.0)
    p.add_argument("--confidence-power", type=float, default=0.5)
    p.add_argument("--margin-power", type=float, default=0.5)
    p.add_argument("--proto-power", type=float, default=0.0,
                   help="visual-class-prototype similarity factor in reliability (0=off)")
    p.add_argument("--aug-consist-power", type=float, default=0.0,
                   help="augmentation-consistency factor in reliability (0=off; needs aug-consist file)")
    p.add_argument("--aug-consist-path", default="outputs/cache/aug_consistency.pt")
    p.add_argument("--trusted-agree", type=float, default=0.0,
                   help="val mode: build a trusted-val (agree>=t & teacher==label) and select best by it (0=off)")
    p.add_argument("--reliability-mode", choices=("mul", "sum"), default="mul",
                   help="combine reliability signals by product (aggressive) or weighted avg (sum=conservative)")
    p.add_argument("--pseudo-soft-alpha", type=float, default=0.0,
                   help="soft pseudo-label: keep this fraction of the original label vs hard-replace (conservative)")
    p.add_argument("--teacher-epochs", type=int, default=20)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--robust-loss", choices=("ce", "gce", "sce", "apl"), default="ce",
                   help="gce=Generalized CE; sce=Symmetric CE; apl=Active-Passive (NCE+RCE)")
    p.add_argument("--gce-q", type=float, default=0.7)
    p.add_argument("--sce-alpha", type=float, default=0.1, help="SCE: CE weight")
    p.add_argument("--sce-beta", type=float, default=1.0, help="SCE: reverse-CE weight")
    p.add_argument("--apl-alpha", type=float, default=1.0, help="APL: normalized-CE weight")
    p.add_argument("--apl-beta", type=float, default=1.0, help="APL: reverse-CE weight")
    p.add_argument("--mixup-alpha", type=float, default=0.0,
                   help="Mixup Beta(a,a) strength (0=off; 0.2 mild, 0.4 stronger)")
    p.add_argument("--peft", choices=("lora", "dora"), default="lora",
                   help="dora = weight-decomposed LoRA (Liu 2024), strictly generalises lora")
    p.add_argument("--feat-fuse", type=int, default=0,
                   help="fuse CLS tokens of the last K transformer blocks (0=off=last layer only)")
    p.add_argument("--attn-pool", action="store_true",
                   help="pool last-block patch tokens with a learned attention query instead of CLS")
    p.add_argument("--manifold-mixup", type=float, default=0.0,
                   help="Manifold Mixup Beta(a,a) strength on features (0=off; mutually exclusive with --mixup-alpha)")
    p.add_argument("--curriculum", action="store_true",
                   help="reliability curriculum: train on most-reliable samples first, widen each epoch")
    p.add_argument("--curriculum-start", type=float, default=0.5,
                   help="curriculum: fraction of lowest-reliability samples excluded at epoch 1 (ramps to 0)")
    p.add_argument("--save-every", type=int, default=0,
                   help="also snapshot ep<N>.pt every N epochs (periodic anchors)")
    p.add_argument("--snapshot-after", type=int, default=0,
                   help="val mode: snapshot best_ep<N>.pt on each new val-best after epoch N")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-blocks", type=int, default=12, choices=(4, 6, 12))
    p.add_argument("--lora-target", choices=("attn", "attn_mlp"), default="attn",
                   help="attn_mlp also adapts mlp fc1/fc2 (~3x trainable params)")
    p.add_argument("--img-size", type=int, default=224,
                   help="input resolution; >224 resamples CLIP pos-embeds (e.g. 448 -> 196 patch tokens)")
    p.add_argument("--ema-decay", type=float, default=0.0,
                   help="EMA of trainable weights for eval/checkpoints (try 0.999)")
    p.add_argument("--randaug", action="store_true",
                   help="RandAugment(2, 7) instead of ColorJitter")
    p.add_argument("--crop-min-scale", type=float, default=0.8)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--checkpoint", choices=("best", "last", "full"), default="best")
    p.add_argument("--resume", default=None, help="resume from a training checkpoint using the same schedule")
    p.add_argument("--no-flip-tta", action="store_true")
    p.add_argument("--no-pin", action="store_true",
                   help="disable pinned host memory (low-RAM safety; pinned alloc can "
                        "deadlock when Windows is under memory pressure)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--predict", action="store_true")
    args = p.parse_args()
    if args.cache_dir is None:
        args.cache_dir = str(Path(args.work_dir) / "cache")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        predict(args)
    else:
        train(args)
