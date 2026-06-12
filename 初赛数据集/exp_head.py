"""Head-level ablation on cached CLIP features.

Loads the frozen-backbone feature cache produced by main.py, holds out a
stratified noisy validation split, and compares denoising strategies:

  A  ce            : CE + label smoothing on all data (stage-1 only)
  B  conf2stage    : main.py baseline (per-class top-k confidence, 2-stage)
  C  knn           : kNN label-agreement filter, 2-stage
  D  gmm           : GMM on per-sample loss -> clean-probability weights
  E  knn_soft      : kNN filter + self-training pseudo-labels for dropped part
  F  knn_soft_mix  : E + feature mixup + EMA

Run ablation:   python exp_head.py
Final training: python exp_head.py --final --method knn_soft_mix
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
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- utilities


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1, init_scale: float = 10.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * (x @ w.t())


def stratified_split(labels: torch.Tensor, val_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    lab = labels.numpy()
    train_parts, val_parts = [], []
    for cls in np.unique(lab):
        idx = np.nonzero(lab == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    return torch.from_numpy(train_idx).long(), torch.from_numpy(val_idx).long()


@torch.inference_mode()
def knn_agreement(
    query: torch.Tensor,
    query_labels: torch.Tensor,
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int,
    exclude_self: bool,
    chunk: int = 2048,
) -> torch.Tensor:
    """Weighted fraction of k nearest gallery neighbours sharing the query label.

    Features must be L2-normalised. exclude_self assumes query == gallery row-aligned.
    """
    n = query.size(0)
    agreement = torch.empty(n, dtype=torch.float32)
    g = gallery.t().contiguous()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sim = query[start:end] @ g  # (b, N) fp16
        if exclude_self:
            rows = torch.arange(start, end, device=sim.device)
            sim[torch.arange(end - start, device=sim.device), rows] = -2.0
        topv, topi = sim.topk(k, dim=1)
        same = (gallery_labels[topi] == query_labels[start:end, None]).float()
        w = topv.float().clamp_min(0)
        agreement[start:end] = (same * w).sum(1) / w.sum(1).clamp_min(1e-6)
    return agreement


def fit_gmm_1d(x: torch.Tensor, iters: int = 50) -> torch.Tensor:
    """2-component 1D GMM via EM. Returns posterior prob of the low-mean (clean) component."""
    x = x.float()
    mu = torch.tensor([x.quantile(0.25), x.quantile(0.75)], device=x.device)
    var = torch.full((2,), x.var().item() + 1e-6, device=x.device)
    pi = torch.tensor([0.5, 0.5], device=x.device)
    for _ in range(iters):
        log_p = (
            -0.5 * (x[:, None] - mu[None, :]) ** 2 / var[None, :]
            - 0.5 * var.log()[None, :]
            + pi.log()[None, :]
        )
        resp = log_p.softmax(dim=1)
        nk = resp.sum(0).clamp_min(1e-6)
        mu = (resp * x[:, None]).sum(0) / nk
        var = ((resp * (x[:, None] - mu[None, :]) ** 2).sum(0) / nk).clamp_min(1e-8)
        pi = nk / x.numel()
    clean_comp = int(mu.argmin())
    return resp[:, clean_comp]


# ---------------------------------------------------------------- training


def soft_ce(logits: torch.Tensor, target_probs: torch.Tensor, sample_w: torch.Tensor | None = None) -> torch.Tensor:
    loss = -(target_probs * logits.log_softmax(dim=1)).sum(1)
    if sample_w is not None:
        loss = loss * sample_w
        return loss.sum() / sample_w.sum().clamp_min(1e-6)
    return loss.mean()


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    t = torch.full((labels.size(0), num_classes), smoothing / (num_classes - 1), device=labels.device)
    t.scatter_(1, labels[:, None], 1.0 - smoothing)
    return t


def train_head(
    features: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    indices: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
    epochs: int = 15,
    batch_size: int = 8192,
    lr: float = 5e-3,
    weight_decay: float = 1e-2,
    smoothing: float = 0.1,
    dropout: float = 0.1,
    mixup_alpha: float = 0.0,
    ema_decay: float = 0.0,
    init_state: dict | None = None,
    device: torch.device = torch.device("cuda"),
    verbose: bool = False,
) -> CosineClassifier:
    """targets: hard labels (long) or soft distributions (float, (N, C))."""
    model = CosineClassifier(features.size(1), num_classes, dropout=dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    ema_model = None
    if ema_decay > 0:
        ema_model = CosineClassifier(features.size(1), num_classes, dropout=dropout).to(device)
        ema_model.load_state_dict(model.state_dict())
        for p in ema_model.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 0.1)
    soft_targets = targets.dim() == 2
    n = indices.numel()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = indices[torch.randperm(n, device=device)]
        tot_loss, tot_seen = 0.0, 0
        for start in range(0, n, batch_size):
            bidx = perm[start : start + batch_size]
            xb = features[bidx]
            if soft_targets:
                tb = targets[bidx]
            else:
                tb = smooth_one_hot(targets[bidx], num_classes, smoothing)
            wb = sample_weights[bidx] if sample_weights is not None else None
            if mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1.0 - lam)
                perm2 = torch.randperm(xb.size(0), device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm2]
                tb = lam * tb + (1.0 - lam) * tb[perm2]
                if wb is not None:
                    wb = lam * wb + (1.0 - lam) * wb[perm2]
            opt.zero_grad(set_to_none=True)
            loss = soft_ce(model(xb), tb, wb)
            loss.backward()
            opt.step()
            if ema_model is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema_model.parameters(), model.parameters()):
                        pe.mul_(ema_decay).add_(pm, alpha=1.0 - ema_decay)
            tot_loss += loss.item() * bidx.numel()
            tot_seen += bidx.numel()
        sched.step()
        if verbose:
            print(f"  epoch {epoch:02d}/{epochs} loss={tot_loss / max(1, tot_seen):.4f}")
    return ema_model if ema_model is not None else model


@torch.inference_mode()
def evaluate(model: CosineClassifier, feats: torch.Tensor, labels: torch.Tensor, batch: int = 16384) -> float:
    model.eval()
    correct = 0
    for start in range(0, feats.size(0), batch):
        logits = model(feats[start : start + batch])
        correct += (logits.argmax(1) == labels[start : start + batch]).sum().item()
    return correct / feats.size(0)


@torch.inference_mode()
def per_sample_stats(model: CosineClassifier, feats: torch.Tensor, labels: torch.Tensor, batch: int = 16384):
    """Returns (loss, prob_of_label, argmax, max_prob) per sample."""
    model.eval()
    losses, p_label, preds, p_max = [], [], [], []
    for start in range(0, feats.size(0), batch):
        logits = model(feats[start : start + batch])
        yb = labels[start : start + batch]
        logp = logits.log_softmax(1)
        losses.append(F.nll_loss(logp, yb, reduction="none"))
        probs = logp.exp()
        p_label.append(probs.gather(1, yb[:, None]).squeeze(1))
        mx = probs.max(1)
        preds.append(mx.indices)
        p_max.append(mx.values)
    return (torch.cat(losses), torch.cat(p_label), torch.cat(preds), torch.cat(p_max))


def per_class_topk_keep(score: torch.Tensor, labels: torch.Tensor, num_classes: int, keep_ratio: float) -> torch.Tensor:
    """Keep top keep_ratio of each class by score; returns bool mask."""
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for cls in range(num_classes):
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).squeeze(1)
        if cls_idx.numel() == 0:
            continue
        k = min(cls_idx.numel(), max(1, int(cls_idx.numel() * keep_ratio)))
        top = torch.topk(score[cls_idx], k=k, largest=True).indices
        keep[cls_idx[top]] = True
    return keep


# ---------------------------------------------------------------- methods


def make_soft_targets_with_pseudo(
    labels: torch.Tensor,
    keep_mask: torch.Tensor,
    teacher_preds: torch.Tensor,
    teacher_pmax: torch.Tensor,
    num_classes: int,
    smoothing: float,
    pseudo_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kept samples: smoothed original label, weight 1.
    Dropped samples with confident teacher: smoothed pseudo-label, weight = teacher confidence.
    Others: excluded (weight 0)."""
    n = labels.size(0)
    targets = smooth_one_hot(labels, num_classes, smoothing)
    weights = torch.zeros(n, device=labels.device)
    weights[keep_mask] = 1.0
    pseudo_mask = (~keep_mask) & (teacher_pmax >= pseudo_thresh)
    if pseudo_mask.any():
        targets[pseudo_mask] = smooth_one_hot(teacher_preds[pseudo_mask], num_classes, smoothing)
        weights[pseudo_mask] = teacher_pmax[pseudo_mask]
    used = torch.nonzero(weights > 0, as_tuple=False).squeeze(1)
    return targets, weights, used


def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(args.work_dir)
    cache = torch.load(work / "cache" / "train_features.pt", map_location="cpu")
    feats_all = cache["features"]  # fp16, L2-normalised
    labels_all = cache["labels"]
    class_names = cache["class_names"]
    num_classes = len(class_names)
    print(f"features {tuple(feats_all.shape)}  classes {num_classes}  device {device}")

    tr_idx, va_idx = stratified_split(labels_all, args.val_frac, args.seed)
    print(f"train {tr_idx.numel()}  val {va_idx.numel()} (noisy labels)")

    # GPU tensors
    f16 = feats_all.to(device)                       # fp16 for kNN
    f32 = feats_all.to(device, dtype=torch.float32)  # fp32 for heads
    y = labels_all.to(device)
    tr_idx_d = tr_idx.to(device)
    va_idx_d = va_idx.to(device)
    ftr, ytr = f32[tr_idx_d], y[tr_idx_d]
    fva, yva = f32[va_idx_d], y[va_idx_d]

    # kNN agreement (gallery = train split only, so val labels never influence training)
    t0 = time.time()
    agree_tr = knn_agreement(f16[tr_idx_d], ytr, f16[tr_idx_d], ytr, k=args.knn_k, exclude_self=True).to(device)
    agree_va = knn_agreement(f16[va_idx_d], yva, f16[tr_idx_d], ytr, k=args.knn_k, exclude_self=False).to(device)
    print(f"kNN agreement computed in {time.time() - t0:.1f}s")
    for thr in (0.1, 0.3, 0.5):
        frac = (agree_tr < thr).float().mean().item()
        print(f"  train samples with agreement<{thr}: {frac:.2%}")
    clean_va_mask = agree_va >= args.clean_val_thresh
    print(f"pseudo-clean val subset: {int(clean_va_mask.sum())}/{va_idx.numel()}")

    results: dict[str, dict] = {}

    def record(name: str, model: CosineClassifier, extra: dict | None = None):
        acc = evaluate(model, fva, yva)
        acc_clean = evaluate(model, fva[clean_va_mask], yva[clean_va_mask])
        results[name] = {"val_acc": round(acc, 4), "clean_val_acc": round(acc_clean, 4), **(extra or {})}
        print(f"[{name}] noisy-val acc={acc:.4f}  pseudo-clean-val acc={acc_clean:.4f}")

    all_tr = torch.arange(ytr.numel(), device=device)
    common = dict(device=device, epochs=args.epochs, batch_size=args.batch_size)

    # --- A: plain CE + LS
    seed_everything(args.seed)
    model_a = train_head(ftr, ytr, num_classes, all_tr, smoothing=0.1, **common)
    record("A_ce", model_a)

    # --- B: main.py 2-stage confidence top-k
    seed_everything(args.seed)
    stage1 = train_head(ftr, ytr, num_classes, all_tr, smoothing=0.1, epochs=5,
                        batch_size=args.batch_size, device=device)
    loss1, p_label1, preds1, pmax1 = per_sample_stats(stage1, ftr, ytr)
    keep_b = per_class_topk_keep(p_label1, ytr, num_classes, keep_ratio=0.8)
    idx_b = torch.nonzero(keep_b, as_tuple=False).squeeze(1)
    model_b = train_head(ftr, ytr, num_classes, idx_b, smoothing=0.05, lr=5e-3 * 0.6,
                         init_state=stage1.state_dict(), epochs=10,
                         batch_size=args.batch_size, device=device)
    record("B_conf2stage", model_b, {"kept": int(idx_b.numel())})

    # --- C: kNN agreement filter (per-class adaptive top-k), 2-stage
    seed_everything(args.seed)
    keep_c = per_class_topk_keep(agree_tr, ytr, num_classes, keep_ratio=args.keep_ratio)
    keep_c |= agree_tr >= 0.7  # never drop high-agreement samples
    idx_c = torch.nonzero(keep_c, as_tuple=False).squeeze(1)
    model_c = train_head(ftr, ytr, num_classes, idx_c, smoothing=0.1, **common)
    record("C_knn", model_c, {"kept": int(idx_c.numel())})

    # --- D: GMM on stage-1 loss -> clean prob as weights
    seed_everything(args.seed)
    clean_prob = fit_gmm_1d(loss1)
    w_d = clean_prob.clamp_min(0.0)
    model_d = train_head(ftr, ytr, num_classes, all_tr, sample_weights=w_d, smoothing=0.1, **common)
    record("D_gmm", model_d, {"mean_clean_prob": round(clean_prob.mean().item(), 3)})

    # --- E: kNN filter + pseudo-label dropped samples with model C as teacher
    seed_everything(args.seed)
    _, _, preds_c, pmax_c = per_sample_stats(model_c, ftr, ytr)
    targets_e, w_e, idx_e = make_soft_targets_with_pseudo(
        ytr, keep_c, preds_c, pmax_c, num_classes, smoothing=0.1, pseudo_thresh=args.pseudo_thresh)
    model_e = train_head(ftr, targets_e, num_classes, idx_e, sample_weights=w_e,
                         init_state=model_c.state_dict(), **common)
    record("E_knn_soft", model_e, {"used": int(idx_e.numel()),
                                   "pseudo": int(((w_e > 0) & ~keep_c).sum())})

    # --- F: E + mixup + EMA
    seed_everything(args.seed)
    model_f = train_head(ftr, targets_e, num_classes, idx_e, sample_weights=w_e,
                         init_state=model_c.state_dict(), mixup_alpha=args.mixup_alpha,
                         ema_decay=args.ema_decay, **common)
    record("F_knn_soft_mix", model_f, {"mixup": args.mixup_alpha, "ema": args.ema_decay})

    out = work / "experiments.json"
    payload = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "noise_scan": {f"agree_lt_{t}": round((agree_tr < t).float().mean().item(), 4) for t in (0.1, 0.3, 0.5)},
        "results": results,
    }
    with out.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print(f"\nsaved {out}")
    print(f"{'method':<16} {'noisy-val':>10} {'clean-val':>10}")
    for name, r in results.items():
        print(f"{name:<16} {r['val_acc']:>10.4f} {r['clean_val_acc']:>10.4f}")


# ---------------------------------------------------------------- final run


def run_final(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = Path(args.work_dir)
    cache = torch.load(work / "cache" / "train_features.pt", map_location="cpu")
    test_cache = torch.load(work / "cache" / "test_features.pt", map_location="cpu")
    feats_all, labels_all, class_names = cache["features"], cache["labels"], cache["class_names"]
    num_classes = len(class_names)
    f16 = feats_all.to(device)
    f32 = feats_all.to(device, dtype=torch.float32)
    y = labels_all.to(device)
    all_idx = torch.arange(y.numel(), device=device)
    print(f"final training on {y.numel()} samples, method={args.method}")

    seed_everything(args.seed)
    agree = knn_agreement(f16, y, f16, y, k=args.knn_k, exclude_self=True).to(device)
    keep = per_class_topk_keep(agree, y, num_classes, keep_ratio=args.keep_ratio)
    keep |= agree >= 0.7
    idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
    print(f"kNN filter keeps {idx_keep.numel()}/{y.numel()} ({idx_keep.numel() / y.numel():.2%})")

    teacher = train_head(f32, y, num_classes, idx_keep, smoothing=0.1, epochs=args.epochs,
                         batch_size=args.batch_size, device=device, verbose=True)

    if args.method == "knn":
        model = teacher
    else:
        _, _, preds_t, pmax_t = per_sample_stats(teacher, f32, y)
        targets, w, idx_used = make_soft_targets_with_pseudo(
            y, keep, preds_t, pmax_t, num_classes, smoothing=0.1, pseudo_thresh=args.pseudo_thresh)
        print(f"self-training uses {idx_used.numel()} samples "
              f"({int(((w > 0) & ~keep).sum())} pseudo-labelled)")
        mixup = args.mixup_alpha if args.method == "knn_soft_mix" else 0.0
        ema = args.ema_decay if args.method == "knn_soft_mix" else 0.0
        model = train_head(f32, targets, num_classes, idx_used, sample_weights=w,
                           init_state=teacher.state_dict(), mixup_alpha=mixup, ema_decay=ema,
                           epochs=args.epochs, batch_size=args.batch_size, device=device, verbose=True)

    artifact_dir = work / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_names": class_names,
                "method": args.method, "knn_k": args.knn_k, "keep_ratio": args.keep_ratio},
               artifact_dir / f"final_head_{args.method}.pt")

    test_feats = test_cache["features"].to(device, dtype=torch.float32)
    test_names = test_cache["names"]
    model.eval()
    preds: list[int] = []
    with torch.inference_mode():
        for start in range(0, test_feats.size(0), args.batch_size):
            preds.extend(model(test_feats[start : start + args.batch_size]).argmax(1).tolist())

    out_csv = Path(args.output_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for name, p in zip(test_names, preds):
            writer.writerow([name, class_names[p]])
    zip_path = out_csv.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)
    print(f"saved {out_csv} and {zip_path} ({len(preds)} predictions)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default=str(config.DEFAULT_WORK_DIR))
    p.add_argument("--output-csv", default=str(config.DEFAULT_OUTPUT_CSV))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--knn-k", type=int, default=16)
    p.add_argument("--keep-ratio", type=float, default=0.75)
    p.add_argument("--clean-val-thresh", type=float, default=0.6)
    p.add_argument("--pseudo-thresh", type=float, default=0.7)
    p.add_argument("--mixup-alpha", type=float, default=0.2)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--final", action="store_true")
    p.add_argument("--method", default="knn_soft_mix", choices=["knn", "knn_soft", "knn_soft_mix"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.final:
        run_final(args)
    else:
        run_ablation(args)
