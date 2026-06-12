"""Head-level ablation on cached CLIP features (thin entry over robustft).

Loads the frozen-backbone feature cache produced by main.py, holds out a
stratified noisy validation split, and compares denoising strategies:

  A  ce            : CE + label smoothing on all data (stage-1 only)
  B  conf2stage    : main.py baseline (per-class top-k confidence, 2-stage)
  C  knn           : kNN label-agreement filter, 2-stage
  D  gmm           : GMM on per-sample loss -> clean-probability weights
  E  knn_soft      : kNN filter + self-training pseudo-labels for dropped part
  F  knn_soft_mix  : E + feature mixup + EMA

Run ablation:   python exp_head.py
Final training: python exp_head.py --final --method knn_soft
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import config
from robustft.denoise import (  # noqa: F401  (re-exported for exp_round2 et al.)
    fit_gmm_1d,
    knn_agreement,
    make_soft_targets_with_pseudo,
    per_class_topk_keep,
)
from robustft.engine import (  # noqa: F401
    banded_eval,
    evaluate,
    per_sample_stats,
    seed_everything,
    smooth_one_hot,
    soft_ce,
    stratified_split,
    train_head,
)
from robustft.models import CosineClassifier  # noqa: F401
from robustft.submission import save_predictions, zip_submission


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
        json.dump(payload, fp, ensure_ascii=False, indent=2, default=str)
    print(f"\nsaved {out}")
    print(f"{'method':<16} {'noisy-val':>10} {'clean-val':>10}")
    for name, r in results.items():
        print(f"{name:<16} {r['val_acc']:>10.4f} {r['clean_val_acc']:>10.4f}")


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
    if not out_csv.is_absolute():
        out_csv = Path.cwd() / out_csv
    save_predictions(out_csv, test_names, preds, class_names)
    zip_path = zip_submission(out_csv)
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
