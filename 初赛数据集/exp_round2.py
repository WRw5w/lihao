"""Round-2 ablation: tune the kNN-filter + self-training recipe (method E).

Adds banded validation: val accuracy inside kNN-agreement bands. The mid band
(0.3-0.6) is the most discriminative — labels there are mostly correct but the
samples are hard. The low band (<0.3) is dominated by wrong labels, so high
accuracy there mostly means noise-fitting.

Run: python exp_round2.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from exp_head import (
    CosineClassifier,
    evaluate,
    knn_agreement,
    make_soft_targets_with_pseudo,
    per_class_topk_keep,
    per_sample_stats,
    seed_everything,
    stratified_split,
    train_head,
)

SEED = 42
VAL_FRAC = 0.1
KNN_K = 16
EPOCHS = 20
BATCH = 8192
WORK = Path("outputs")


def banded_eval(model: CosineClassifier, fva, yva, agree_va) -> dict:
    bands = {
        "noisy_all": torch.ones_like(yva, dtype=torch.bool),
        "low_lt03": agree_va < 0.3,
        "mid_03_06": (agree_va >= 0.3) & (agree_va < 0.6),
        "high_ge06": agree_va >= 0.6,
    }
    return {name: round(evaluate(model, fva[m], yva[m]), 4) for name, m in bands.items() if int(m.sum()) > 0}


def main() -> None:
    device = torch.device("cuda")
    cache = torch.load(WORK / "cache" / "train_features.pt", map_location="cpu")
    feats, labels, class_names = cache["features"], cache["labels"], cache["class_names"]
    num_classes = len(class_names)

    tr_idx, va_idx = stratified_split(labels, VAL_FRAC, SEED)
    f16 = feats.to(device)
    f32 = feats.to(device, dtype=torch.float32)
    y = labels.to(device)
    ftr, ytr = f32[tr_idx.to(device)], y[tr_idx.to(device)]
    fva, yva = f32[va_idx.to(device)], y[va_idx.to(device)]

    t0 = time.time()
    agree_tr = knn_agreement(f16[tr_idx.to(device)], ytr, f16[tr_idx.to(device)], ytr, k=KNN_K, exclude_self=True).to(device)
    agree_va = knn_agreement(f16[va_idx.to(device)], yva, f16[tr_idx.to(device)], ytr, k=KNN_K, exclude_self=False).to(device)
    print(f"kNN agreement in {time.time() - t0:.1f}s")
    for name, m in (("low_lt03", agree_va < 0.3), ("mid_03_06", (agree_va >= 0.3) & (agree_va < 0.6)), ("high_ge06", agree_va >= 0.6)):
        print(f"  val band {name}: {int(m.sum())} samples")

    results: dict[str, dict] = {}

    def record(name: str, model: CosineClassifier, extra: dict | None = None):
        r = banded_eval(model, fva, yva, agree_va)
        results[name] = {**r, **(extra or {})}
        print(f"[{name}] " + "  ".join(f"{k}={v}" for k, v in r.items()) + (f"  {extra}" if extra else ""))

    common = dict(device=device, epochs=EPOCHS, batch_size=BATCH)
    all_tr = torch.arange(ytr.numel(), device=device)

    # reference: plain CE
    seed_everything(SEED)
    model_a = train_head(ftr, ytr, num_classes, all_tr, smoothing=0.1, **common)
    record("ref_ce", model_a)

    # stage-1 stats for fused score
    _, p_label_a, _, _ = per_sample_stats(model_a, ftr, ytr)

    def knn_keep(keep_ratio: float, score: torch.Tensor) -> torch.Tensor:
        keep = per_class_topk_keep(score, ytr, num_classes, keep_ratio=keep_ratio)
        keep |= agree_tr >= 0.7
        return keep

    def self_train(keep: torch.Tensor, teacher: CosineClassifier, pseudo_thresh: float,
                   rounds: int, tag: str) -> CosineClassifier:
        model = teacher
        for r in range(rounds):
            _, _, preds_t, pmax_t = per_sample_stats(model, ftr, ytr)
            targets, w, idx_used = make_soft_targets_with_pseudo(
                ytr, keep, preds_t, pmax_t, num_classes, smoothing=0.1, pseudo_thresh=pseudo_thresh)
            seed_everything(SEED + r)
            model = train_head(ftr, targets, num_classes, idx_used, sample_weights=w,
                               init_state=model.state_dict(), **common)
            n_pseudo = int(((w > 0) & ~keep).sum())
            record(f"{tag}_r{r + 1}", model, {"used": int(idx_used.numel()), "pseudo": n_pseudo})
        return model

    # keep-ratio sweep with kNN score, 1 round self-training
    for keep_ratio in (0.5, 0.6, 0.75):
        seed_everything(SEED)
        keep = knn_keep(keep_ratio, agree_tr)
        idx_keep = torch.nonzero(keep, as_tuple=False).squeeze(1)
        teacher = train_head(ftr, ytr, num_classes, idx_keep, smoothing=0.1, **common)
        record(f"knn{int(keep_ratio * 100)}_teacher", teacher, {"kept": int(idx_keep.numel())})
        self_train(keep, teacher, pseudo_thresh=0.7, rounds=1, tag=f"knn{int(keep_ratio * 100)}_soft")

    # fused score (agreement + model confidence), keep 0.6
    seed_everything(SEED)
    fused = 0.5 * agree_tr + 0.5 * p_label_a
    keep_f = knn_keep(0.6, fused)
    idx_f = torch.nonzero(keep_f, as_tuple=False).squeeze(1)
    teacher_f = train_head(ftr, ytr, num_classes, idx_f, smoothing=0.1, **common)
    record("fused60_teacher", teacher_f, {"kept": int(idx_f.numel())})
    best_fused = self_train(keep_f, teacher_f, pseudo_thresh=0.7, rounds=3, tag="fused60_soft")

    # pseudo threshold variant on fused60
    seed_everything(SEED)
    self_train(keep_f, teacher_f, pseudo_thresh=0.5, rounds=1, tag="fused60_soft_p50")

    out = WORK / "experiments_round2.json"
    with out.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\nsaved {out}")
    header = f"{'config':<22} {'noisy':>7} {'low':>7} {'mid':>7} {'high':>7}"
    print(header)
    for name, r in results.items():
        print(f"{name:<22} {r.get('noisy_all', 0):>7.4f} {r.get('low_lt03', 0):>7.4f} "
              f"{r.get('mid_03_06', 0):>7.4f} {r.get('high_ge06', 0):>7.4f}")


if __name__ == "__main__":
    main()
