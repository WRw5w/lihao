"""Smoke + numeric regression tests. Plain script, no pytest dependency.

Run from the project root: python tests/test_regression.py
  --skip-gpu  : only run the synthetic-tensor unit checks (no cache/GPU needed)

The A_ce regression retrains the plain CE head on the cached features with the
exact seed/order used by the original ablation; the noisy-val accuracy must
reproduce 0.6264 (tolerance 0.003). This is the behaviour-preservation gate
for any refactor of robustft.engine / robustft.denoise.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from robustft.denoise import fit_gmm_1d, knn_agreement  # noqa: E402
from robustft.engine import evaluate, seed_everything, smooth_one_hot, stratified_split, train_head  # noqa: E402
from robustft.submission import save_predictions, validate_submission, zip_submission  # noqa: E402

A_CE_EXPECTED = 0.6264
A_CE_TOL = 0.003


def check(name: str, cond: bool) -> bool:
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    return cond


def unit_tests() -> bool:
    ok = True

    # smooth_one_hot: rows sum to 1, true class gets 1-smoothing
    t = smooth_one_hot(torch.tensor([0, 2]), num_classes=4, smoothing=0.1)
    ok &= check("smooth_one_hot rows sum to 1", torch.allclose(t.sum(1), torch.ones(2)))
    ok &= check("smooth_one_hot true-class mass", torch.allclose(t[0, 0], torch.tensor(0.9)))

    # knn_agreement: two tight clusters with consistent labels -> agreement ~1
    g = torch.nn.functional.normalize(
        torch.cat([torch.randn(50, 8) * 0.01 + torch.tensor([1.0] + [0.0] * 7),
                   torch.randn(50, 8) * 0.01 + torch.tensor([0.0, 1.0] + [0.0] * 6)]), dim=1)
    lab = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    agree = knn_agreement(g, lab, g, lab, k=5, exclude_self=True)
    ok &= check("knn_agreement consistent clusters ~1", bool((agree > 0.95).all()))
    flipped = lab.clone()
    flipped[0] = 1  # one wrong label inside cluster 0
    agree2 = knn_agreement(g, flipped, g, flipped, k=5, exclude_self=True)
    ok &= check("knn_agreement flags the flipped label", bool(agree2[0] < 0.1))

    # fit_gmm_1d: separable mixture -> low-mean component gets high posterior
    x = torch.cat([torch.randn(500) * 0.1 + 0.5, torch.randn(500) * 0.1 + 3.0])
    p_clean = fit_gmm_1d(x)
    ok &= check("fit_gmm_1d separates components", bool(p_clean[:500].mean() > 0.9 and p_clean[500:].mean() < 0.1))

    # submission roundtrip in a temp dir
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_dir = td_path / "test"
        test_dir.mkdir()
        names = [f"img_{i}.jpg" for i in range(5)]
        for n in names:
            (test_dir / n).touch()
        csv_path = save_predictions(td_path / "pred.csv", names, [0, 1, 2, 3, 4], [f"{i:04d}" for i in range(5)])
        zip_path = zip_submission(csv_path)
        ok &= check("submission roundtrip validates", validate_submission(csv_path, zip_path, test_dir, 5))

    return ok


def a_ce_regression() -> bool:
    cache_path = Path(config.DEFAULT_WORK_DIR) / "cache" / "train_features.pt"
    if not cache_path.exists():
        print(f"SKIP A_ce regression: cache not found at {cache_path}")
        return True
    if not torch.cuda.is_available():
        print("SKIP A_ce regression: no CUDA")
        return True
    device = torch.device("cuda")
    cache = torch.load(cache_path, map_location="cpu")
    feats, labels = cache["features"], cache["labels"]
    num_classes = len(cache["class_names"])
    tr_idx, va_idx = stratified_split(labels, 0.1, 42)
    f32 = feats.to(device, dtype=torch.float32)
    y = labels.to(device)
    ftr, ytr = f32[tr_idx.to(device)], y[tr_idx.to(device)]
    fva, yva = f32[va_idx.to(device)], y[va_idx.to(device)]
    seed_everything(42)
    model = train_head(ftr, ytr, num_classes, torch.arange(ytr.numel(), device=device),
                       smoothing=0.1, epochs=15, batch_size=8192, device=device)
    acc = evaluate(model, fva, yva)
    print(f"A_ce noisy-val acc = {acc:.4f} (expected {A_CE_EXPECTED} +/- {A_CE_TOL})")
    return check("A_ce numeric regression", abs(acc - A_CE_EXPECTED) <= A_CE_TOL)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-gpu", action="store_true")
    args = p.parse_args()

    ok = unit_tests()
    if not args.skip_gpu:
        ok &= a_ce_regression()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
