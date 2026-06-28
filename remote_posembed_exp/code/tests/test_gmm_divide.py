"""Unit tests for the DivideMix per-epoch GMM clean/noisy re-selection.

Pure-logic tests (CPU, no GPU/images) for robustft.denoise.{fit_gmm_1d,
gmm_divide_select}. Runnable either via pytest or as a plain script:

    python tests/test_gmm_divide.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustft.denoise import fit_gmm_1d, gmm_divide_select


def _synthetic(seed: int = 0):
    """Bimodal losses: a low-loss clean cluster + a high-loss noisy cluster.

    Returns (losses, confs, preds, labels, idx) with three named index groups:
      clean        - low loss, pred == label, high conf.
      noisy_conf   - high loss, pred != label, high conf  (should be relabelled).
      noisy_unsure - high loss, pred != label, low  conf  (should be dropped).
    """
    g = torch.Generator().manual_seed(seed)
    n_clean, n_conf, n_unsure = 400, 120, 120
    n = n_clean + n_conf + n_unsure
    labels = torch.randint(0, 50, (n,), generator=g)

    losses = torch.empty(n)
    confs = torch.empty(n)
    preds = labels.clone()

    clean = slice(0, n_clean)
    conf = slice(n_clean, n_clean + n_conf)
    unsure = slice(n_clean + n_conf, n)

    losses[clean] = 0.05 + 0.05 * torch.rand(n_clean, generator=g)
    confs[clean] = 0.95 + 0.04 * torch.rand(n_clean, generator=g)

    losses[conf] = 3.0 + 0.5 * torch.rand(n_conf, generator=g)
    confs[conf] = 0.97 + 0.02 * torch.rand(n_conf, generator=g)
    preds[conf] = (labels[conf] + 1) % 50  # confidently disagree with the label

    losses[unsure] = 3.0 + 0.5 * torch.rand(n_unsure, generator=g)
    confs[unsure] = 0.40 + 0.10 * torch.rand(n_unsure, generator=g)
    preds[unsure] = (labels[unsure] + 2) % 50

    return losses, confs, preds, labels, {"clean": clean, "conf": conf, "unsure": unsure}


def test_fit_gmm_1d_separates_bimodal():
    losses, *_ = _synthetic()
    w_clean = fit_gmm_1d(losses)
    # low-loss samples get high clean posterior, high-loss samples get low.
    assert w_clean[:400].mean() > 0.9, w_clean[:400].mean()
    assert w_clean[400:].mean() < 0.1, w_clean[400:].mean()
    assert w_clean.min() >= 0.0 and w_clean.max() <= 1.0


def test_gmm_divide_routes_clean_noisy_dropped():
    losses, confs, preds, labels, idx = _synthetic()
    target, weights, w_clean = gmm_divide_select(
        losses, confs, preds, labels,
        clean_thresh=0.5, conf_gate=0.9, clean_weight=1.0, noisy_weight=0.5)

    c, q, u = idx["clean"], idx["conf"], idx["unsure"]
    # clean: kept with original label, weight 1.
    assert torch.all(weights[c] == 1.0)
    assert torch.all(target[c] == labels[c])
    # confident-noisy: relabelled to model prediction, weight = noisy_weight.
    assert torch.all(weights[q] == 0.5)
    assert torch.all(target[q] == preds[q])
    assert torch.all(target[q] != labels[q])
    # unsure-noisy: dropped this epoch.
    assert torch.all(weights[u] == 0.0)
    # labels of the dropped/clean samples are never silently corrupted.
    assert torch.all(target[u] == labels[u])
    assert weights.dtype == torch.float32


def test_gmm_divide_per_sample_clean_weight():
    losses, confs, preds, labels, idx = _synthetic()
    cw = torch.full_like(losses, 0.7)
    target, weights, _ = gmm_divide_select(
        losses, confs, preds, labels,
        clean_thresh=0.5, conf_gate=0.9, clean_weight=cw, noisy_weight=0.3)
    c = idx["clean"]
    assert torch.allclose(weights[c], torch.full((c.stop - c.start,), 0.7))


def test_gmm_divide_conf_gate_drops_all_noisy_when_high():
    losses, confs, preds, labels, idx = _synthetic()
    # gate above every confidence -> no noisy sample rejoins.
    _, weights, _ = gmm_divide_select(
        losses, confs, preds, labels, clean_thresh=0.5, conf_gate=1.01)
    assert torch.all(weights[idx["conf"]] == 0.0)
    assert torch.all(weights[idx["unsure"]] == 0.0)
    assert torch.all(weights[idx["clean"]] == 1.0)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
