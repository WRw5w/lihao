"""Unit tests for the Early-Learning Regularization term (robustft.engine.elr_regularizer).

Pure-logic, CPU. Runnable via pytest or as a script: python tests/test_elr.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robustft.engine import elr_regularizer


def test_orthogonal_pred_target_is_zero():
    # <p, t> = 0  ->  lam * log(1 - 0) = 0
    p = torch.tensor([[1.0, 0.0, 0.0]])
    t = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(elr_regularizer(p, t, 3.0), torch.tensor(0.0), atol=1e-6)


def test_aligned_pred_target_is_negative():
    # <p, t> -> 1  ->  log(small) -> large negative (term rewards alignment)
    p = torch.tensor([[1.0, 0.0, 0.0]])
    t = torch.tensor([[1.0, 0.0, 0.0]])
    val = elr_regularizer(p, t, 3.0)
    assert val.item() < -10.0, val.item()  # 3*log(1e-4) ~ -27.6


def test_monotonic_decreasing_in_alignment():
    t = torch.tensor([[1.0, 0.0, 0.0]])
    low = elr_regularizer(torch.tensor([[0.4, 0.3, 0.3]]), t, 3.0)
    high = elr_regularizer(torch.tensor([[0.9, 0.05, 0.05]]), t, 3.0)
    assert high.item() < low.item()  # more aligned -> smaller (more negative) term


def test_gradient_pushes_prediction_toward_target():
    # minimising the term should increase the logit of the target class
    logits = torch.zeros(1, 5, requires_grad=True)
    t = F.one_hot(torch.tensor([2]), 5).float()
    reg = elr_regularizer(logits.softmax(1), t, 3.0)
    reg.backward()
    g = logits.grad[0]
    # gradient-descent step (logit -= lr*g) must raise class 2 most -> g[2] is the min
    assert g[2].item() == g.min().item()
    assert g[2].item() < 0.0


def test_batch_mean_and_scalar():
    p = torch.tensor([[0.7, 0.3], [0.2, 0.8]])
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    out = elr_regularizer(p, t, 2.0)
    assert out.dim() == 0
    expected = 2.0 * (torch.log(torch.tensor(1 - 0.7)) + torch.log(torch.tensor(1 - 0.8))) / 2
    assert torch.allclose(out, expected, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1; print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
