"""CLI argument-validation tests for finetune_lora.parse_args.

Guards the periodic-refresh divisors (--ssl-every / --divide-every used as modulo
divisors) and the --dynamic-divide preconditions, so bad flags fail fast with a
clear parser error instead of crashing mid-training (e.g. ZeroDivisionError).

Runnable via pytest or as a plain script:  python tests/test_cli_validation.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import finetune_lora


def _parse(argv):
    with mock.patch.object(sys, "argv", ["finetune_lora.py", *argv]):
        return finetune_lora.parse_args()


def _expect_exit(argv):
    try:
        _parse(argv)
    except SystemExit as e:
        assert e.code != 0, f"expected non-zero exit for {argv}, got {e.code}"
        return
    raise AssertionError(f"expected SystemExit for {argv}, but parse_args succeeded")


def test_ssl_every_zero_rejected():
    _expect_exit(["--ssl-every", "0"])


def test_divide_every_zero_rejected():
    _expect_exit(["--divide-every", "0"])


def test_dynamic_divide_requires_ema():
    # --ema-decay defaults to 0.0, so dynamic-divide alone must be rejected.
    _expect_exit(["--dynamic-divide"])


def test_dynamic_divide_excludes_ssl_recover():
    _expect_exit(["--dynamic-divide", "--ema-decay", "0.999", "--ssl-recover"])


def test_valid_dynamic_divide_parses():
    args = _parse(["--dynamic-divide", "--ema-decay", "0.999"])
    assert args.dynamic_divide is True
    assert args.ema_decay == 0.999


def test_defaults_parse():
    args = _parse([])
    assert args.ssl_every > 0 and args.divide_every > 0
    assert args.dynamic_divide is False


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
