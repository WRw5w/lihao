"""Validate pred_results.csv / pred_results.zip against the competition rules.

Checks:
  1. zip contains exactly pred_results.csv
  2. every row is "<filename>,<4-digit class id>"
  3. filenames exactly match the test directory (case-sensitive, no missing/extra)
  4. class ids are within the known class set
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import config


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(config.DEFAULT_OUTPUT_CSV))
    p.add_argument("--zip", default=str(config.DEFAULT_OUTPUT_ZIP))
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--num-classes", type=int, default=500)
    args = p.parse_args()

    ok = True

    zpath = Path(args.zip)
    if zpath.exists():
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            if names != ["pred_results.csv"]:
                print(f"FAIL zip contents: {names} (expected exactly ['pred_results.csv'])")
                ok = False
            else:
                print("OK   zip contains exactly pred_results.csv")
    else:
        print(f"WARN zip not found: {zpath}")

    rows: list[tuple[str, str]] = []
    with Path(args.csv).open(newline="", encoding="utf-8") as fp:
        for lineno, row in enumerate(csv.reader(fp), 1):
            if len(row) != 2:
                print(f"FAIL line {lineno}: expected 2 fields, got {len(row)}: {row}")
                ok = False
                continue
            rows.append((row[0], row[1].strip()))

    valid_ids = {f"{i:04d}" for i in range(args.num_classes)}
    bad_ids = [(n, c) for n, c in rows if c not in valid_ids]
    if bad_ids:
        print(f"FAIL {len(bad_ids)} rows with invalid class id, e.g. {bad_ids[:3]}")
        ok = False
    else:
        print(f"OK   all {len(rows)} class ids are 4-digit and within 0000..{args.num_classes - 1:04d}")

    test_files = {f.name for f in Path(args.test_dir).iterdir() if f.is_file()}
    pred_files = [n for n, _ in rows]
    pred_set = set(pred_files)
    if len(pred_files) != len(pred_set):
        print(f"FAIL duplicate filenames in csv: {len(pred_files) - len(pred_set)}")
        ok = False
    missing = test_files - pred_set
    extra = pred_set - test_files
    if missing:
        print(f"FAIL {len(missing)} test images missing from csv, e.g. {sorted(missing)[:3]}")
        ok = False
    if extra:
        print(f"FAIL {len(extra)} csv rows not in test dir, e.g. {sorted(extra)[:3]}")
        ok = False
    if not missing and not extra:
        print(f"OK   filenames exactly match test dir ({len(test_files)} files)")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
