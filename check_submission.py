"""Validate a submission csv/zip against the competition rules (thin entry).

Run: python check_submission.py
     python check_submission.py --csv submissions/pred_results_lora.csv --zip submissions/pred_results_lora.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from robustft.submission import validate_submission


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(config.DEFAULT_OUTPUT_CSV))
    p.add_argument("--zip", default=str(config.DEFAULT_OUTPUT_ZIP))
    p.add_argument("--test-dir", default=str(config.TEST_DIR))
    p.add_argument("--num-classes", type=int, default=config.NUM_CLASSES)
    args = p.parse_args()

    ok = validate_submission(Path(args.csv), Path(args.zip), Path(args.test_dir), args.num_classes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
