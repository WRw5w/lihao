import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustft.robust_utils import (  # noqa: E402
    choose_checkpoint,
    consensus_pseudo_mask,
    reliability_weight,
    validate_disjoint_split,
)


class SplitValidationTests(unittest.TestCase):
    def test_rejects_overlapping_train_and_validation_indices(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_disjoint_split([0, 1, 2], [2, 3])

    def test_accepts_disjoint_nonempty_training_indices(self):
        validate_disjoint_split([0, 1], [2, 3])


class ReliabilityTests(unittest.TestCase):
    def test_reliability_weight_is_bounded_and_monotonic(self):
        low = reliability_weight(0.2, 0.4, 0.1)
        high = reliability_weight(0.8, 0.9, 0.7)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, low)

    def test_consensus_pseudo_label_requires_confidence_margin_and_knn_agreement(self):
        self.assertTrue(consensus_pseudo_mask(7, 7, 0.9, 0.5, 0.7, 0.2))
        self.assertFalse(consensus_pseudo_mask(7, 8, 0.9, 0.5, 0.7, 0.2))
        self.assertFalse(consensus_pseudo_mask(7, 7, 0.6, 0.5, 0.7, 0.2))
        self.assertFalse(consensus_pseudo_mask(7, 7, 0.9, 0.1, 0.7, 0.2))


class CheckpointSelectionTests(unittest.TestCase):
    def test_full_policy_does_not_silently_choose_best(self):
        root = Path("checkpoints")
        with patch.object(Path, "is_file", return_value=True):
            self.assertEqual(choose_checkpoint(root, "full"), root / "full.pt")

    def test_missing_requested_checkpoint_is_an_error(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaises(FileNotFoundError):
                choose_checkpoint(Path("checkpoints"), "best")


if __name__ == "__main__":
    unittest.main()
