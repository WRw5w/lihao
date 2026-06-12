import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustft.engine import stratified_three_way_split  # noqa: E402


class ThreeWaySplitTests(unittest.TestCase):
    def test_partitions_are_disjoint_and_cover_everything(self):
        labels = torch.arange(20).repeat_interleave(50)
        tr, va, ho = stratified_three_way_split(labels, 0.1, 0.1, 42)
        all_idx = torch.cat([tr, va, ho]).sort().values
        self.assertTrue(torch.equal(all_idx, torch.arange(labels.numel())))
        self.assertEqual(set(tr.tolist()) & set(va.tolist()), set())
        self.assertEqual(set(tr.tolist()) & set(ho.tolist()), set())
        self.assertEqual(set(va.tolist()) & set(ho.tolist()), set())

    def test_ratios_match_8_1_1(self):
        labels = torch.arange(10).repeat_interleave(100)
        tr, va, ho = stratified_three_way_split(labels, 0.1, 0.1, 0)
        self.assertEqual(ho.numel(), 100)
        self.assertEqual(va.numel(), 100)
        self.assertEqual(tr.numel(), 800)

    def test_deterministic_for_same_seed(self):
        labels = torch.randint(0, 5, (500,))
        first = stratified_three_way_split(labels, 0.1, 0.1, 7)
        second = stratified_three_way_split(labels, 0.1, 0.1, 7)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
