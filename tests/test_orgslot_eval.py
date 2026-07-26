import math
import unittest

import torch

from eval_orgslot import (
    aggregate_slices_by_case,
    binary_dice,
    forgetting,
    numeric_slice_index,
    old_new_all_summary,
)


class EvaluationTests(unittest.TestCase):
    def test_binary_dice_and_empty_policy(self):
        prediction = torch.tensor([1, 1, 0, 0])
        target = torch.tensor([1, 0, 1, 0])
        self.assertAlmostEqual(float(binary_dice(prediction, target)), 0.5)
        empty = torch.zeros(4)
        self.assertTrue(math.isnan(float(binary_dice(empty, empty))))
        self.assertEqual(float(binary_dice(empty, empty, empty_policy="one")), 1)

    def test_numeric_slice_sort_and_case_aggregation(self):
        self.assertLess(numeric_slice_index("scan_2.npy"), numeric_slice_index("scan_10.npy"))
        records = [
            {"case_id": "c", "slice_index": 10, "prediction": torch.tensor(10), "target": torch.tensor(10)},
            {"case_id": "c", "slice_index": 2, "prediction": torch.tensor(2), "target": torch.tensor(2)},
        ]
        volume = aggregate_slices_by_case(records)["c"]
        self.assertEqual(volume["slice_indices"], [2, 10])
        self.assertEqual(volume["prediction"].tolist(), [2, 10])

    def test_old_new_all_and_forgetting(self):
        summary = old_new_all_summary(
            {"old1": 0.8, "old2": 0.6, "new": 0.5},
            ["old1", "old2"], ["new"],
        )
        self.assertAlmostEqual(summary["old_dsc"], 0.7)
        self.assertAlmostEqual(summary["new_dsc"], 0.5)
        self.assertAlmostEqual(summary["all_dsc"], 1.9 / 3)
        self.assertAlmostEqual(forgetting(0.7, 0.65), 0.05)


if __name__ == "__main__":
    unittest.main()
