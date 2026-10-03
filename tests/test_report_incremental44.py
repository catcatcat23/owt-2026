import unittest
from tools.report_incremental44 import group_scores


class TestGroups(unittest.TestCase):
    def test_background_does_not_inflate_means(self):
        metrics = {str(c): {"case_dice_presence_mean": .8 if c <= 4 else .6} for c in range(1, 9)}
        metrics["0"] = {"case_dice_presence_mean": 1.}
        out = group_scores(metrics, "offline")
        self.assertAlmostEqual(out["old"], 80)
        self.assertAlmostEqual(out["new"], 60)
        self.assertAlmostEqual(out["all"], 70)

    def test_unseen_classes_are_not_fabricated_scores(self):
        metrics = {str(c): {"case_dice_presence_mean": .8} for c in range(1, 5)}
        out = group_scores(metrics, "stage1")
        self.assertIsNone(out["new"])
        self.assertIsNone(out["all"])
        with self.assertRaises(ValueError):
            group_scores(metrics, "stage2")
