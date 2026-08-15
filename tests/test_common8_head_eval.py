import unittest

from tools.eval_common8_orgslot_heads import (
    add_volume_ratios,
    parse_threshold_sweep,
    select_thresholds,
)


class Common8HeadEvaluationTests(unittest.TestCase):
    def test_threshold_sweep_is_sorted_unique_and_validated(self):
        self.assertEqual(parse_threshold_sweep("0.5,0.25,0.5"), (0.25, 0.5))
        with self.assertRaises(ValueError):
            parse_threshold_sweep("0,0.5")

    def test_threshold_selection_uses_validation_dice_and_stable_tie_break(self):
        metrics = {
            "sweep_0p25_post": {
                "1": {"case_dice_presence_mean": 0.8},
            },
            "sweep_0p5_post": {
                "1": {"case_dice_presence_mean": 0.8},
            },
            "sweep_0p75_post": {
                "1": {"case_dice_presence_mean": 0.7},
            },
        }
        self.assertEqual(select_thresholds(metrics, (1,), (0.25, 0.5, 0.75)), {1: 0.5})

    def test_volume_ratio_is_reported_globally_and_per_case(self):
        metrics = {
            "binary_post": {
                "1": {"prediction_voxels": 12, "target_voxels": 8},
                "foreground_mean_global_dice": 0.5,
            }
        }
        rows = [{"prediction_voxels": 3, "target_voxels": 2}]
        add_volume_ratios(metrics, rows)
        self.assertEqual(
            metrics["binary_post"]["1"]["prediction_to_target_volume_ratio"],
            1.5,
        )
        self.assertEqual(rows[0]["prediction_to_target_volume_ratio"], 1.5)


if __name__ == "__main__":
    unittest.main()
