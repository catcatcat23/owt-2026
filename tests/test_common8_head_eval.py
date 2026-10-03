import unittest
import numpy as np

from tools.eval_common8_orgslot_heads import (
    add_volume_ratios,
    parse_threshold_sweep,
    select_thresholds,
    stage_background,
)
from tools.eval_common8_orgslot_reconstruction_threshold import parse_class_configuration


class Common8HeadEvaluationTests(unittest.TestCase):
    def test_stage1_has_exactly_five_slots_and_offline_stays_nine(self):
        slots, names = parse_class_configuration("configs/orgslot/common8_incremental44.json", "stage1")
        self.assertEqual([s["raw_class_id"] for s in slots], [0, 1, 2, 3, 4])
        self.assertEqual(len(parse_class_configuration("configs/orgslot/common8_offline.json")[0]), 9)

    def test_stage1_background_includes_unseen_organs(self):
        labels = np.array([[0, 1, 5, 8]])
        predictions = {c: labels == c for c in (1, 2, 3, 4)}
        pred, target = stage_background(labels, predictions, (1, 2, 3, 4))
        np.testing.assert_array_equal(target, [[True, False, True, True]])
        np.testing.assert_array_equal(pred, target)

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
