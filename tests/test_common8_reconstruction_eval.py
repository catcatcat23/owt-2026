import unittest

import numpy as np

from tools.eval_common8_orgslot_reconstruction_threshold import (
    dice_from_counts,
    iou_from_counts,
    new_counter,
    postprocess_slice,
    summarize,
    update_counter,
)


class Common8ReconstructionEvaluationTests(unittest.TestCase):
    def test_overlap_metrics_handle_empty_and_nonempty(self):
        self.assertEqual(dice_from_counts(0, 0, 0), 1.0)
        self.assertEqual(iou_from_counts(0, 0, 0), 1.0)
        self.assertAlmostEqual(dice_from_counts(2, 3, 4), 4 / 7)
        self.assertAlmostEqual(iou_from_counts(2, 3, 4), 2 / 5)

    def test_postprocess_removes_small_components(self):
        prediction = np.zeros((16, 16), dtype=bool)
        prediction[1, 1] = True
        prediction[8:12, 8:12] = True
        cleaned = postprocess_slice(prediction, min_size=4, opening_radius=0)
        self.assertFalse(cleaned[1, 1])
        self.assertEqual(int(cleaned.sum()), 16)

    def test_summary_uses_gt_present_cases_for_primary_case_macro(self):
        class_names = {0: "background", 1: "spleen"}
        counters = {"indirect_post": {1: {}}}
        present = new_counter()
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [0, 0]], dtype=bool)
        update_counter(present, prediction, target)
        counters["indirect_post"][1]["case_present"] = present
        absent = new_counter()
        update_counter(
            absent,
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=bool),
        )
        counters["indirect_post"][1]["case_absent"] = absent

        metrics, rows, records = summarize(
            counters, (1,), class_names, "test method"
        )
        spleen = metrics["indirect_post"]["1"]
        self.assertAlmostEqual(spleen["case_dice_presence_mean"], 2 / 3)
        self.assertAlmostEqual(
            spleen["case_dice_all_mean_empty_empty_one"], (2 / 3 + 1) / 2
        )
        self.assertEqual(spleen["gt_present_case_count"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["method"], "test method")


if __name__ == "__main__":
    unittest.main()
