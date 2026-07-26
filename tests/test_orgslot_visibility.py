import unittest

import torch
from torch.utils.data import Dataset

from util.label_visibility import (
    EvaluationVisibilityDataset,
    StrictVisibilityDataset,
    assert_case_splits_disjoint,
    load_visibility_config,
)


class _ToyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        label = torch.zeros(3, 8, 8)
        label[:, :2, :2] = 1
        label[:, 2:4, :2] = 2
        label[:, 4:6, :2] = 3
        label[:, 6:8, :2] = 4
        return {
            "image": torch.rand(3, 8, 8),
            "label": label,
            "case_name": "case_001",
            "slice_index": 7,
        }


class StrictVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes, cls.stages = load_visibility_config(
            "configs/orgslot/owt_legacy_debug.json"
        )

    def test_base_hides_full_label_and_future_liver(self):
        sample = StrictVisibilityDataset(
            _ToyDataset(), self.classes, self.stages["base"]
        )[0]
        self.assertEqual(
            set(sample),
            {"image", "visible_masks", "sample_index", "case_id", "slice_index"},
        )
        self.assertNotIn("label", sample)
        self.assertNotIn("full_label", sample)
        self.assertEqual(
            set(sample["visible_masks"]),
            {"background", "kidney_combined", "spleen", "pancreas"},
        )
        # Future liver is unknown during base training, hence provisional BG.
        self.assertEqual(sample["visible_masks"]["background"][0, 7, 0], 1)
        self.assertNotIn("liver", sample["visible_masks"])

    def test_incremental_exposes_only_current_new_organ(self):
        sample = StrictVisibilityDataset(
            _ToyDataset(), self.classes, self.stages["incremental_liver"]
        )[0]
        self.assertEqual(set(sample["visible_masks"]), {"liver"})
        self.assertNotIn("background", sample["visible_masks"])
        self.assertNotIn("label", sample)
        self.assertEqual(sample["visible_masks"]["liver"].sum(), 4)

    def test_evaluation_is_the_only_wrapper_with_full_label(self):
        sample = EvaluationVisibilityDataset(_ToyDataset())[0]
        self.assertIn("full_label", sample)
        self.assertNotIn("label", sample)

    def test_case_split_overlap_is_rejected(self):
        assert_case_splits_disjoint({"base": ["a"], "inc": ["b"]})
        with self.assertRaisesRegex(ValueError, "case leakage"):
            assert_case_splits_disjoint({"base": ["a"], "inc": ["a"]})


if __name__ == "__main__":
    unittest.main()
