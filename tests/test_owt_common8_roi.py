import unittest

import torch

from engine_pretrain_owt_common8_a100 import _force_batch_focus_classes


class OWTCommon8ROITests(unittest.TestCase):
    def test_focus_union_is_retained_for_every_row(self):
        keep = torch.tensor([
            [True, True, True, True, False, False, False, True, True],
            [True, True, True, True, False, False, False, True, True],
            [True, True, True, True, False, False, False, True, True],
        ])
        batch = {
            "focus_class_id": torch.tensor([4, -1, 6]),
            "roi_applied": torch.tensor([True, False, True]),
        }
        result, roi = _force_batch_focus_classes(
            batch, keep, tuple(range(9)), torch.device("cpu")
        )
        self.assertTrue(torch.all(result[:, 4]))
        self.assertTrue(torch.all(result[:, 6]))
        self.assertTrue(torch.all(~result[:, 5]))
        self.assertTrue(torch.equal(result, result[:1].expand_as(result)))
        self.assertTrue(torch.equal(roi, torch.tensor([True, False, True])))

    def test_absent_roi_metadata_keeps_control_behavior(self):
        keep = torch.tensor([[True, False], [True, False]])
        result, roi = _force_batch_focus_classes(
            {}, keep.clone(), (0, 1), torch.device("cpu")
        )
        self.assertTrue(torch.equal(result, keep))
        self.assertFalse(torch.any(roi))

    def test_inconsistent_roi_metadata_is_rejected(self):
        keep = torch.ones(2, 3, dtype=torch.bool)
        batch = {
            "focus_class_id": torch.tensor([1, -1]),
            "roi_applied": torch.tensor([False, False]),
        }
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _force_batch_focus_classes(
                batch, keep, (0, 1, 2), torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
