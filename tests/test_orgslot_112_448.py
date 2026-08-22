import unittest

import numpy as np
import torch
import torch.nn.functional as F

from datasets.orgslot_highres import OrganSlotHighResTransform
from tools.preprocess_common8_112_448 import remap_common8


def make_sample(frames=None):
    label = torch.zeros(1, 501, 501, dtype=torch.long)
    label[:, 180:240, 210:250] = 4
    image = label.float().div(8).repeat(3, 1, 1)
    if frames is not None:
        image = image[:, None].repeat(1, frames, 1, 1)
        label = label[:, None].repeat(1, frames, 1, 1)
    return {"image": image, "label": label, "case_id": "case"}


class OrganSlot112Crop448Tests(unittest.TestCase):
    def test_common8_remap_drops_non_common_classes(self):
        source = np.array([[[0], [1], [2], [9], [11]]], dtype=np.int16)
        output = remap_common8(source, {1: 7, 2: 1})
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(output[:, :, 0].tolist(), [[0, 7, 1, 0, 0]])

    def test_roi_crop_is_paired_for_2d_and_fixfr4(self):
        for frames in (None, 4):
            with self.subTest(frames=frames):
                torch.manual_seed(7)
                transform = OrganSlotHighResTransform(
                    training=True,
                    global_crop_size=448,
                    roi_crop_size=384,
                    roi_center_jitter=0.0,
                    flip_probability=0.0,
                    rotation_probability=0.0,
                    gamma_probability=0.0,
                    photometric_operations=0,
                )
                sample = make_sample(frames)
                sample["focus_class_id"] = 4
                sample["roi_applied"] = True
                output = transform(sample)
                top, left, crop_h, crop_w, source_h, source_w = (
                    output["crop_box"].tolist()
                )
                self.assertEqual((source_h, source_w), (501, 501))
                self.assertEqual((crop_h, crop_w), (384, 384))
                self.assertTrue(output["roi_applied"])
                self.assertEqual(output["focus_class_id"], 4)
                self.assertEqual(output["image"].shape[-2:], (448, 448))
                self.assertEqual(output["label"].shape[-2:], (448, 448))
                self.assertLessEqual(set(torch.unique(output["label"]).tolist()), {0, 4})
                self.assertTrue(torch.any(output["label"] == 4))

                expected = sample["label"][
                    ..., top : top + crop_h, left : left + crop_w
                ]
                if expected.ndim == 3:
                    expected = F.interpolate(
                        expected[None].float(), size=(448, 448), mode="nearest"
                    )[0].long()
                else:
                    expected = expected.permute(1, 0, 2, 3)
                    expected = F.interpolate(
                        expected.float(), size=(448, 448), mode="nearest"
                    ).permute(1, 0, 2, 3).long()
                self.assertTrue(torch.equal(output["label"], expected))

    def test_evaluation_uses_center_448_crop(self):
        transform = OrganSlotHighResTransform(training=False)
        output = transform(make_sample())
        self.assertTrue(output["crop_applied"])
        self.assertFalse(output["crop_randomized"])
        self.assertFalse(output["roi_applied"])
        self.assertEqual(
            output["crop_box"].tolist(), [26, 26, 448, 448, 501, 501]
        )
        self.assertEqual(tuple(output["image"].shape), (3, 448, 448))

    def test_evaluation_uses_center_336_crop_without_losing_focus_organs(self):
        transform = OrganSlotHighResTransform(
            output_size=336,
            training=False,
            global_crop_size=336,
            roi_crop_size=224,
        )
        output = transform(make_sample())
        self.assertEqual(
            output["crop_box"].tolist(), [82, 82, 336, 336, 501, 501]
        )
        self.assertEqual(tuple(output["image"].shape), (3, 336, 336))
        self.assertTrue(torch.any(output["label"] == 4))

    def test_roi_224_retains_focus_organ_and_resizes_to_336(self):
        transform = OrganSlotHighResTransform(
            output_size=336,
            training=True,
            global_crop_size=336,
            roi_crop_size=224,
            roi_center_jitter=0.1,
            flip_probability=0.0,
            rotation_probability=0.0,
            gamma_probability=0.0,
            photometric_operations=0,
        )
        sample = make_sample()
        sample.update({"focus_class_id": 4, "roi_applied": True})
        output = transform(sample)
        self.assertEqual(output["crop_box"].tolist()[2:4], [224, 224])
        self.assertEqual(tuple(output["image"].shape), (3, 336, 336))
        self.assertTrue(torch.any(output["label"] == 4))

    def test_roi_384_at_point7mm_retains_large_pancreas_proxy(self):
        label = torch.zeros(1, 716, 716, dtype=torch.long)
        label[:, 240:474, 300:436] = 6
        sample = {
            "image": label.float().div(8).repeat(3, 1, 1),
            "label": label,
            "focus_class_id": 6,
            "roi_applied": True,
        }
        transform = OrganSlotHighResTransform(
            output_size=448,
            training=True,
            global_crop_size=448,
            roi_crop_size=384,
            roi_center_jitter=0.1,
            flip_probability=0.0,
            rotation_probability=0.0,
            gamma_probability=0.0,
            photometric_operations=0,
        )
        output = transform(sample)
        self.assertEqual(output["crop_box"].tolist()[2:4], [384, 384])
        self.assertEqual(tuple(output["image"].shape), (3, 448, 448))
        self.assertTrue(torch.any(output["label"] == 6))

    def test_transform_rejects_mismatched_spatial_size(self):
        transform = OrganSlotHighResTransform(training=False)
        with self.assertRaisesRegex(ValueError, "spatial shapes differ"):
            transform({
                "image": torch.zeros(3, 501, 501),
                "label": torch.zeros(1, 500, 501, dtype=torch.long),
            })

    def test_roi_crop_rejects_missing_focus_organ(self):
        transform = OrganSlotHighResTransform(training=True)
        sample = make_sample()
        sample["focus_class_id"] = 5
        sample["roi_applied"] = True
        with self.assertRaisesRegex(ValueError, "does not contain focus class"):
            transform(sample)

    def test_fixed_global_crop_rejects_small_native_matrix(self):
        transform = OrganSlotHighResTransform(training=False)
        with self.assertRaisesRegex(ValueError, "does not fit native matrix"):
            transform({
                "image": torch.zeros(3, 447, 500),
                "label": torch.zeros(1, 447, 500, dtype=torch.long),
            })


if __name__ == "__main__":
    unittest.main()
