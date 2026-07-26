import csv
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from datasets.orgslot_manifest import (
    OrganSlotManifestDataset,
    parse_case_and_slice,
)
from util.label_visibility import StrictVisibilityDataset, load_visibility_config


class OrganSlotManifestTests(unittest.TestCase):
    def _make_manifest(self, root, frames=4):
        image_dir = Path(root) / "image" / "case_alpha"
        mask_dir = Path(root) / "mask" / "case_alpha"
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        rows = []
        for index in range(frames):
            image = np.full((32, 32, 3), 20 + index, dtype=np.uint8)
            image[4:12, 4:12] = 200
            label = np.zeros((32, 32), dtype=np.uint8)
            label[2:8, 2:8] = 1
            label[20:28, 20:28] = 4
            image_path = image_dir / f"case_alpha_{index}.jpg"
            mask_path = mask_dir / f"case_alpha_{index}.png"
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(mask_path), label)
            rows.append({"image_pth": str(image_path), "mask_pth": str(mask_path)})
        csv_path = Path(root) / "manifest.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("image_pth", "mask_pth"))
            writer.writeheader()
            writer.writerows(rows)
        return csv_path

    def test_case_and_numeric_slice_are_stable(self):
        self.assertEqual(
            parse_case_and_slice("/tmp/case_a/case_a_10.png"),
            ("case_a", 10),
        )

    def test_2d_manifest_and_strict_visibility(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = self._make_manifest(root)
            raw = OrganSlotManifestDataset(
                csv_path, dataset_type="2D", expected_size=32
            )
            self.assertEqual(raw[0]["image"].shape, (3, 32, 32))
            self.assertEqual(raw[0]["label"].shape, (1, 32, 32))
            self.assertEqual(raw[0]["case_id"], "case_alpha")
            self.assertEqual(raw[0]["slice_index"], 0)

            classes, stages = load_visibility_config(
                "configs/orgslot/owt_legacy_debug.json"
            )
            visible = StrictVisibilityDataset(raw, classes, stages["base"])[0]
            self.assertNotIn("label", visible)
            self.assertNotIn("full_label", visible)
            self.assertNotIn("liver", visible["visible_masks"])
            self.assertEqual(visible["case_id"], "case_alpha")

    def test_fixfr4_manifest_shape(self):
        with tempfile.TemporaryDirectory() as root:
            csv_path = self._make_manifest(root)
            raw = OrganSlotManifestDataset(
                csv_path,
                dataset_type="3D",
                fix_frame=4,
                expected_size=32,
                max_samples=1,
            )
            sample = raw[0]
            self.assertEqual(sample["image"].shape, (3, 4, 32, 32))
            self.assertEqual(sample["label"].shape, (1, 4, 32, 32))
            self.assertEqual(sample["slice_index"], 0)


if __name__ == "__main__":
    unittest.main()
