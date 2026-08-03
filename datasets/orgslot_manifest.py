"""Deterministic OWT CSV reader used only by OrganSlotBank stages.

The legacy reader mixes I/O, augmentation, and tensor reshaping.  This adapter
keeps the real-data Gate-B path deliberately small: read the already
preprocessed slices, recover stable case/slice identifiers, and return the raw
label only to the strict visibility wrapper.
"""

import csv
from pathlib import Path
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_SLICE_PATTERN = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def parse_case_and_slice(path):
    path = Path(path)
    match = _SLICE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse numeric slice index from {path}")
    return path.parent.name, int(match.group(1))


def _normalize_image(image, mode):
    image = image.astype(np.float32)
    if mode == "per_sample":
        minimum = float(image.min())
        maximum = float(image.max())
        return (image - minimum) / (maximum - minimum + 1e-8)
    if mode == "fixed_255":
        return image / 255.0
    raise ValueError(f"unsupported intensity normalization: {mode}")


def _read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_pth", "mask_pth"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"manifest must contain {sorted(required)}, got {reader.fieldnames}"
            )
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


class OrganSlotManifestDataset(Dataset):
    """Read OWT-native 2D slices or consecutive Fixfr4 windows.

    Returned labels intentionally remain full raw maps here.  Training code
    must wrap this dataset with ``StrictVisibilityDataset`` before creating a
    DataLoader.
    """

    def __init__(
        self,
        csv_path,
        dataset_type="2D",
        fix_frame=4,
        intensity_norm="per_sample",
        expected_size=224,
        max_samples=None,
    ):
        if dataset_type not in {"2D", "3D"}:
            raise ValueError("dataset_type must be '2D' or '3D'")
        self.csv_path = str(csv_path)
        self.dataset_type = dataset_type
        self.fix_frame = int(fix_frame)
        self.intensity_norm = intensity_norm
        self.expected_size = (
            None if expected_size is None else int(expected_size)
        )
        self.records = _read_csv(csv_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]

        self.case_ids = tuple(
            sorted({parse_case_and_slice(row["image_pth"])[0] for row in self.records})
        )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _checked_read(path, flag):
        value = cv2.imread(str(path), flag)
        if value is None:
            raise FileNotFoundError(f"failed to read {path}")
        return value

    def _check_spatial_size(self, value, path):
        if self.expected_size is None:
            return
        if tuple(value.shape[:2]) != (self.expected_size, self.expected_size):
            raise ValueError(
                f"expected {self.expected_size}x{self.expected_size}, "
                f"got {value.shape[:2]} for {path}"
            )

    def _load_2d(self, record):
        image = self._checked_read(record["image_pth"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self._checked_read(record["mask_pth"], cv2.IMREAD_GRAYSCALE)
        self._check_spatial_size(image, record["image_pth"])
        self._check_spatial_size(label, record["mask_pth"])
        image = _normalize_image(image, self.intensity_norm)
        return (
            torch.from_numpy(image).permute(2, 0, 1).float(),
            torch.from_numpy(label.astype(np.int64)).unsqueeze(0),
        )

    def _window_paths(self, path, start, suffix):
        path = Path(path)
        stem = path.name.rsplit("_", 1)[0]
        return [path.with_name(f"{stem}_{index}{suffix}") for index in range(
            start, start + self.fix_frame
        )]

    def _load_3d(self, record, start):
        image_paths = self._window_paths(record["image_pth"], start, ".jpg")
        mask_paths = self._window_paths(record["mask_pth"], start, ".png")
        images = []
        labels = []
        for image_path, mask_path in zip(image_paths, mask_paths):
            image = self._checked_read(image_path, cv2.IMREAD_GRAYSCALE)
            label = self._checked_read(mask_path, cv2.IMREAD_GRAYSCALE)
            self._check_spatial_size(image, image_path)
            self._check_spatial_size(label, mask_path)
            images.append(image)
            labels.append(label)
        image = _normalize_image(np.stack(images, axis=0), self.intensity_norm)
        label = np.stack(labels, axis=0).astype(np.int64)
        image_tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1, 1)
        label_tensor = torch.from_numpy(label).unsqueeze(0)
        return image_tensor.float(), label_tensor.long()

    def __getitem__(self, index):
        record = self.records[index]
        image_case, image_slice = parse_case_and_slice(record["image_pth"])
        mask_case, mask_slice = parse_case_and_slice(record["mask_pth"])
        if (image_case, image_slice) != (mask_case, mask_slice):
            raise ValueError(
                "image/mask case or slice mismatch: "
                f"{record['image_pth']} vs {record['mask_pth']}"
            )
        if self.dataset_type == "2D":
            image, label = self._load_2d(record)
        else:
            image, label = self._load_3d(record, image_slice)
        return {
            "image": image,
            "label": label,
            "sample_index": int(index),
            "case_name": image_case,
            "case_id": image_case,
            "slice_index": image_slice,
        }
