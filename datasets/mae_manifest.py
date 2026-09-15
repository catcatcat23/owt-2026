"""Image-only CSV dataset and patient-level split for CT MAE pretraining."""

import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_f

from datasets.orgslot_manifest import parse_case_and_slice


_MAE_SLICE_PATTERN = re.compile(r"_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def read_image_records(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_pth" not in reader.fieldnames:
            raise ValueError("MAE manifest requires an image_pth column")
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError("empty MAE manifest: {}".format(csv_path))
    return records


def split_records_by_case(records, validation_fraction=0.1, seed=0):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0,1)")
    cases = sorted(
        {parse_case_and_slice(row["image_pth"])[0] for row in records}
    )
    if len(cases) < 2:
        raise ValueError("at least two cases are required for a split")
    shuffled = list(cases)
    random.Random(int(seed)).shuffle(shuffled)
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_cases = set(shuffled[:validation_count])
    train = []
    validation = []
    for row in records:
        case_name, _ = parse_case_and_slice(row["image_pth"])
        (validation if case_name in validation_cases else train).append(row)
    if not train or not validation:
        raise ValueError("patient-level split produced an empty partition")
    return train, validation


class MAEManifestDataset(Dataset):
    def __init__(self, records, input_size=224, training=True):
        self.records = list(records)
        self.input_size = int(input_size)
        self.training = bool(training)
        self.case_ids = tuple(
            sorted(
                {
                    parse_case_and_slice(row["image_pth"])[0]
                    for row in self.records
                }
            )
        )

    def __len__(self):
        return len(self.records)

    def _transform(self, image):
        if self.training:
            scale = 0.8 + 0.2 * float(torch.rand(()))
            crop_size = max(16, int(round(min(image.shape[-2:]) * scale)))
            max_top = image.shape[-2] - crop_size
            max_left = image.shape[-1] - crop_size
            top = int(torch.randint(0, max_top + 1, ()).item()) if max_top else 0
            left = int(torch.randint(0, max_left + 1, ()).item()) if max_left else 0
            image = image[:, top : top + crop_size, left : left + crop_size]
        image = vision_f.resize(
            image,
            [self.input_size, self.input_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return image.clamp_(0.0, 1.0)

    def __getitem__(self, index):
        row = self.records[index]
        image_path = Path(row["image_pth"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError("failed to read {}".format(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        image = image.permute(2, 0, 1).contiguous()
        case_name, slice_index = parse_case_and_slice(image_path)
        return {
            "image": self._transform(image),
            "sample_index": int(index),
            "case_name": case_name,
            "slice_index": int(slice_index),
        }


class MAEVolumeManifestDataset(Dataset):
    """Four consecutive grayscale CT slices with one coherent spatial crop."""

    def __init__(self, records, input_size=224, frames=4, training=True):
        self.records = list(records)
        self.input_size = int(input_size)
        self.frames = int(frames)
        self.training = bool(training)
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        self.case_ids = tuple(
            sorted(
                {
                    parse_case_and_slice(row["image_pth"])[0]
                    for row in self.records
                }
            )
        )

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _slice_path(path, slice_index):
        path = Path(path)
        match = _MAE_SLICE_PATTERN.search(path.name)
        if match is None:
            raise ValueError("cannot replace slice index in {}".format(path))
        name = (
            path.name[: match.start(1)]
            + str(int(slice_index))
            + path.name[match.end(1) :]
        )
        return path.with_name(name)

    def _transform(self, image):
        if self.training:
            scale = 0.8 + 0.2 * float(torch.rand(()))
            crop_size = max(16, int(round(min(image.shape[-2:]) * scale)))
            max_top = image.shape[-2] - crop_size
            max_left = image.shape[-1] - crop_size
            top = int(torch.randint(0, max_top + 1, ()).item()) if max_top else 0
            left = int(torch.randint(0, max_left + 1, ()).item()) if max_left else 0
            image = image[..., top : top + crop_size, left : left + crop_size]
        image = vision_f.resize(
            image,
            [self.input_size, self.input_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        return image.clamp_(0.0, 1.0)

    def __getitem__(self, index):
        row = self.records[index]
        case_name, start_slice = parse_case_and_slice(row["image_pth"])
        images = []
        for offset in range(self.frames):
            image_path = self._slice_path(row["image_pth"], start_slice + offset)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError("failed to read {}".format(image_path))
            images.append(image)
        image = torch.from_numpy(np.stack(images).astype(np.float32))
        minimum = image.amin()
        maximum = image.amax()
        image = (image - minimum) / (maximum - minimum + 1e-8)
        image = image.unsqueeze(0).repeat(3, 1, 1, 1)
        return {
            "image": self._transform(image),
            "sample_index": int(index),
            "case_name": case_name,
            "slice_index": int(start_slice),
            "slice_indices": torch.arange(
                start_slice, start_slice + self.frames, dtype=torch.int64
            ),
        }
