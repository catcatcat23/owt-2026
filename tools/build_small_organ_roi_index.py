#!/usr/bin/env python3
"""Build reproducible class-balanced positive-slice pools for ROI sampling."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ids(value):
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or len(set(result)) != len(result):
        raise ValueError("class IDs must be a non-empty unique list")
    return result


def main():
    parser = argparse.ArgumentParser("Build small-organ positive slice index")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-class-ids", default="4,5,6")
    parser.add_argument("--allowed-class-ids", default="0,1,2,3,4,5,6,7,8")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("refusing to overwrite {}".format(args.output))
    focus_ids = parse_ids(args.focus_class_ids)
    allowed_ids = set(parse_ids(args.allowed_class_ids))
    if not set(focus_ids).issubset(allowed_ids):
        raise ValueError("focus classes must be included in allowed classes")

    with open(args.manifest, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"image_pth", "mask_pth"}.issubset(rows[0]):
        raise ValueError("manifest is empty or lacks image_pth/mask_pth")

    class_indices = {str(class_id): [] for class_id in focus_ids}
    class_positive_pixels = {str(class_id): 0 for class_id in focus_ids}
    unexpected_ids = set()
    for index, row in enumerate(rows):
        label = cv2.imread(row["mask_pth"], cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(row["mask_pth"])
        observed = {int(value) for value in np.unique(label)}
        unexpected_ids.update(observed - allowed_ids)
        for class_id in focus_ids:
            pixel_count = int(np.count_nonzero(label == class_id))
            if pixel_count:
                class_indices[str(class_id)].append(index)
                class_positive_pixels[str(class_id)] += pixel_count
    if unexpected_ids:
        raise ValueError("unexpected label IDs: {}".format(sorted(unexpected_ids)))
    empty = [key for key, values in class_indices.items() if not values]
    if empty:
        raise ValueError("no positive slices for focus classes {}".format(empty))

    output = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "sample_count": len(rows),
        "focus_class_ids": list(focus_ids),
        "allowed_class_ids": sorted(allowed_ids),
        "class_indices": class_indices,
        "class_positive_slice_counts": {
            key: len(values) for key, values in class_indices.items()
        },
        "class_positive_pixels": class_positive_pixels,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "output": str(args.output),
        "sample_count": len(rows),
        "positive_slice_counts": output["class_positive_slice_counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
