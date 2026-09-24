#!/usr/bin/env python3
"""Validate that every Fixfr4 CSV row resolves to four aligned slices."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--fix-frame", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=8)
    return parser.parse_args()


def indexed_path(path: str, offset: int) -> Path:
    source = Path(path)
    prefix, index_text = source.stem.rsplit("_", 1)
    return source.with_name(f"{prefix}_{int(index_text) + offset}{source.suffix}")


def validate(csv_path: str, args: argparse.Namespace) -> None:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    missing: list[str] = []
    decoded = 0
    labels_seen: set[int] = set()
    sample_indices = {0, len(rows) // 2, len(rows) - 1}

    for row_index, row in enumerate(rows):
        for key in ("image_pth", "mask_pth"):
            for offset in range(args.fix_frame):
                candidate = indexed_path(row[key], offset)
                if not candidate.is_file():
                    missing.append(str(candidate))
        if row_index not in sample_indices:
            continue

        for offset in range(args.fix_frame):
            image_path = indexed_path(row["image_pth"], offset)
            mask_path = indexed_path(row["mask_pth"], offset)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            expected = (args.input_size, args.input_size)
            if image is None or mask is None:
                raise ValueError(f"Decode failed: {image_path} or {mask_path}")
            if image.shape != expected or mask.shape != expected:
                raise ValueError(
                    f"Expected {expected}, got {image.shape}/{mask.shape}: {image_path}"
                )
            labels_seen.update(int(value) for value in np.unique(mask))
            decoded += 1

    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} missing Fixfr4 files in {csv_path}:\n{preview}"
        )
    invalid = sorted(value for value in labels_seen if value > args.num_classes)
    if invalid:
        raise ValueError(f"Labels exceed {args.num_classes}: {invalid}")
    print(
        f"OK {csv_path}: windows={len(rows)}, checked_paths={len(rows) * 8}, "
        f"decoded={decoded}, sampled_labels={sorted(labels_seen)}"
    )


def main() -> None:
    args = parse_args()
    for csv_path in args.csv:
        validate(csv_path, args)


if __name__ == "__main__":
    main()
