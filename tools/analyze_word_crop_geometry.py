#!/usr/bin/env python3
"""Quantify black-border removal and label retention for centered crops."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--crop-sizes", nargs="+", type=int, default=(448, 384, 336))
    parser.add_argument("--roi-size", type=int, default=224)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--black-threshold", type=int, default=1)
    return parser.parse_args()


def center_window(height, width, size):
    if min(height, width) < size:
        raise ValueError("crop {} does not fit {}x{}".format(size, height, width))
    top = (height - size) // 2
    left = (width - size) // 2
    return top, left


def quantiles(values):
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    }


def main():
    args = parse_args()
    sizes = sorted(set(args.crop_sizes), reverse=True)
    rows = []
    for csv_path in args.csv:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    stats = {
        size: {
            "black_fractions": [],
            "border_black_fractions": [],
            "foreground_full": np.zeros(9, dtype=np.int64),
            "foreground_crop": np.zeros(9, dtype=np.int64),
            "positive_slices": np.zeros(9, dtype=np.int64),
            "retained_positive_slices": np.zeros(9, dtype=np.int64),
        }
        for size in sizes
    }
    shapes = {}
    roi_bbox_max = {class_id: [0, 0] for class_id in (4, 5, 6)}
    roi_bbox_exceeds = {class_id: 0 for class_id in (4, 5, 6)}

    for row_index, row in enumerate(rows, start=1):
        image = cv2.imread(row["image_pth"], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(row["mask_pth"], cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise FileNotFoundError(row)
        if label.ndim == 3:
            label = label[..., 0]
        if image.shape != label.shape:
            raise ValueError("image/label shape mismatch: {}".format(row))
        height, width = image.shape
        shape_key = "{}x{}".format(height, width)
        shapes[shape_key] = shapes.get(shape_key, 0) + 1

        full_counts = np.bincount(label.ravel(), minlength=9)[:9]
        for class_id in (4, 5, 6):
            locations = np.argwhere(label == class_id)
            if locations.size:
                bbox_height = int(locations[:, 0].max() - locations[:, 0].min() + 1)
                bbox_width = int(locations[:, 1].max() - locations[:, 1].min() + 1)
                roi_bbox_max[class_id][0] = max(roi_bbox_max[class_id][0], bbox_height)
                roi_bbox_max[class_id][1] = max(roi_bbox_max[class_id][1], bbox_width)
                if bbox_height > args.roi_size or bbox_width > args.roi_size:
                    roi_bbox_exceeds[class_id] += 1

        for size in sizes:
            top, left = center_window(height, width, size)
            image_crop = image[top : top + size, left : left + size]
            label_crop = label[top : top + size, left : left + size]
            crop_counts = np.bincount(label_crop.ravel(), minlength=9)[:9]
            record = stats[size]
            record["black_fractions"].append(
                float(np.mean(image_crop <= args.black_threshold))
            )
            border = np.concatenate(
                (
                    image_crop[:16].ravel(),
                    image_crop[-16:].ravel(),
                    image_crop[16:-16, :16].ravel(),
                    image_crop[16:-16, -16:].ravel(),
                )
            )
            record["border_black_fractions"].append(
                float(np.mean(border <= args.black_threshold))
            )
            record["foreground_full"] += full_counts
            record["foreground_crop"] += crop_counts
            record["positive_slices"] += full_counts > 0
            record["retained_positive_slices"] += crop_counts > 0

        if row_index % 5000 == 0:
            print("processed {}/{}".format(row_index, len(rows)), flush=True)

    output = {
        "csv": args.csv,
        "rows": len(rows),
        "black_threshold": args.black_threshold,
        "source_shapes": shapes,
        "roi_{}".format(args.roi_size): {
            "bbox_max_hw": {str(k): v for k, v in roi_bbox_max.items()},
            "positive_slices_exceeding_crop": {
                str(k): v for k, v in roi_bbox_exceeds.items()
            },
        },
        "crops": {},
    }
    for size, record in stats.items():
        full = record["foreground_full"]
        crop = record["foreground_crop"]
        positive = record["positive_slices"]
        retained = record["retained_positive_slices"]
        output["crops"][str(size)] = {
            "black_fraction_quantiles": quantiles(record["black_fractions"]),
            "outer_16px_black_fraction_quantiles": quantiles(
                record["border_black_fractions"]
            ),
            "class_pixel_retention": {
                str(class_id): float(crop[class_id] / max(full[class_id], 1))
                for class_id in range(1, 9)
            },
            "class_positive_slice_retention": {
                str(class_id): float(retained[class_id] / max(positive[class_id], 1))
                for class_id in range(1, 9)
            },
            "class_positive_slices_missed": {
                str(class_id): int(positive[class_id] - retained[class_id])
                for class_id in range(1, 9)
            },
            "class_positive_slices": {
                str(class_id): int(positive[class_id])
                for class_id in range(1, 9)
            },
            "class_retained_positive_slices": {
                str(class_id): int(retained[class_id])
                for class_id in range(1, 9)
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print("wrote {}".format(args.output))


if __name__ == "__main__":
    main()
