#!/usr/bin/env python3
"""Audit focus retention, scale, and collateral clipping for the ROI policy."""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.orgslot_highres import OrganSlotHighResTransform


def accumulator():
    return {"events": 0, "sum": 0.0, "minimum": 1.0, "disappeared": 0, "partial": 0}


def update(stats, retained, original):
    ratio = retained / original
    stats["events"] += 1
    stats["sum"] += ratio
    stats["minimum"] = min(stats["minimum"], ratio)
    stats["disappeared"] += int(retained == 0)
    stats["partial"] += int(retained < original)


def finalize(stats):
    events = stats["events"]
    return {
        "events": events,
        "mean_retained_fraction": stats["sum"] / events if events else None,
        "minimum_retained_fraction": stats["minimum"] if events else None,
        "partial_rate": stats["partial"] / events if events else None,
        "disappearance_rate": stats["disappeared"] / events if events else None,
    }


def main():
    parser = argparse.ArgumentParser("Audit 20% small-organ ROI crop policy")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials-per-slice", type=int, default=4)
    parser.add_argument("--max-slices-per-class", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.trials_per_slice < 1:
        raise ValueError("trials-per-slice must be positive")

    rows = list(csv.DictReader(open(args.manifest, "r", encoding="utf-8", newline="")))
    metadata = json.load(open(args.roi_index, "r", encoding="utf-8"))
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
    focus_stats = {str(value): accumulator() for value in metadata["focus_class_ids"]}
    resized_factors = {str(value): [] for value in metadata["focus_class_ids"]}
    collateral = {str(value): accumulator() for value in range(1, 9)}

    for focus_id in metadata["focus_class_ids"]:
        indices = metadata["class_indices"][str(focus_id)]
        if args.max_slices_per_class is not None:
            indices = indices[: args.max_slices_per_class]
        for index in indices:
            label_np = cv2.imread(rows[index]["mask_pth"], cv2.IMREAD_GRAYSCALE)
            if label_np is None:
                raise FileNotFoundError(rows[index]["mask_pth"])
            label = torch.from_numpy(label_np.astype(np.int64)).unsqueeze(0)
            source_h, source_w = label_np.shape
            focus_original = int(np.count_nonzero(label_np == focus_id))
            if not focus_original:
                raise ValueError("ROI index points to a non-positive slice")
            for trial in range(args.trials_per_slice):
                torch.manual_seed(
                    args.seed + int(index) * 1009 + int(focus_id) * 9176 + trial
                )
                top, left, crop_h, crop_w = transform._organ_crop(
                    label, focus_id, source_h, source_w
                )
                crop = label_np[top : top + crop_h, left : left + crop_w]
                focus_retained = int(np.count_nonzero(crop == focus_id))
                update(focus_stats[str(focus_id)], focus_retained, focus_original)
                resized = cv2.resize(crop, (448, 448), interpolation=cv2.INTER_NEAREST)
                resized_factors[str(focus_id)].append(
                    int(np.count_nonzero(resized == focus_id)) / focus_original
                )
                for class_id in range(1, 9):
                    original = int(np.count_nonzero(label_np == class_id))
                    if original:
                        retained = int(np.count_nonzero(crop == class_id))
                        update(collateral[str(class_id)], retained, original)

    report = {
        "manifest": str(args.manifest),
        "roi_index": str(args.roi_index),
        "policy": {
            "global_probability": 0.8,
            "roi_probability": 0.2,
            "global_crop_size": 448,
            "roi_crop_size": 384,
            "output_size": 448,
            "roi_center_jitter": 0.1,
        },
        "focus_retention": {
            key: {
                **finalize(value),
                "mean_resized_pixel_factor": float(np.mean(resized_factors[key])),
                "minimum_resized_pixel_factor": float(np.min(resized_factors[key])),
                "maximum_resized_pixel_factor": float(np.max(resized_factors[key])),
            }
            for key, value in focus_stats.items()
        },
        "collateral_retention": {
            key: finalize(value) for key, value in collateral.items()
        },
    }
    for key, stats in report["focus_retention"].items():
        if stats["minimum_retained_fraction"] != 1.0:
            raise RuntimeError("focus class {} was clipped".format(key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report["focus_retention"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
