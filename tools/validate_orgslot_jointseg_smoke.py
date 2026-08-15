#!/usr/bin/env python3
"""Validate a real WORD ROI20 joint reconstruction+slot-segmentation smoke."""

import argparse
import glob
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    if not config.get("organ_roi_aug"):
        raise RuntimeError("ROI augmentation was not enabled")
    if config.get("seg_supervision") != "all":
        raise RuntimeError("seg_supervision must be all")
    if float(config.get("lambda_seg")) != 0.01:
        raise RuntimeError("lambda_seg must be 0.01")
    if float(config.get("lambda_lpips")) != 1.0:
        raise RuntimeError("lambda_lpips must be 1")

    records = [
        json.loads(line)
        for line in (run_dir / "log.txt").read_text().splitlines()
        if line.strip()
    ]
    if not records:
        raise RuntimeError("training log contains no epoch records")
    latest = records[-1]
    for name in (
        "train_loss", "train_reconstruction_loss", "train_p_loss",
        "train_segmentation_loss", "train_weighted_segmentation_loss",
    ):
        if not math.isfinite(float(latest[name])):
            raise RuntimeError("non-finite {}".format(name))
    expected_weighted = 0.01 * float(latest["train_segmentation_loss"])
    if abs(float(latest["train_weighted_segmentation_loss"]) - expected_weighted) > 1e-6:
        raise RuntimeError("weighted segmentation loss is inconsistent")
    if float(latest["train_roi_fraction"]) <= 0:
        raise RuntimeError("ROI path was not exercised")
    if any("positive_roi_loss" in name for name in latest):
        raise RuntimeError("legacy fused Loss3 metrics unexpectedly exist")
    for slot in (
        "background", "spleen", "right_kidney", "left_kidney",
        "gallbladder", "esophagus", "pancreas", "liver", "stomach",
    ):
        loss_name = "train_seg_{}_loss".format(slot)
        count_name = "train_seg_{}_supervised_samples".format(slot)
        if not math.isfinite(float(latest[loss_name])):
            raise RuntimeError("non-finite {}".format(loss_name))
        if float(latest[count_name]) <= 0:
            raise RuntimeError("{} received no supervision".format(slot))
    checkpoints = glob.glob(str(run_dir / "checkpoint-*.pth"))
    if not checkpoints:
        raise RuntimeError("smoke produced no checkpoint")
    print("validated joint-seg ROI smoke", checkpoints[-1])


if __name__ == "__main__":
    main()
