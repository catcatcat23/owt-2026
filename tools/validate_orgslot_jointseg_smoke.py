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
    parser.add_argument("--expected-seg-loss", default="dice_bce")
    parser.add_argument(
        "--expected-seg-supervision", choices=("retained", "all"), default="all"
    )
    parser.add_argument("--expected-slot-head-type", default="linear")
    parser.add_argument("--expected-background-weight", default=0.25, type=float)
    parser.add_argument("--expected-lambda-seg", default=0.01, type=float)
    parser.add_argument("--expected-focal-alpha", default=0.75, type=float)
    parser.add_argument("--expected-focal-gamma", default=2.0, type=float)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    if not config.get("organ_roi_aug"):
        raise RuntimeError("ROI augmentation was not enabled")
    if config.get("seg_supervision") != args.expected_seg_supervision:
        raise RuntimeError("unexpected seg_supervision")
    if config.get("slot_head_type") != args.expected_slot_head_type:
        raise RuntimeError("unexpected slot_head_type")
    if float(config.get("lambda_bg_seg")) != args.expected_background_weight:
        raise RuntimeError("unexpected lambda_bg_seg")
    if config.get("seg_loss_type") != args.expected_seg_loss:
        raise RuntimeError("unexpected segmentation loss type")
    if float(config.get("lambda_seg")) != args.expected_lambda_seg:
        raise RuntimeError("unexpected lambda_seg")
    if args.expected_seg_loss == "focal":
        if float(config.get("focal_alpha")) != args.expected_focal_alpha:
            raise RuntimeError("unexpected focal_alpha")
        if float(config.get("focal_gamma")) != args.expected_focal_gamma:
            raise RuntimeError("unexpected focal_gamma")
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
    expected_weighted = args.expected_lambda_seg * float(
        latest["train_segmentation_loss"]
    )
    if abs(float(latest["train_weighted_segmentation_loss"]) - expected_weighted) > 1e-6:
        raise RuntimeError("weighted segmentation loss is inconsistent")
    if float(latest["train_roi_fraction"]) <= 0:
        raise RuntimeError("ROI path was not exercised")
    if any("positive_roi_loss" in name for name in latest):
        raise RuntimeError("legacy fused Loss3 metrics unexpectedly exist")
    observed_slots = []
    for slot in (
        "background", "spleen", "right_kidney", "left_kidney",
        "gallbladder", "esophagus", "pancreas", "liver", "stomach",
    ):
        loss_name = "train_seg_{}_loss".format(slot)
        count_name = "train_seg_{}_supervised_samples".format(slot)
        if loss_name not in latest:
            if slot == "background" and args.expected_background_weight == 0:
                continue
            continue
        if not math.isfinite(float(latest[loss_name])):
            raise RuntimeError("non-finite {}".format(loss_name))
        if float(latest[count_name]) <= 0:
            raise RuntimeError("{} received no supervision".format(slot))
        observed_slots.append(slot)
        for suffix in (
            "predicted_fraction", "target_fraction", "positive_probability"
        ):
            metric = "train_seg_{}_{}".format(slot, suffix)
            if metric not in latest or not math.isfinite(float(latest[metric])):
                raise RuntimeError(
                    "missing or non-finite {}".format(metric)
                )
    if not set(("gallbladder", "esophagus", "pancreas")) & set(observed_slots):
        raise RuntimeError("smoke did not supervise a small-organ focus slot")
    checkpoints = glob.glob(str(run_dir / "checkpoint-*.pth"))
    if not checkpoints:
        raise RuntimeError("smoke produced no checkpoint")
    print("validated joint-seg ROI smoke", checkpoints[-1])


if __name__ == "__main__":
    main()
