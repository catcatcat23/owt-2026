#!/usr/bin/env python3
"""Validate a completed OrganSlot WORD448 ROI + Loss-v3 smoke run."""

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
    if float(config.get("positive_roi_loss_weight", 0)) != 0.25:
        raise RuntimeError("Loss-v3 positive weight is not 0.25")
    if len(config.get("roi_class_weights", ())) != 9:
        raise RuntimeError("expected nine Common8 class weights")

    records = []
    for line in (run_dir / "log.txt").read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        raise RuntimeError("training log contains no epoch records")
    latest = records[-1]
    required = (
        "train_loss", "train_reconstruction_loss", "train_p_loss",
        "train_positive_roi_loss", "train_positive_valid_samples",
        "train_positive_weighted_mass", "train_roi_fraction",
    )
    for name in required:
        value = float(latest[name])
        if not math.isfinite(value):
            raise RuntimeError("non-finite {}".format(name))
    if float(latest["train_positive_valid_samples"]) <= 0:
        raise RuntimeError("Loss-v3 had no positive supervision")
    if float(latest["train_roi_fraction"]) <= 0:
        raise RuntimeError("ROI path was not exercised")
    checkpoints = glob.glob(str(run_dir / "checkpoint-*.pth"))
    if not checkpoints:
        raise RuntimeError("smoke produced no checkpoint")
    print("validated", run_dir, "checkpoint", checkpoints[-1])


if __name__ == "__main__":
    main()
