#!/usr/bin/env python3
"""Require a deterministic tiny joint objective to decrease."""

import argparse
import glob
import json
import math
from pathlib import Path
from statistics import mean


def window_mean(records, name, first):
    window = records[:5] if first else records[-5:]
    return mean(float(record[name]) for record in window)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    config = json.loads((run_dir / "resolved_config.json").read_text())
    expected = {
        "seg_supervision": "all",
        "tgr_mode": "fixed_per_sample",
        "disable_train_augmentation": True,
        "organ_roi_aug": False,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise RuntimeError("{}={} expected {}".format(name, config.get(name), value))
    if float(config.get("lambda_seg")) != 0.01:
        raise RuntimeError("tiny run must use lambda_seg=0.01")

    records = [
        json.loads(line)
        for line in (run_dir / "log.txt").read_text().splitlines()
        if line.strip()
    ]
    if len(records) < 10:
        raise RuntimeError("tiny run has too few epoch records")
    ratios = {}
    for name in (
        "train_loss",
        "train_reconstruction_loss",
        "train_segmentation_loss",
    ):
        initial = window_mean(records, name, True)
        final = window_mean(records, name, False)
        if not math.isfinite(initial) or not math.isfinite(final):
            raise RuntimeError("non-finite {}".format(name))
        ratios[name] = final / max(initial, 1e-12)
        if ratios[name] >= 0.9:
            raise RuntimeError("{} did not overfit enough: {}".format(name, ratios[name]))
    latest = records[-1]
    if abs(float(latest["train_roi_fraction"])) > 1e-12:
        raise RuntimeError("tiny control unexpectedly used ROI")
    for slot in (
        "background", "spleen", "right_kidney", "left_kidney",
        "gallbladder", "esophagus", "pancreas", "liver", "stomach",
    ):
        if float(latest["train_seg_{}_supervised_samples".format(slot)]) <= 0:
            raise RuntimeError("{} received no segmentation supervision".format(slot))
    checkpoints = glob.glob(str(run_dir / "checkpoint-*.pth"))
    if not checkpoints:
        raise RuntimeError("tiny run produced no checkpoint")
    print(json.dumps({"validated": True, "loss_ratios": ratios}, sort_keys=True))


if __name__ == "__main__":
    main()
