#!/usr/bin/env python3
"""Fail a dependent full run when a LossBalance-v3 smoke is incomplete."""

import argparse
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "train_global_recon_loss",
    "train_positive_roi_loss",
    "train_positive_valid_samples",
    "train_positive_weighted_mass",
    "train_removed_monitor_loss",
    "train_removed_valid_samples",
    "train_removed_mass",
    "train_optimized_loss",
    "train_p_loss",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    log_path = run_dir / "log.txt"
    checkpoint_path = run_dir / "checkpoint-0.pth"
    if not log_path.is_file() or not checkpoint_path.is_file():
        raise RuntimeError("smoke did not create log.txt and checkpoint-0.pth")

    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("smoke log.txt is empty")
    metrics = json.loads(lines[-1])
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise RuntimeError(f"missing smoke metrics: {missing}")

    for key in REQUIRED_METRICS:
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite smoke metric: {key}={metrics[key]}")
    if metrics["train_positive_valid_samples"] <= 0:
        raise RuntimeError("smoke observed no kept foreground supervision")
    if metrics["train_removed_valid_samples"] <= 0:
        raise RuntimeError("smoke observed no removed foreground state")
    if metrics["train_positive_weighted_mass"] <= 0:
        raise RuntimeError("positive weighted mass must be greater than zero")
    if metrics["train_removed_mass"] <= 0:
        raise RuntimeError("removed monitoring mass must be greater than zero")

    print("LossBalance-v3 smoke validation passed")
    for key in REQUIRED_METRICS:
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    main()
