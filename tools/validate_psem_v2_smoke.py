#!/usr/bin/env python3
"""Fail a dependent full run when a PSEM-v2 smoke is incomplete."""

import argparse
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "train_loss",
    "train_p_loss",
    "train_present_classes",
    "train_kept_classes",
    "train_dropped_classes",
    "train_padding_fraction",
    "train_negative_hallucination_energy",
    "train_negative_query_count",
    "train_mode_direct_positive_fraction",
    "train_mode_direct_negative_fraction",
    "train_mode_leave_one_out_fraction",
    "train_mode_whole_fraction",
    "train_mode_background_only_fraction",
    "train_mode_organs_only_fraction",
    "train_mode_random_subset_fraction",
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
    if metrics.get("method") != "PSEM-v2a":
        raise RuntimeError(f"unexpected smoke method: {metrics.get('method')}")
    if metrics["train_kept_classes"] <= 0:
        raise RuntimeError("smoke kept no token groups")
    if not 0 <= metrics["train_padding_fraction"] < 1:
        raise RuntimeError("padding fraction must be in [0, 1)")
    if metrics["train_negative_query_count"] <= 0:
        raise RuntimeError("smoke observed no direct-negative queries")
    if metrics["train_negative_hallucination_energy"] < 0:
        raise RuntimeError("negative hallucination energy must be non-negative")
    mode_keys = [key for key in REQUIRED_METRICS if "_mode_" in key]
    for key in mode_keys:
        if metrics[key] <= 0:
            raise RuntimeError(f"smoke did not cover query mode: {key}")

    print("PSEM-v2 smoke validation passed")
    for key in REQUIRED_METRICS:
        print(f"{key}={metrics[key]}")


if __name__ == "__main__":
    main()
