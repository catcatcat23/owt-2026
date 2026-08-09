#!/usr/bin/env python3
"""Validate a completed PSEM-v3 triplet smoke run."""

import argparse
import json
import math
from pathlib import Path


REQUIRED_METRICS = (
    "train_loss",
    "train_branch_loss",
    "train_global_recon_loss",
    "train_positive_roi_loss",
    "train_delta_loss",
    "train_delta_global_loss",
    "train_delta_positive_roi_loss",
    "train_p_loss",
    "train_direct_mse",
    "train_context_mse",
    "train_plus_mse",
    "train_anchor_present_fraction",
    "train_full_context_fraction",
    "train_direct_kept_classes",
    "train_context_kept_classes",
    "train_plus_kept_classes",
    "train_padding_fraction",
    "train_negative_direct_energy",
    "train_negative_delta_energy",
    "train_negative_anchor_count",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-foreground-classes", type=int, required=True)
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
    if metrics.get("method") != "PSEM-v3-triplet-Loss3":
        raise RuntimeError(f"unexpected method: {metrics.get('method')}")

    for key in REQUIRED_METRICS:
        if not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"non-finite metric: {key}={metrics[key]}")
    if metrics["train_direct_kept_classes"] != 1:
        raise RuntimeError("Direct queries must keep exactly one class")
    if not 0 < metrics["train_full_context_fraction"] < 1:
        raise RuntimeError("smoke did not cover full and random contexts")
    if not 0 < metrics["train_anchor_present_fraction"] < 1:
        raise RuntimeError("smoke did not cover present and absent anchors")
    if metrics["train_negative_anchor_count"] <= 0:
        raise RuntimeError("smoke observed no negative anchor queries")
    if metrics["train_delta_loss"] < 0:
        raise RuntimeError("delta loss must be non-negative")
    if metrics["train_negative_delta_energy"] < 0:
        raise RuntimeError("negative delta energy must be non-negative")

    for class_id in range(1, args.num_foreground_classes + 1):
        key = f"train_anchor_c{class_id}_fraction"
        if key not in metrics or not math.isfinite(float(metrics[key])):
            raise RuntimeError(f"missing/non-finite anchor coverage: {key}")
        if metrics[key] <= 0:
            raise RuntimeError(f"smoke did not query class {class_id}")

    print("PSEM-v3 triplet smoke validation passed")


if __name__ == "__main__":
    main()
