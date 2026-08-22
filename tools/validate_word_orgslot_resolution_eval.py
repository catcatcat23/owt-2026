#!/usr/bin/env python3
"""Validate a complete WORD OrganSlot resolution-ablation evaluation."""

import argparse
import csv
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--expected-input-size", type=int, required=True)
    parser.add_argument("--expected-global-crop", type=int, required=True)
    parser.add_argument("--expected-spacing", nargs=3, type=float, required=True)
    parser.add_argument("--expected-cases", type=int, default=24)
    args = parser.parse_args()

    root = Path(args.eval_dir)
    required = (
        root / "resolved_config.json",
        root / "checkpoint_load.json",
        root / "results.json",
        root / "progress.json",
        root / "per_case.csv",
        root / "per_case.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads((root / "resolved_config.json").read_text())
    if int(config["input_size"]) != args.expected_input_size:
        raise RuntimeError("unexpected input size")
    if int(config["global_crop_size"]) != args.expected_global_crop:
        raise RuntimeError("unexpected global crop")
    recorded_spacing = [float(value) for value in config["preprocess_spacing_mm"]]
    if any(abs(left - right) > 1e-6 for left, right in zip(recorded_spacing, args.expected_spacing)):
        raise RuntimeError("unexpected preprocessing spacing")

    progress = json.loads((root / "progress.json").read_text())
    if not progress.get("complete") or float(progress.get("percent", 0)) != 100.0:
        raise RuntimeError("evaluation is incomplete")
    if int(progress["processed"]) != int(progress["total"]):
        raise RuntimeError("processed/total mismatch")

    load = json.loads((root / "checkpoint_load.json").read_text())
    if not load.get("exact"):
        raise RuntimeError("checkpoint load was not exact")
    if int(load["input_size"]) != args.expected_input_size:
        raise RuntimeError("checkpoint input mismatch")

    with open(root / "per_case.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    case_ids = {row["case_id"] for row in rows}
    class_ids = {int(row["class_id"]) for row in rows}
    modes = {row["mode"] for row in rows}
    if len(case_ids) != args.expected_cases:
        raise RuntimeError("expected {} cases, found {}".format(args.expected_cases, len(case_ids)))
    if class_ids != set(range(1, 9)):
        raise RuntimeError("expected Common8 foreground classes")
    if modes != {"direct_raw", "direct_post", "indirect_raw", "indirect_post"}:
        raise RuntimeError("missing evaluation modes")
    if len(rows) != args.expected_cases * 8 * 4:
        raise RuntimeError("unexpected per-case row count")
    for row in rows:
        for field in ("case_dice", "case_iou"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError("non-finite {} in per_case.csv".format(field))

    result = json.loads((root / "results.json").read_text())
    if not result.get("complete_test_set"):
        raise RuntimeError("results are not from the complete test set")
    if result.get("primary_mode") != "direct_post":
        raise RuntimeError("unexpected primary mode")
    print("validated complete evaluation", root)


if __name__ == "__main__":
    main()
