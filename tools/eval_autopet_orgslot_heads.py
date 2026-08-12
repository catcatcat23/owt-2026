#!/usr/bin/env python3
"""Evaluate OrganSlotBank segmentation heads on complete 2D CT cases."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from OWT_models_orgslot import FUSION_MODES
from torch.utils.data import DataLoader

from eval_orgslot import calibrated_multiclass_prediction
from tools.eval_autopet_reconstruction_threshold import (
    CLASS_NAMES,
    AutoPETManifestDataset,
    atomic_json,
    build_model,
    dice_from_counts,
    postprocess_slice,
    sha256,
)


def parse_args():
    parser = argparse.ArgumentParser("AutoPET OrganSlot head evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="post_layernorm",
    )
    return parser.parse_args()


def new_counter():
    return {
        "slices": 0,
        "prediction": 0,
        "target": 0,
        "intersection": 0,
        "present_slice_dice_sum": 0.0,
        "present_slices": 0,
    }


def update(counter, prediction, target):
    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    intersection = int(np.logical_and(prediction, target).sum())
    counter["slices"] += 1
    counter["prediction"] += prediction_count
    counter["target"] += target_count
    counter["intersection"] += intersection
    if target_count:
        counter["present_slice_dice_sum"] += dice_from_counts(
            intersection, prediction_count, target_count
        )
        counter["present_slices"] += 1


def summarize(counters):
    metrics = {}
    rows = []
    for mode, mode_counters in counters.items():
        metrics[mode] = {}
        for class_id, class_counters in mode_counters.items():
            present_case_dice = []
            totals = {"intersection": 0, "prediction": 0, "target": 0}
            slice_dice_sum = 0.0
            present_slices = 0
            for case_id in sorted(class_counters):
                counter = class_counters[case_id]
                case_dice = dice_from_counts(
                    counter["intersection"],
                    counter["prediction"],
                    counter["target"],
                )
                if counter["target"]:
                    present_case_dice.append(case_dice)
                for key in totals:
                    totals[key] += counter[key]
                slice_dice_sum += counter["present_slice_dice_sum"]
                present_slices += counter["present_slices"]
                rows.append({
                    "mode": mode,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "case_id": case_id,
                    "slices": counter["slices"],
                    "prediction_voxels": counter["prediction"],
                    "target_voxels": counter["target"],
                    "case_dice": case_dice,
                })
            metrics[mode][str(class_id)] = {
                "class_name": CLASS_NAMES[class_id],
                "case_count": len(class_counters),
                "gt_present_case_count": len(present_case_dice),
                "case_dice_presence_mean": (
                    float(np.mean(present_case_dice))
                    if present_case_dice else float("nan")
                ),
                "global_dice": dice_from_counts(
                    totals["intersection"],
                    totals["prediction"],
                    totals["target"],
                ),
                "slice_dice_present_mean": (
                    slice_dice_sum / present_slices
                    if present_slices else float("nan")
                ),
                "prediction_voxels": totals["prediction"],
                "target_voxels": totals["target"],
            }
        foreground = [str(class_id) for class_id in CLASS_NAMES if class_id]
        for metric_name in (
            "case_dice_presence_mean",
            "global_dice",
            "slice_dice_present_mean",
        ):
            metrics[mode]["foreground_mean_" + metric_name] = float(np.nanmean([
                metrics[mode][class_id][metric_name]
                for class_id in foreground
            ]))
    return metrics, rows


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    test_csv = Path(args.test_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not checkpoint.is_file() or not test_csv.is_file():
        raise FileNotFoundError("checkpoint and test CSV must exist")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({
        "checkpoint": str(checkpoint),
        "test_csv": str(test_csv),
        "output_dir": str(output_dir),
        "test_csv_sha256": sha256(test_csv),
        "class_names": CLASS_NAMES,
        "primary_mode": "argmax_raw",
    })
    atomic_json(output_dir / "resolved_config.json", config)

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this evaluator requires CUDA")
    device = torch.device(args.device)
    model, load_report = build_model(
        "orgslot", checkpoint, fusion_mode=args.fusion_mode
    )
    model.to(device).eval()
    atomic_json(output_dir / "checkpoint_load.json", load_report)

    dataset = AutoPETManifestDataset(test_csv, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    modes = ("argmax_raw", "argmax_post", "binary_raw", "binary_post")
    counters = {
        mode: {class_id: {} for class_id in CLASS_NAMES}
        for mode in modes
    }
    started = time.time()
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].numpy()
            output = model(image)
            calibrated = output["calibrated_logits"]
            argmax = calibrated_multiclass_prediction(
                calibrated, model.slot_names
            ).cpu().numpy()
            binary = {
                class_id: (
                    torch.sigmoid(calibrated[name])[:, 0]
                    > args.binary_threshold
                ).cpu().numpy()
                for class_id, name in CLASS_NAMES.items()
            }
            for sample_index, case_id in enumerate(batch["case_id"]):
                for class_id in CLASS_NAMES:
                    target = labels[sample_index] == class_id
                    raw_predictions = {
                        "argmax_raw": argmax[sample_index] == class_id,
                        "binary_raw": binary[class_id][sample_index],
                    }
                    raw_predictions["argmax_post"] = postprocess_slice(
                        raw_predictions["argmax_raw"],
                        args.min_size,
                        args.opening_radius,
                    )
                    raw_predictions["binary_post"] = postprocess_slice(
                        raw_predictions["binary_raw"],
                        args.min_size,
                        args.opening_radius,
                    )
                    for mode, prediction in raw_predictions.items():
                        counter = counters[mode][class_id].setdefault(
                            case_id, new_counter()
                        )
                        update(counter, prediction, target)
            processed += image.shape[0]
            if batch_index % args.print_freq == 0 or processed == len(dataset):
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-8)
                progress = {
                    "processed": processed,
                    "total": len(dataset),
                    "percent": 100.0 * processed / len(dataset),
                    "elapsed_seconds": elapsed,
                    "slices_per_second": rate,
                    "eta_seconds": (len(dataset) - processed) / max(rate, 1e-8),
                }
                atomic_json(output_dir / "progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)

    metrics, rows = summarize(counters)
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "complete_test_set": args.max_samples is None,
        "samples": len(dataset),
        "elapsed_seconds": time.time() - started,
        "checkpoint_load": load_report,
        "metrics": metrics,
    }
    atomic_json(output_dir / "results.json", result)
    atomic_json(output_dir / "progress.json", {
        "complete": True,
        "processed": len(dataset),
        "total": len(dataset),
        "percent": 100.0,
    })
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
