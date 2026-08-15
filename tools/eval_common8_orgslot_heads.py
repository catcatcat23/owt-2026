#!/usr/bin/env python3
"""Evaluate WORD OrganSlot binary heads with validation-only threshold selection."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from OWT_models_orgslot import FUSION_MODES
from tools.eval_common8_orgslot_reconstruction_threshold import (
    atomic_json,
    build_dataset,
    build_model,
    new_counter,
    parse_class_configuration,
    postprocess_slice,
    sha256,
    summarize,
    update_counter,
)


def parse_args():
    parser = argparse.ArgumentParser("Common8/WORD OrganSlot head evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--class-map", default="configs/orgslot/common8_offline.json"
    )
    parser.add_argument(
        "--split",
        choices=("train_calibration", "validation", "test"),
        required=True,
    )
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--global-crop-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-sweep",
        default="",
        help="validation only; comma-separated probability thresholds",
    )
    parser.add_argument(
        "--class-thresholds",
        help="test only; selected_thresholds.json from validation",
    )
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--class-ids", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--fusion-mode", choices=FUSION_MODES, default="linear_sqrt"
    )
    return parser.parse_args()


def parse_threshold_sweep(value):
    if not value:
        return ()
    thresholds = tuple(sorted(set(float(item) for item in value.split(","))))
    if not thresholds or any(item <= 0 or item >= 1 for item in thresholds):
        raise ValueError("threshold sweep values must lie strictly between 0 and 1")
    return thresholds


def threshold_label(value):
    return ("{:.4f}".format(float(value))).rstrip("0").rstrip(".").replace(".", "p")


def load_class_thresholds(path, class_ids):
    payload = json.loads(Path(path).read_text())
    values = payload.get("thresholds", payload)
    thresholds = {int(key): float(value) for key, value in values.items()}
    if set(thresholds) != set(class_ids):
        raise ValueError("class threshold IDs do not match requested class IDs")
    if any(value <= 0 or value >= 1 for value in thresholds.values()):
        raise ValueError("class thresholds must lie strictly between 0 and 1")
    return thresholds


def select_thresholds(metrics, class_ids, candidates):
    selected = {}
    for class_id in class_ids:
        scored = []
        for threshold in candidates:
            mode = "sweep_{}_post".format(threshold_label(threshold))
            score = metrics[mode][str(class_id)]["case_dice_presence_mean"]
            if np.isfinite(score):
                scored.append((float(score), -abs(threshold - 0.5), threshold))
        if not scored:
            raise RuntimeError("no finite validation score for class {}".format(class_id))
        selected[class_id] = max(scored)[2]
    return selected


def add_volume_ratios(metrics, rows):
    for mode_metrics in metrics.values():
        for key, values in mode_metrics.items():
            if not str(key).isdigit():
                continue
            target = int(values["target_voxels"])
            values["prediction_to_target_volume_ratio"] = (
                float(values["prediction_voxels"]) / target
                if target else float("nan")
            )
    for row in rows:
        target = int(row["target_voxels"])
        row["prediction_to_target_volume_ratio"] = (
            float(row["prediction_voxels"]) / target
            if target else float("nan")
        )


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    data_csv = Path(args.data_csv).resolve()
    class_map = Path(args.class_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    for required in (checkpoint, data_csv, class_map):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.input_size != 448 or args.global_crop_size != 448:
        raise ValueError("formal WORD evaluation requires deterministic 448 crop")
    if not 0 < args.binary_threshold < 1:
        raise ValueError("binary threshold must lie strictly between 0 and 1")
    sweep = parse_threshold_sweep(args.threshold_sweep)
    if sweep and args.split not in ("train_calibration", "validation"):
        raise ValueError("threshold sweep is forbidden on test")
    if args.class_thresholds and args.split != "test":
        raise ValueError("class thresholds are allowed only on test")
    if sweep and args.class_thresholds:
        raise ValueError("threshold sweep and class thresholds are mutually exclusive")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal evaluation requires CUDA")

    slot_specs, class_names = parse_class_configuration(class_map)
    class_ids = tuple(int(value) for value in args.class_ids.split(",") if value)
    if not class_ids or any(value == 0 or value not in class_names for value in class_ids):
        raise ValueError("class_ids must be a non-empty subset of foreground IDs 1..8")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must not contain duplicates")

    default_thresholds = {
        class_id: float(args.binary_threshold) for class_id in class_ids
    }
    threshold_sets = {}
    if sweep:
        for threshold in sweep:
            threshold_sets["sweep_{}".format(threshold_label(threshold))] = {
                class_id: threshold for class_id in class_ids
            }
    else:
        threshold_sets["binary"] = default_thresholds
        if args.class_thresholds:
            threshold_sets["selected"] = load_class_thresholds(
                args.class_thresholds, class_ids
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({
        "checkpoint": str(checkpoint),
        "data_csv": str(data_csv),
        "class_map": str(class_map),
        "output_dir": str(output_dir),
        "checkpoint_sha256": sha256(checkpoint),
        "data_csv_sha256": sha256(data_csv),
        "class_map_sha256": sha256(class_map),
        "class_ids_resolved": class_ids,
        "class_names": class_names,
        "threshold_sets": threshold_sets,
        "threshold_selection_policy": (
            "non-test calibration case Dice only" if sweep else "fixed before this split"
        ),
    })
    atomic_json(output_dir / "resolved_config.json", config)

    device = torch.device(args.device)
    model, load_report = build_model(
        "orgslot",
        checkpoint,
        slot_specs,
        args.input_size,
        args.fusion_mode,
    )
    model.to(device).eval()
    atomic_json(output_dir / "checkpoint_load.json", load_report)
    dataset = build_dataset(
        data_csv,
        args.input_size,
        args.global_crop_size,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    modes = tuple(
        "{}_{}".format(name, suffix)
        for name in threshold_sets
        for suffix in ("raw", "post")
    )
    counters = {
        mode: {class_id: {} for class_id in class_ids}
        for mode in modes
    }
    id_to_name = {
        int(spec["raw_class_id"]): spec["name"] for spec in slot_specs
    }
    started = time.time()
    processed = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["full_label"][:, 0].numpy()
            keep = torch.ones(
                image.shape[0], len(slot_specs), dtype=torch.bool, device=device
            )
            with torch.cuda.amp.autocast(enabled=not args.no_amp):
                output = model(
                    image,
                    slot_keep_mask=keep,
                    decode_reconstruction=False,
                )
                probabilities = {
                    class_id: torch.sigmoid(
                        output["calibrated_logits"][id_to_name[class_id]][:, 0]
                    ).cpu().numpy()
                    for class_id in class_ids
                }
            for sample_index, case_id in enumerate(batch["case_id"]):
                for class_id in class_ids:
                    target = labels[sample_index] == class_id
                    for set_name, class_thresholds in threshold_sets.items():
                        raw = (
                            probabilities[class_id][sample_index]
                            > class_thresholds[class_id]
                        )
                        predictions = {
                            "{}_raw".format(set_name): raw,
                            "{}_post".format(set_name): postprocess_slice(
                                raw, args.min_size, args.opening_radius
                            ),
                        }
                        for mode, prediction in predictions.items():
                            counter = counters[mode][class_id].setdefault(
                                str(case_id), new_counter()
                            )
                            update_counter(counter, prediction, target)
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

    metrics, rows, records = summarize(
        counters, class_ids, class_names, "OrganSlotBank binary heads"
    )
    add_volume_ratios(metrics, rows)
    for record, row in zip(records, rows):
        record["method_family"] = "slot_binary_segmentation_head"
        record["input_privilege"] = "image"
        record["split"] = args.split
        record["prediction_to_target_volume_ratio"] = row[
            "prediction_to_target_volume_ratio"
        ]
    selected = None
    if sweep:
        selected = select_thresholds(metrics, class_ids, sweep)
        atomic_json(output_dir / "selected_thresholds.json", {
            "split": args.split,
            "selection_metric": "case_dice_presence_mean after postprocessing",
            "thresholds": {str(key): value for key, value in selected.items()},
            "candidates": sweep,
        })
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "per_case.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "method": "OrganSlotBank binary heads",
        "dataset": "WORD",
        "split": args.split,
        "samples": len(dataset),
        "complete_split": args.max_samples is None,
        "elapsed_seconds": time.time() - started,
        "checkpoint_load": load_report,
        "selected_thresholds": selected,
        "metrics": metrics,
    }
    atomic_json(output_dir / "results.json", result)
    atomic_json(output_dir / "progress.json", {
        "processed": len(dataset),
        "total": len(dataset),
        "percent": 100.0,
        "complete": True,
    })
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
