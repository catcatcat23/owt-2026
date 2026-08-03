#!/usr/bin/env python3
"""Fair reconstruction-threshold evaluation for OWT and OrganSlotBank.

The legacy score reproduces the original 2D post-processing and empty-slice
policy.  Case-level scores aggregate voxel counts across the 112 slices of a
case and are the preferred reportable metrics.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset

import OWT_models
from OWT_models_orgslot import mae_vit_base_patch16 as build_orgslot


CLASS_NAMES = {
    0: "background",
    1: "kidney_combined",
    2: "spleen",
    3: "pancreas",
    4: "liver",
}


def parse_args():
    parser = argparse.ArgumentParser(
        "AutoPET 2D reconstruction-threshold evaluation"
    )
    parser.add_argument("--method", choices=("owt", "orgslot"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--class-ids", default="0,1,2,3,4")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fusion-mode",
        choices=("post_layernorm", "linear_sqrt"),
        default="post_layernorm",
    )
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=True)
    temporary.replace(path)


def parse_slice_index(path):
    match = re.search(r"_(\d+)\.[^.]+$", Path(path).name)
    if match is None:
        raise ValueError("cannot parse slice index from {}".format(path))
    return int(match.group(1))


class AutoPETManifestDataset(Dataset):
    def __init__(self, manifest, max_samples=None):
        frame = pd.read_csv(manifest)
        if tuple(frame.columns) != ("image_pth", "mask_pth"):
            raise ValueError(
                "expected image_pth,mask_pth columns; got {}".format(
                    list(frame.columns)
                )
            )
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            frame = frame.iloc[:max_samples].copy()
        self.records = frame.to_dict("records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image_path = record["image_pth"]
        mask_path = record["mask_pth"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(image_path)
        if label is None:
            raise FileNotFoundError(mask_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        if label.ndim == 3:
            if not np.array_equal(label[..., 0], label[..., -1]):
                raise ValueError("mask channels disagree: {}".format(mask_path))
            label = label[..., 0]
        observed = set(int(value) for value in np.unique(label))
        unexpected = observed - set(CLASS_NAMES)
        if unexpected:
            raise ValueError(
                "unexpected labels {} in {}".format(sorted(unexpected), mask_path)
            )
        case_id = Path(image_path).parent.name
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "label": torch.from_numpy(label.astype(np.int64)),
            "case_id": case_id,
            "slice_index": parse_slice_index(image_path),
        }


def model_args():
    return SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=20,
        organ_token_total=100,
        fix_frame=0,
        temp_stride=0,
        loss_version=["L2"],
        text_encoding="None",
    )


def build_model(method, checkpoint_path, fusion_mode="post_layernorm"):
    args = model_args()
    if method == "owt":
        model = OWT_models.mae_vit_base_patch16(
            img_size=224,
            norm_pix_loss=False,
            model_args=args,
        )
    else:
        slot_specs = [
            {"name": CLASS_NAMES[class_id], "raw_class_id": class_id}
            for class_id in sorted(CLASS_NAMES)
        ]
        model = build_orgslot(
            img_size=224,
            norm_pix_loss=False,
            model_args=args,
            slot_specs=slot_specs,
            slot_tg_depth=1,
            fusion_mode=fusion_mode,
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    full_state = checkpoint["model"]
    state = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("perceptual_loss.")
    }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint load was not exact: {}".format(result))
    return model, {
        "epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_tensor_count": len(full_state),
        "inference_tensor_count": len(state),
        "stripped_lpips_tensor_count": len(full_state) - len(state),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
    }


@torch.inference_mode()
def reconstruct(model, method, image, kept_class_ids):
    all_class_ids = tuple(sorted(CLASS_NAMES))
    kept_class_ids = tuple(int(value) for value in kept_class_ids)
    if method == "orgslot":
        keep = torch.zeros(
            image.shape[0], len(all_class_ids), dtype=torch.bool, device=image.device
        )
        keep[:, list(kept_class_ids)] = True
        return model(image, slot_keep_mask=keep)["reconstruction"]
    dropped = [value for value in all_class_ids if value not in kept_class_ids]
    middle = {"image_target": image, "random_selected_class": dropped}
    tokens, cls_tokens, middle_output = model.forward_encoder(
        image, mask_ratio=1.0, middle=middle
    )
    patches = model.forward_decoder(tokens, cls_tokens, middle_output)
    return model.unpatchify(patches)


def postprocess_slice(binary, min_size, opening_radius):
    connectivity_one = ndimage.generate_binary_structure(2, 1)
    labeled, _ = ndimage.label(binary, structure=connectivity_one)
    sizes = np.bincount(labeled.ravel())
    large_enough = sizes >= min_size
    large_enough[0] = False
    cleaned = large_enough[labeled]
    if opening_radius > 0:
        coordinates = np.arange(-opening_radius, opening_radius + 1)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        footprint = xx * xx + yy * yy <= opening_radius * opening_radius
        cleaned = ndimage.binary_opening(cleaned, structure=footprint)
    return np.asarray(cleaned, dtype=bool)


def dice_from_counts(intersection, prediction, target):
    denominator = prediction + target
    if denominator == 0:
        return 1.0
    return 2.0 * intersection / denominator


def new_case_counter():
    return {
        "slices": 0,
        "prediction": 0,
        "raw_target": 0,
        "legacy_target": 0,
        "raw_intersection": 0,
        "legacy_intersection": 0,
        "legacy_slice_dice_sum": 0.0,
        "raw_present_slice_dice_sum": 0.0,
        "raw_present_slices": 0,
    }


def update_counter(counter, prediction, raw_target, legacy_target):
    prediction_count = int(prediction.sum())
    raw_count = int(raw_target.sum())
    legacy_count = int(legacy_target.sum())
    raw_intersection = int(np.logical_and(prediction, raw_target).sum())
    legacy_intersection = int(np.logical_and(prediction, legacy_target).sum())
    counter["slices"] += 1
    counter["prediction"] += prediction_count
    counter["raw_target"] += raw_count
    counter["legacy_target"] += legacy_count
    counter["raw_intersection"] += raw_intersection
    counter["legacy_intersection"] += legacy_intersection
    counter["legacy_slice_dice_sum"] += dice_from_counts(
        legacy_intersection, prediction_count, legacy_count
    )
    if raw_count:
        counter["raw_present_slice_dice_sum"] += dice_from_counts(
            raw_intersection, prediction_count, raw_count
        )
        counter["raw_present_slices"] += 1


def summarize(counters, class_ids):
    rows = []
    metrics = {}
    for mode in ("direct", "indirect"):
        metrics[mode] = {}
        for class_id in class_ids:
            class_counters = counters[mode][class_id]
            raw_case_dice = []
            legacy_case_dice = []
            raw_global = {"intersection": 0, "prediction": 0, "target": 0}
            legacy_global = {"intersection": 0, "prediction": 0, "target": 0}
            legacy_slice_sum = 0.0
            slice_count = 0
            raw_present_slice_sum = 0.0
            raw_present_slice_count = 0
            for case_id in sorted(class_counters):
                counter = class_counters[case_id]
                raw_dice = dice_from_counts(
                    counter["raw_intersection"],
                    counter["prediction"],
                    counter["raw_target"],
                )
                legacy_dice = dice_from_counts(
                    counter["legacy_intersection"],
                    counter["prediction"],
                    counter["legacy_target"],
                )
                if counter["raw_target"]:
                    raw_case_dice.append(raw_dice)
                if counter["legacy_target"]:
                    legacy_case_dice.append(legacy_dice)
                raw_global["intersection"] += counter["raw_intersection"]
                raw_global["prediction"] += counter["prediction"]
                raw_global["target"] += counter["raw_target"]
                legacy_global["intersection"] += counter["legacy_intersection"]
                legacy_global["prediction"] += counter["prediction"]
                legacy_global["target"] += counter["legacy_target"]
                legacy_slice_sum += counter["legacy_slice_dice_sum"]
                slice_count += counter["slices"]
                raw_present_slice_sum += counter["raw_present_slice_dice_sum"]
                raw_present_slice_count += counter["raw_present_slices"]
                rows.append(
                    {
                        "mode": mode,
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "case_id": case_id,
                        "slices": counter["slices"],
                        "prediction_voxels": counter["prediction"],
                        "raw_target_voxels": counter["raw_target"],
                        "legacy_target_voxels": counter["legacy_target"],
                        "raw_case_dice": raw_dice,
                        "legacy_case_dice": legacy_dice,
                    }
                )
            metrics[mode][str(class_id)] = {
                "class_name": CLASS_NAMES[class_id],
                "case_count": len(class_counters),
                "raw_gt_present_case_count": len(raw_case_dice),
                "legacy_gt_present_case_count": len(legacy_case_dice),
                "raw_case_dice_presence_mean": (
                    float(np.mean(raw_case_dice)) if raw_case_dice else float("nan")
                ),
                "legacy_case_dice_presence_mean": (
                    float(np.mean(legacy_case_dice))
                    if legacy_case_dice
                    else float("nan")
                ),
                "raw_global_dice": dice_from_counts(
                    raw_global["intersection"],
                    raw_global["prediction"],
                    raw_global["target"],
                ),
                "legacy_global_dice": dice_from_counts(
                    legacy_global["intersection"],
                    legacy_global["prediction"],
                    legacy_global["target"],
                ),
                "legacy_slice_dice_mean_empty_is_one": (
                    legacy_slice_sum / slice_count if slice_count else float("nan")
                ),
                "raw_slice_dice_present_mean": (
                    raw_present_slice_sum / raw_present_slice_count
                    if raw_present_slice_count
                    else float("nan")
                ),
            }
        foreground = [str(value) for value in class_ids if value != 0]
        for metric_name in (
            "raw_case_dice_presence_mean",
            "legacy_case_dice_presence_mean",
            "raw_global_dice",
            "legacy_global_dice",
            "legacy_slice_dice_mean_empty_is_one",
            "raw_slice_dice_present_mean",
        ):
            values = [metrics[mode][value][metric_name] for value in foreground]
            metrics[mode]["foreground_mean_" + metric_name] = float(
                np.mean(values)
            )
    return metrics, rows


def main():
    args = parse_args()
    class_ids = tuple(int(value) for value in args.class_ids.split(","))
    if not class_ids or any(value not in CLASS_NAMES for value in class_ids):
        raise ValueError("class_ids must be a non-empty subset of 0,1,2,3,4")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must not contain duplicates")
    checkpoint_path = Path(args.checkpoint).resolve()
    test_csv = Path(args.test_csv).resolve()
    if not checkpoint_path.is_file() or not test_csv.is_file():
        raise FileNotFoundError("checkpoint and test CSV must exist")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    config = vars(args).copy()
    config.update(
        {
            "checkpoint": str(checkpoint_path),
            "test_csv": str(test_csv),
            "output_dir": str(output_dir),
            "test_csv_sha256": sha256(test_csv),
            "class_ids_resolved": class_ids,
            "class_names": CLASS_NAMES,
            "legacy_protocol": {
                "threshold": args.threshold,
                "remove_small_objects_min_size": args.min_size,
                "binary_opening_disk_radius": args.opening_radius,
                "empty_slice_dice": 1.0,
            },
        }
    )
    atomic_json(output_dir / "resolved_config.json", config)

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("this evaluator requires CUDA")
    device = torch.device(args.device)
    model, load_report = build_model(
        args.method, checkpoint_path, fusion_mode=args.fusion_mode
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
    counters = {
        mode: {class_id: {} for class_id in class_ids}
        for mode in ("direct", "indirect")
    }
    all_class_ids = tuple(sorted(CLASS_NAMES))
    processed = 0
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].numpy()
        image_mean = image.mean(dim=1).cpu().numpy()
        batch_predictions = {"direct": {}, "indirect": {}}
        for class_id in class_ids:
            direct = reconstruct(model, args.method, image, (class_id,))
            batch_predictions["direct"][class_id] = (
                direct.mean(dim=1) > args.threshold
            ).cpu().numpy()
            indirect_keep = tuple(
                value for value in all_class_ids if value != class_id
            )
            complement = reconstruct(model, args.method, image, indirect_keep)
            residual = (image - complement).clamp_min_(0)
            batch_predictions["indirect"][class_id] = (
                residual.mean(dim=1) > args.threshold
            ).cpu().numpy()

        for sample_index, case_id in enumerate(batch["case_id"]):
            for class_id in class_ids:
                raw_target = label[sample_index] == class_id
                legacy_target = raw_target & (
                    image_mean[sample_index] > args.threshold
                )
                for mode in ("direct", "indirect"):
                    prediction = postprocess_slice(
                        batch_predictions[mode][class_id][sample_index],
                        args.min_size,
                        args.opening_radius,
                    )
                    case_counters = counters[mode][class_id]
                    counter = case_counters.setdefault(case_id, new_case_counter())
                    update_counter(counter, prediction, raw_target, legacy_target)
        processed += image.shape[0]
        if batch_index % args.print_freq == 0 or processed == len(dataset):
            elapsed = time.time() - started
            rate = processed / max(elapsed, 1e-8)
            progress = {
                "method": args.method,
                "processed": processed,
                "total": len(dataset),
                "percent": 100.0 * processed / len(dataset),
                "elapsed_seconds": elapsed,
                "slices_per_second": rate,
                "eta_seconds": (len(dataset) - processed) / max(rate, 1e-8),
            }
            atomic_json(output_dir / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)

    metrics, rows = summarize(counters, class_ids)
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed = {
        "method": args.method,
        "samples": len(dataset),
        "complete_test_set": args.max_samples is None,
        "elapsed_seconds": time.time() - started,
        "checkpoint_load": load_report,
        "metrics": metrics,
    }
    atomic_json(output_dir / "results.json", completed)
    atomic_json(
        output_dir / "progress.json",
        {
            "method": args.method,
            "processed": len(dataset),
            "total": len(dataset),
            "percent": 100.0,
            "complete": True,
        },
    )
    print(json.dumps(completed, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
