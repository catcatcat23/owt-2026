#!/usr/bin/env python3
"""Case-level 3D evaluation for WORD OrganSlot short-slab checkpoints.

The model consumes overlapping short slabs.  This evaluator averages calibrated
foreground probabilities for every original slice, reconstructs one complete
volume per case, and only then thresholds, postprocesses, and computes metrics.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Subset

from datasets.orgslot_highres import OrganSlotHighResTransform, TransformDataset
from datasets.orgslot_manifest import OrganSlotManifestDataset, parse_case_and_slice
from OWT_models_orgslot import FUSION_MODES, mae_vit_base_patch16 as build_orgslot
from tools.eval_common8_orgslot_reconstruction_threshold import (
    atomic_json,
    checkpoint_value,
    parse_class_configuration,
    sha256,
)
from util.label_visibility import EvaluationVisibilityDataset


PRIMARY_MODE = "head_post3d"


def parse_args():
    parser = argparse.ArgumentParser("Common8/WORD OrganSlot true-3D head evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--preprocess-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--class-map", default="configs/orgslot/common8_offline.json")
    parser.add_argument("--expected-spacing", nargs=3, type=float, required=True)
    parser.add_argument("--expected-cases", type=int, default=24)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--global-crop-size", type=int, default=448)
    parser.add_argument("--fix-frame", type=int, default=4)
    parser.add_argument("--temp-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument(
        "--class-thresholds",
        help="optional thresholds selected without using the test set",
    )
    parser.add_argument("--min-component-voxels", type=int, default=20)
    parser.add_argument("--opening-radius-mm", type=float, default=0.0)
    parser.add_argument("--nsd-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--class-ids", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--fusion-mode", choices=FUSION_MODES, default="linear_sqrt")
    return parser.parse_args()


def _checkpoint_args(token_factor, slot_count, fix_frame, temp_stride, arch_version):
    return SimpleNamespace(
        LA=True,
        arch_version=str(arch_version),
        dataset_type="3D",
        token_factor=int(token_factor),
        organ_token_total=int(token_factor) * int(slot_count),
        fix_frame=int(fix_frame),
        temp_stride=int(temp_stride),
        loss_version=["L2"],
        text_encoding="None",
    )


def build_3d_model(checkpoint_path, slot_specs, input_size, fusion_mode, fix_frame, temp_stride):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dimension = checkpoint_value(checkpoint, "dimension")
    dataset_type = checkpoint_value(checkpoint, "dataset_type")
    if dimension != "3D" and dataset_type != "3D":
        raise ValueError("checkpoint is not a 3D OrganSlot model")
    saved_frame = int(checkpoint_value(checkpoint, "fix_frame", -1))
    saved_stride = int(checkpoint_value(checkpoint, "temp_stride", -1))
    if saved_frame != int(fix_frame) or saved_stride != int(temp_stride):
        raise ValueError(
            "checkpoint slab geometry ({}, {}) != requested ({}, {})".format(
                saved_frame, saved_stride, fix_frame, temp_stride
            )
        )
    saved_input = int(checkpoint_value(checkpoint, "input_size", -1))
    if saved_input != int(input_size):
        raise ValueError("checkpoint input_size {} != {}".format(saved_input, input_size))
    saved_fusion = checkpoint_value(checkpoint, "fusion_mode")
    if saved_fusion is not None and saved_fusion != fusion_mode:
        raise ValueError("checkpoint fusion_mode {} != {}".format(saved_fusion, fusion_mode))

    token_factor = int(checkpoint_value(checkpoint, "token_factor", 20))
    slot_tg_depth = int(checkpoint_value(checkpoint, "slot_tg_depth", 1))
    slot_head_type = checkpoint_value(checkpoint, "slot_head_type", "linear")
    slot_head_channels = int(checkpoint_value(checkpoint, "slot_head_channels", 128))
    fusion_reference_count = int(
        checkpoint_value(checkpoint, "fusion_reference_count", len(slot_specs))
    )
    arch_version = checkpoint_value(checkpoint, "arch_version", "v11")
    model_args = _checkpoint_args(
        token_factor, len(slot_specs), fix_frame, temp_stride, arch_version
    )
    model = build_orgslot(
        img_size=input_size,
        norm_pix_loss=False,
        model_args=model_args,
        slot_specs=slot_specs,
        slot_tg_depth=slot_tg_depth,
        fusion_mode=fusion_mode,
        fusion_reference_count=fusion_reference_count,
        slot_head_type=slot_head_type,
        slot_head_channels=slot_head_channels,
    )
    full_state = checkpoint["model"]
    state = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("perceptual_loss.")
    }
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError("checkpoint load was not exact: {}".format(loaded))
    report = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "exact": True,
        "dimension": "3D",
        "fix_frame": saved_frame,
        "temp_stride": saved_stride,
        "input_size": saved_input,
        "fusion_mode": fusion_mode,
        "slot_head_type": slot_head_type,
        "slot_head_channels": slot_head_channels,
        "token_factor": token_factor,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_tensor_count": len(full_state),
        "inference_tensor_count": len(state),
        "stripped_lpips_tensor_count": len(full_state) - len(state),
    }
    return model, report


def build_3d_dataset(test_csv, input_size, global_crop_size, fix_frame):
    raw = OrganSlotManifestDataset(
        test_csv,
        dataset_type="3D",
        fix_frame=fix_frame,
        intensity_norm="fixed_255",
        expected_size=None,
    )
    transform = OrganSlotHighResTransform(
        output_size=input_size,
        training=False,
        global_crop_size=global_crop_size,
        roi_crop_size=global_crop_size,
        roi_center_jitter=0.0,
    )
    dataset = EvaluationVisibilityDataset(TransformDataset(raw, transform))
    case_to_dataset_indices = {}
    case_to_slices = {}
    for dataset_index, record_index in enumerate(raw.sample_record_indices):
        case_id, slice_index = parse_case_and_slice(raw.records[record_index]["image_pth"])
        case_to_dataset_indices.setdefault(case_id, []).append(dataset_index)
        case_to_slices.setdefault(case_id, []).append(slice_index)
    for case_id, slice_indices in case_to_slices.items():
        if len(slice_indices) != len(set(slice_indices)):
            raise ValueError("duplicate slice indices in case {}".format(case_id))
        case_to_slices[case_id] = tuple(sorted(slice_indices))
    return dataset, case_to_dataset_indices, case_to_slices


def load_thresholds(path, class_ids, default_threshold):
    if path is None:
        return (
            {class_id: float(default_threshold) for class_id in class_ids},
            "fixed_{:g}".format(default_threshold),
        )
    payload = json.loads(Path(path).read_text())
    values = payload.get("thresholds", payload)
    thresholds = {int(key): float(value) for key, value in values.items()}
    if set(thresholds) != set(class_ids):
        raise ValueError("class threshold IDs do not match requested class IDs")
    if any(not 0.0 < value < 1.0 for value in thresholds.values()):
        raise ValueError("class thresholds must lie strictly between zero and one")
    return thresholds, "external_non_test_calibration"


def accumulate_slab(probability_sum, counts, target_volume, slice_to_position,
                    slice_indices, probabilities, labels):
    """Accumulate one slab; exposed separately for deterministic unit testing."""
    for slab_position, raw_slice_index in enumerate(slice_indices):
        volume_position = slice_to_position[int(raw_slice_index)]
        probability_sum[:, volume_position] += probabilities[:, slab_position]
        counts[volume_position] += 1
        label_slice = labels[slab_position].astype(np.uint8, copy=False)
        if target_volume[volume_position] is None:
            target_volume[volume_position] = label_slice.copy()
        elif not np.array_equal(target_volume[volume_position], label_slice):
            raise ValueError("inconsistent target for overlapping slab slice")


def _physical_ball(radius_mm, spacing_zyx):
    extents = [int(math.ceil(radius_mm / spacing)) for spacing in spacing_zyx]
    grids = np.ogrid[tuple(slice(-extent, extent + 1) for extent in extents)]
    distance_squared = sum(
        (grid * spacing) ** 2 for grid, spacing in zip(grids, spacing_zyx)
    )
    return distance_squared <= radius_mm ** 2 + 1e-9


def postprocess_volume(binary, min_component_voxels, opening_radius_mm, spacing_zyx):
    binary = np.asarray(binary, dtype=bool)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, _ = ndimage.label(binary, structure=structure)
    sizes = np.bincount(labeled.ravel())
    retained = sizes >= int(min_component_voxels)
    retained[0] = False
    cleaned = retained[labeled]
    if opening_radius_mm > 0:
        cleaned = ndimage.binary_opening(
            cleaned,
            structure=_physical_ball(opening_radius_mm, spacing_zyx),
        )
    return np.asarray(cleaned, dtype=bool)


def overlap_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    prediction_count = int(prediction.sum())
    target_count = int(target.sum())
    intersection = int(np.logical_and(prediction, target).sum())
    union = prediction_count + target_count - intersection
    denominator = prediction_count + target_count
    return {
        "prediction_voxels": prediction_count,
        "target_voxels": target_count,
        "intersection_voxels": intersection,
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "iou": 1.0 if union == 0 else intersection / union,
        "precision": (
            1.0 if prediction_count == 0 and target_count == 0
            else 0.0 if prediction_count == 0
            else intersection / prediction_count
        ),
        "recall": (
            1.0 if target_count == 0 and prediction_count == 0
            else 0.0 if target_count == 0
            else intersection / target_count
        ),
    }


def surface_metrics(prediction, target, spacing_zyx, tolerance_mm):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if not prediction.any() and not target.any():
        return {"nsd": 1.0, "hd95_mm": 0.0}
    if not prediction.any() or not target.any():
        return {"nsd": 0.0, "hd95_mm": float("inf")}

    union_coordinates = np.nonzero(np.logical_or(prediction, target))
    slices = tuple(
        slice(max(0, int(values.min()) - 1), min(size, int(values.max()) + 2))
        for values, size in zip(union_coordinates, prediction.shape)
    )
    prediction = prediction[slices]
    target = target[slices]
    structure = ndimage.generate_binary_structure(3, 1)
    prediction_surface = np.logical_and(
        prediction, np.logical_not(ndimage.binary_erosion(prediction, structure=structure))
    )
    target_surface = np.logical_and(
        target, np.logical_not(ndimage.binary_erosion(target, structure=structure))
    )
    distance_to_target = ndimage.distance_transform_edt(
        np.logical_not(target_surface), sampling=spacing_zyx
    )
    distance_to_prediction = ndimage.distance_transform_edt(
        np.logical_not(prediction_surface), sampling=spacing_zyx
    )
    prediction_distances = distance_to_target[prediction_surface]
    target_distances = distance_to_prediction[target_surface]
    all_distances = np.concatenate((prediction_distances, target_distances))
    within = int((prediction_distances <= tolerance_mm).sum())
    within += int((target_distances <= tolerance_mm).sum())
    return {
        "nsd": within / len(all_distances),
        "hd95_mm": float(np.percentile(all_distances, 95.0)),
    }


def summarize(rows, class_ids, class_names):
    metrics = {}
    foreground_values = {
        key: [] for key in (
            "case_dice_presence_mean", "case_iou_presence_mean",
            "case_precision_presence_mean", "case_recall_presence_mean",
            "case_nsd_presence_mean", "case_hd95_finite_mean_mm",
        )
    }
    for class_id in class_ids:
        class_rows = [row for row in rows if row["class_id"] == class_id]
        present = [row for row in class_rows if row["target_voxels"] > 0]
        finite_hd95 = [row["hd95_mm"] for row in present if np.isfinite(row["hd95_mm"])]
        prediction = sum(row["prediction_voxels"] for row in class_rows)
        target = sum(row["target_voxels"] for row in class_rows)
        intersection = sum(row["intersection_voxels"] for row in class_rows)
        union = prediction + target - intersection
        values = {
            "class_name": class_names[class_id],
            "case_count": len(class_rows),
            "gt_present_case_count": len(present),
            "case_dice_presence_mean": float(np.mean([row["dice"] for row in present])),
            "case_iou_presence_mean": float(np.mean([row["iou"] for row in present])),
            "case_precision_presence_mean": float(np.mean([row["precision"] for row in present])),
            "case_recall_presence_mean": float(np.mean([row["recall"] for row in present])),
            "case_nsd_presence_mean": float(np.mean([row["nsd"] for row in present])),
            "case_hd95_presence_mean_mm": float(np.mean([row["hd95_mm"] for row in present])),
            "case_hd95_finite_mean_mm": float(np.mean(finite_hd95)) if finite_hd95 else float("inf"),
            "hd95_infinite_case_count": sum(not np.isfinite(row["hd95_mm"]) for row in present),
            "global_dice": 1.0 if prediction + target == 0 else 2.0 * intersection / (prediction + target),
            "global_iou": 1.0 if union == 0 else intersection / union,
            "prediction_voxels": prediction,
            "target_voxels": target,
            "prediction_to_target_volume_ratio": float(prediction / target) if target else float("nan"),
        }
        metrics[str(class_id)] = values
        for key in foreground_values:
            foreground_values[key].append(values[key])
    metrics["foreground_macro"] = {
        key: float(np.mean(values)) for key, values in foreground_values.items()
    }
    return metrics


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    test_csv = Path(args.test_csv).resolve()
    preprocess_summary = Path(args.preprocess_summary).resolve()
    class_map = Path(args.class_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    required = (checkpoint_path, test_csv, preprocess_summary, class_map)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal 3D evaluation requires CUDA")
    if args.input_size != 448 or args.global_crop_size != 448:
        raise ValueError("formal WORD 3D evaluation requires deterministic 448 crop")
    if args.fix_frame <= 0 or args.temp_stride <= 0:
        raise ValueError("invalid slab geometry")
    if not 0.0 < args.binary_threshold < 1.0:
        raise ValueError("binary threshold must lie strictly between zero and one")
    if args.min_component_voxels < 0 or args.opening_radius_mm < 0:
        raise ValueError("postprocessing parameters must be non-negative")
    if args.nsd_tolerance_mm <= 0:
        raise ValueError("NSD tolerance must be positive")

    preprocess = json.loads(preprocess_summary.read_text())
    recorded_spacing = preprocess.get("config", {}).get("spacing_mm")
    if recorded_spacing is None or not np.allclose(
        recorded_spacing, args.expected_spacing, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "preprocessing spacing {} != expected {}".format(
                recorded_spacing, list(args.expected_spacing)
            )
        )
    spacing_zyx = tuple(float(value) for value in args.expected_spacing[::-1])

    slot_specs, class_names = parse_class_configuration(class_map)
    class_ids = tuple(int(value) for value in args.class_ids.split(",") if value)
    if not class_ids or any(value == 0 or value not in class_names for value in class_ids):
        raise ValueError("class_ids must be a non-empty subset of foreground IDs")
    if len(class_ids) != len(set(class_ids)):
        raise ValueError("class_ids must not contain duplicates")
    thresholds, threshold_policy = load_thresholds(
        args.class_thresholds, class_ids, args.binary_threshold
    )

    dataset, case_to_indices, case_to_slices = build_3d_dataset(
        test_csv, args.input_size, args.global_crop_size, args.fix_frame
    )
    if len(case_to_indices) != args.expected_cases:
        raise ValueError(
            "test manifest has {} cases, expected {}".format(
                len(case_to_indices), args.expected_cases
            )
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update({
        "checkpoint": str(checkpoint_path),
        "test_csv": str(test_csv),
        "preprocess_summary": str(preprocess_summary),
        "class_map": str(class_map),
        "output_dir": str(output_dir),
        "checkpoint_sha256": sha256(checkpoint_path),
        "test_csv_sha256": sha256(test_csv),
        "preprocess_summary_sha256": sha256(preprocess_summary),
        "class_map_sha256": sha256(class_map),
        "class_ids_resolved": class_ids,
        "thresholds_resolved": thresholds,
        "threshold_policy": threshold_policy,
        "primary_mode": PRIMARY_MODE,
        "slab_fusion": "mean calibrated probability per original slice",
        "metric_geometry": "complete case volume after deterministic center crop",
        "spacing_zyx_mm": spacing_zyx,
        "surface_distance_implementation": "symmetric voxel-surface EDT in physical mm",
    })
    atomic_json(output_dir / "resolved_config.json", config)

    device = torch.device(args.device)
    model, load_report = build_3d_model(
        checkpoint_path, slot_specs, args.input_size, args.fusion_mode,
        args.fix_frame, args.temp_stride
    )
    model.to(device).eval()
    atomic_json(output_dir / "checkpoint_load.json", load_report)
    id_to_name = {int(spec["raw_class_id"]): spec["name"] for spec in slot_specs}

    rows = []
    started = time.time()
    case_ids_ordered = sorted(case_to_indices)
    with torch.inference_mode():
        for case_number, case_id in enumerate(case_ids_ordered, start=1):
            slice_indices = case_to_slices[case_id]
            slice_to_position = {value: index for index, value in enumerate(slice_indices)}
            depth = len(slice_indices)
            probability_sum = np.zeros(
                (len(class_ids), depth, args.input_size, args.input_size), dtype=np.float32
            )
            counts = np.zeros(depth, dtype=np.uint16)
            target_slices = [None] * depth
            loader = DataLoader(
                Subset(dataset, case_to_indices[case_id]),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=False,
            )
            for batch_number, batch in enumerate(loader):
                if any(str(value) != case_id for value in batch["case_id"]):
                    raise ValueError("case-mixed evaluation batch")
                image = batch["image"].to(device, non_blocking=True)
                labels = batch["full_label"][:, 0].numpy()
                keep = torch.ones(
                    image.shape[0], len(slot_specs), dtype=torch.bool, device=device
                )
                with torch.cuda.amp.autocast(enabled=not args.no_amp):
                    output = model(
                        image, slot_keep_mask=keep,
                        decode_reconstruction=False, decode_heads=True
                    )
                    probabilities = torch.stack(
                        [
                            torch.sigmoid(output["calibrated_logits"][id_to_name[class_id]][:, 0])
                            for class_id in class_ids
                        ],
                        dim=1,
                    ).float().cpu().numpy()
                slab_indices = batch["slice_indices"].numpy()
                for sample_index in range(image.shape[0]):
                    accumulate_slab(
                        probability_sum, counts, target_slices, slice_to_position,
                        slab_indices[sample_index], probabilities[sample_index],
                        labels[sample_index],
                    )
                if batch_number % args.print_freq == 0:
                    print(json.dumps({
                        "case": case_id,
                        "case_number": case_number,
                        "cases_total": len(case_ids_ordered),
                        "slabs_processed": min(
                            (batch_number + 1) * args.batch_size,
                            len(case_to_indices[case_id]),
                        ),
                    }, sort_keys=True), flush=True)
            if np.any(counts == 0):
                raise RuntimeError("case {} has uncovered slices".format(case_id))
            probabilities = probability_sum / counts[None, :, None, None]
            target_volume = np.stack(target_slices, axis=0)
            for class_offset, class_id in enumerate(class_ids):
                target = target_volume == class_id
                raw = probabilities[class_offset] > thresholds[class_id]
                prediction = postprocess_volume(
                    raw, args.min_component_voxels,
                    args.opening_radius_mm, spacing_zyx
                )
                overlap = overlap_metrics(prediction, target)
                surfaces = surface_metrics(
                    prediction, target, spacing_zyx, args.nsd_tolerance_mm
                )
                row = {
                    "case_id": case_id,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "mode": PRIMARY_MODE,
                    "slices": depth,
                    "threshold": thresholds[class_id],
                    **overlap,
                    **surfaces,
                }
                row["prediction_to_target_volume_ratio"] = (
                    row["prediction_voxels"] / row["target_voxels"]
                    if row["target_voxels"] else float("nan")
                )
                rows.append(row)
            atomic_json(output_dir / "progress.json", {
                "processed_cases": case_number,
                "total_cases": len(case_ids_ordered),
                "percent": 100.0 * case_number / len(case_ids_ordered),
                "complete": False,
                "elapsed_seconds": time.time() - started,
            })

    metrics = summarize(rows, class_ids, class_names)
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "per_case.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
    result = {
        "method": "OrganSlotBank 3D slab head",
        "dataset": "WORD",
        "split": "test",
        "cases": len(case_ids_ordered),
        "slab_anchors": len(dataset),
        "complete_test_set": True,
        "primary_mode": PRIMARY_MODE,
        "elapsed_seconds": time.time() - started,
        "checkpoint_load": load_report,
        "metrics": metrics,
    }
    atomic_json(output_dir / "results.json", result)
    atomic_json(output_dir / "progress.json", {
        "processed_cases": len(case_ids_ordered),
        "total_cases": len(case_ids_ordered),
        "percent": 100.0,
        "complete": True,
        "elapsed_seconds": time.time() - started,
    })
    print(json.dumps(result, sort_keys=True, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
