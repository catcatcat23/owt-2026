#!/usr/bin/env python3
"""Evaluate Common8 OWT/OrganSlot checkpoints by reconstruction thresholding.

The primary metric is case-level 3D Dice over the unmodified ground-truth mask.
The evaluator uses the deterministic center crop from training validation and
never uses labels to select a crop or threshold.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
import torch
from torch.utils.data import DataLoader

from datasets.orgslot_highres import OrganSlotHighResTransform, TransformDataset
from datasets.orgslot_manifest import OrganSlotManifestDataset
import OWT_models
from OWT_models_orgslot import (
    FUSION_MODES,
    mae_vit_base_patch16 as build_orgslot,
)
from util.label_visibility import EvaluationVisibilityDataset, load_visibility_config


PRIMARY_MODE = "direct_post"


def parse_args():
    parser = argparse.ArgumentParser(
        "Common8/WORD OWT reconstruction-threshold evaluation"
    )
    parser.add_argument(
        "--method", choices=("owt", "orgslot"), default="orgslot"
    )
    parser.add_argument("--preprocess-summary")
    parser.add_argument(
        "--expected-spacing",
        nargs=3,
        type=float,
        default=None,
        metavar=("SX", "SY", "SZ"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--class-map", default="configs/orgslot/common8_offline.json"
    )
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--global-crop-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--min-size", type=int, default=20)
    parser.add_argument("--opening-radius", type=int, default=1)
    parser.add_argument("--class-ids", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--fusion-mode",
        choices=FUSION_MODES,
        default="linear_sqrt",
    )
    parser.add_argument(
        "--allow-fusion-override",
        action="store_true",
        help="diagnostic only: evaluate a checkpoint with a different fusion rule",
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


def parse_class_configuration(path):
    classes, stages = load_visibility_config(path)
    stage = stages["base"]
    by_name = {item.name: item for item in classes}
    slot_specs = [
        {"name": name, "raw_class_id": int(by_name[name].raw_id)}
        for name in stage.visible_slots
    ]
    class_names = {
        int(item.raw_id): item.name
        for item in classes
        if item.name in stage.visible_slots
    }
    if sorted(class_names) != list(range(9)):
        raise ValueError("Common8 requires contiguous raw IDs 0 through 8")
    return slot_specs, class_names


def model_args(token_factor, slot_count):
    return SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=int(token_factor),
        organ_token_total=int(token_factor) * int(slot_count),
        fix_frame=0,
        temp_stride=0,
        loss_version=["L2"],
        text_encoding="None",
    )


def checkpoint_value(checkpoint, name, default=None):
    if name in checkpoint:
        return checkpoint[name]
    saved_args = checkpoint.get("args")
    if saved_args is None:
        return default
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def build_model(
    method,
    checkpoint_path,
    slot_specs,
    input_size,
    fusion_mode,
    allow_fusion_override=False,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_input_size = checkpoint_value(checkpoint, "input_size")
    if saved_input_size is not None and int(saved_input_size) != int(input_size):
        raise ValueError(
            "checkpoint input_size {} != requested {}".format(
                saved_input_size, input_size
            )
        )
    saved_fusion = checkpoint_value(checkpoint, "fusion_mode")
    fusion_overridden = (
        method == "orgslot"
        and saved_fusion is not None
        and saved_fusion != fusion_mode
    )
    if fusion_overridden and not allow_fusion_override:
        raise ValueError(
            "checkpoint fusion_mode {} != requested {}; pass "
            "--allow-fusion-override only for a diagnostic ablation".format(
                saved_fusion, fusion_mode
            )
        )

    token_factor = int(checkpoint_value(checkpoint, "token_factor", 20))
    slot_tg_depth = int(checkpoint_value(checkpoint, "slot_tg_depth", 1))
    slot_head_type = checkpoint_value(checkpoint, "slot_head_type", "linear")
    slot_head_channels = int(
        checkpoint_value(checkpoint, "slot_head_channels", 128)
    )
    fusion_reference_count = int(
        checkpoint_value(checkpoint, "fusion_reference_count", len(slot_specs))
    )
    args = model_args(token_factor, len(slot_specs))
    if method == "orgslot":
        model = build_orgslot(
            img_size=input_size,
            norm_pix_loss=False,
            model_args=args,
            slot_specs=slot_specs,
            slot_tg_depth=slot_tg_depth,
            fusion_mode=fusion_mode,
            fusion_reference_count=fusion_reference_count,
            slot_head_type=slot_head_type,
            slot_head_channels=slot_head_channels,
        )
    else:
        model = OWT_models.mae_vit_base_patch16(
            img_size=input_size,
            norm_pix_loss=False,
            model_args=args,
        )
    full_state = checkpoint["model"]
    state = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("perceptual_loss.")
    }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint load was not exact: {}".format(result))
    report = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "optimizer_updates": int(checkpoint.get("optimizer_updates", -1)),
        "checkpoint_tensor_count": len(full_state),
        "inference_tensor_count": len(state),
        "stripped_lpips_tensor_count": len(full_state) - len(state),
        "model_parameter_count": sum(p.numel() for p in model.parameters()),
        "input_size": int(input_size),
        "method": method,
        "fusion_mode": fusion_mode if method == "orgslot" else None,
        "checkpoint_fusion_mode": saved_fusion,
        "fusion_overridden": fusion_overridden,
        "fusion_reference_count": (
            fusion_reference_count if method == "orgslot" else None
        ),
        "token_factor": token_factor,
        "slot_tg_depth": (
            slot_tg_depth if method == "orgslot" else len(model.blocks2)
        ),
        "slot_head_type": slot_head_type if method == "orgslot" else None,
        "slot_head_channels": slot_head_channels if method == "orgslot" else None,
        "exact": True,
    }
    return model, report


def build_dataset(test_csv, input_size, global_crop_size, max_samples=None):
    raw = OrganSlotManifestDataset(
        test_csv,
        dataset_type="2D",
        intensity_norm="fixed_255",
        expected_size=None,
        max_samples=max_samples,
    )
    transform = OrganSlotHighResTransform(
        output_size=input_size,
        training=False,
        global_crop_size=global_crop_size,
        roi_crop_size=global_crop_size,
        roi_center_jitter=0.0,
    )
    return EvaluationVisibilityDataset(TransformDataset(raw, transform))


def postprocess_slice(binary, min_size, opening_radius):
    binary = np.asarray(binary, dtype=bool)
    connectivity_one = ndimage.generate_binary_structure(2, 1)
    labeled, _ = ndimage.label(binary, structure=connectivity_one)
    sizes = np.bincount(labeled.ravel())
    large_enough = sizes >= int(min_size)
    large_enough[0] = False
    cleaned = large_enough[labeled]
    if opening_radius > 0:
        coordinates = np.arange(-opening_radius, opening_radius + 1)
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        footprint = xx * xx + yy * yy <= opening_radius * opening_radius
        cleaned = ndimage.binary_opening(cleaned, structure=footprint)
    return np.asarray(cleaned, dtype=bool)


def dice_from_counts(intersection, prediction, target):
    denominator = int(prediction) + int(target)
    return 1.0 if denominator == 0 else 2.0 * int(intersection) / denominator


def iou_from_counts(intersection, prediction, target):
    union = int(prediction) + int(target) - int(intersection)
    return 1.0 if union == 0 else int(intersection) / union


def new_counter():
    return {
        "slices": 0,
        "prediction": 0,
        "target": 0,
        "intersection": 0,
        "present_slice_dice_sum": 0.0,
        "present_slices": 0,
    }


def update_counter(counter, prediction, target):
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


def summarize(counters, class_ids, class_names, method_name):
    metrics = {}
    rows = []
    records = []
    for mode, mode_counters in counters.items():
        metrics[mode] = {}
        for class_id in class_ids:
            present_case_dice = []
            present_case_iou = []
            all_case_dice = []
            totals = {"intersection": 0, "prediction": 0, "target": 0}
            present_slice_dice_sum = 0.0
            present_slices = 0
            for case_id in sorted(mode_counters[class_id]):
                counter = mode_counters[class_id][case_id]
                dice = dice_from_counts(
                    counter["intersection"], counter["prediction"], counter["target"]
                )
                iou = iou_from_counts(
                    counter["intersection"], counter["prediction"], counter["target"]
                )
                all_case_dice.append(dice)
                if counter["target"]:
                    present_case_dice.append(dice)
                    present_case_iou.append(iou)
                for key in totals:
                    totals[key] += counter[key]
                present_slice_dice_sum += counter["present_slice_dice_sum"]
                present_slices += counter["present_slices"]
                row = {
                    "mode": mode,
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "case_id": case_id,
                    "slices": counter["slices"],
                    "prediction_voxels": counter["prediction"],
                    "target_voxels": counter["target"],
                    "intersection_voxels": counter["intersection"],
                    "case_dice": dice,
                    "case_iou": iou,
                    "prediction_to_target_volume_ratio": (
                        float(counter["prediction"]) / counter["target"]
                        if counter["target"] else float("nan")
                    ),
                }
                rows.append(row)
                records.append({
                    "method": method_name,
                    "method_family": "class_conditioned_reconstruction",
                    "input_privilege": "image+target_class",
                    "split": "test",
                    "prompt_level": "class_id",
                    "dataset": "WORD",
                    "case_id": case_id,
                    "target_class_name": class_names[class_id],
                    "target_label_id": class_id,
                    "target_mask_mode": "semantic_class",
                    "prediction_status": "ok",
                    "mode": mode,
                    "dice": dice,
                    "iou": iou,
                    "nsd": None,
                    "hd95": None,
                    "prediction_voxels": counter["prediction"],
                    "target_voxels": counter["target"],
                })
            metrics[mode][str(class_id)] = {
                "class_name": class_names[class_id],
                "case_count": len(mode_counters[class_id]),
                "gt_present_case_count": len(present_case_dice),
                "case_dice_presence_mean": (
                    float(np.mean(present_case_dice))
                    if present_case_dice else float("nan")
                ),
                "case_iou_presence_mean": (
                    float(np.mean(present_case_iou))
                    if present_case_iou else float("nan")
                ),
                "case_dice_all_mean_empty_empty_one": (
                    float(np.mean(all_case_dice)) if all_case_dice else float("nan")
                ),
                "global_dice": dice_from_counts(
                    totals["intersection"], totals["prediction"], totals["target"]
                ),
                "global_iou": iou_from_counts(
                    totals["intersection"], totals["prediction"], totals["target"]
                ),
                "slice_dice_present_mean": (
                    present_slice_dice_sum / present_slices
                    if present_slices else float("nan")
                ),
                "prediction_voxels": totals["prediction"],
                "target_voxels": totals["target"],
                "prediction_to_target_volume_ratio": (
                    float(totals["prediction"]) / totals["target"]
                    if totals["target"] else float("nan")
                ),
            }
        for metric_name in (
            "case_dice_presence_mean",
            "case_iou_presence_mean",
            "global_dice",
            "global_iou",
            "slice_dice_present_mean",
        ):
            values = [
                metrics[mode][str(class_id)][metric_name]
                for class_id in class_ids
            ]
            metrics[mode]["foreground_mean_" + metric_name] = float(
                np.nanmean(values)
            )
    return metrics, rows, records


def slot_canvases(model, encoded):
    return {
        name: model.slot_bank.get_slot(name).forward_canvas(encoded)[0]
        for name in model.slot_names
    }


def reconstruct_from_keep(model, canvases, keep):
    fused = model.fuse_canvases(canvases, keep, model.slot_names)
    return model.forward_decoder(fused)


def encode_owt_organ_tokens(model, image):
    """Run the shared ViT and OrganCollector once before class-specific TGEnc."""
    encoded = model.patch_embed(image)
    encoded = encoded + model.pos_embed[:, 1:, :]
    cls_token = model.cls_token + model.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(encoded.shape[0], -1, -1)
    encoded = torch.cat((cls_tokens, encoded), dim=1)
    for block in model.blocks1:
        encoded = block(encoded)
    encoded = model.norm(encoded)[:, 1:, :]
    organ_tokens, _ = model.organ_embed(encoded)
    return organ_tokens


def reconstruct_owt_from_keep(model, organ_tokens, kept_class_ids, class_count):
    """Exactly reproduce legacy masking/TGEnc/AHER for one retained class set."""
    kept = {int(value) for value in kept_class_ids}
    dropped = [value for value in range(class_count) if value not in kept]
    masked_tokens, _ = model.random_masking(organ_tokens, dropped)
    transformed = model.blocks2[0](masked_tokens)
    for block in model.blocks2[1:]:
        transformed = block(transformed)
    transformed = model.norm(transformed)
    transformed = masked_tokens + transformed
    patches = model.forward_decoder(transformed, None, {})
    return model.unpatchify(patches)


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    test_csv = Path(args.test_csv).resolve()
    class_map = Path(args.class_map).resolve()
    output_dir = Path(args.output_dir).resolve()
    required_paths = [checkpoint_path, test_csv, class_map]
    preprocess_summary = None
    if args.preprocess_summary is not None:
        preprocess_summary = Path(args.preprocess_summary).resolve()
        required_paths.append(preprocess_summary)
    for required in required_paths:
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.input_size <= 0 or args.input_size % 16 != 0:
        raise ValueError("input-size must be positive and divisible by patch size 16")
    if args.global_crop_size <= 0:
        raise ValueError("global-crop-size must be positive")
    if args.expected_spacing is not None and preprocess_summary is None:
        raise ValueError("expected-spacing requires preprocess-summary")
    if preprocess_summary is not None:
        with open(preprocess_summary, "r", encoding="utf-8") as handle:
            preprocess = json.load(handle)
        recorded_spacing = preprocess.get("config", {}).get("spacing_mm")
        if args.expected_spacing is not None and (
            recorded_spacing is None
            or not np.allclose(
                recorded_spacing, args.expected_spacing, rtol=0.0, atol=1e-6
            )
        ):
            raise ValueError(
                "preprocessing spacing {} != expected {}".format(
                    recorded_spacing, list(args.expected_spacing)
                )
            )
    if args.threshold <= 0:
        raise ValueError("threshold must be positive")

    slot_specs, class_names = parse_class_configuration(class_map)
    class_ids = tuple(int(value) for value in args.class_ids.split(",") if value)
    if not class_ids or any(value == 0 or value not in class_names for value in class_ids):
        raise ValueError("class_ids must be a non-empty subset of foreground IDs 1..8")
    if len(set(class_ids)) != len(class_ids):
        raise ValueError("class_ids must not contain duplicates")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal evaluation requires CUDA")

    output_dir.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config.update(
        {
            "checkpoint": str(checkpoint_path),
            "test_csv": str(test_csv),
            "class_map": str(class_map),
            "output_dir": str(output_dir),
            "test_csv_sha256": sha256(test_csv),
            "checkpoint_sha256": sha256(checkpoint_path),
            "class_map_sha256": sha256(class_map),
            "class_ids_resolved": class_ids,
            "class_names": class_names,
            "primary_mode": PRIMARY_MODE,
            "threshold_policy": "fixed legacy OWT value; not selected on WORD test",
            "ground_truth_policy": (
                "unmodified semantic label after deterministic center crop"
            ),
        }
    )
    if preprocess_summary is not None:
        config["preprocess_summary"] = str(preprocess_summary)
        config["preprocess_summary_sha256"] = sha256(preprocess_summary)
        config["preprocess_spacing_mm"] = preprocess.get("config", {}).get(
            "spacing_mm"
        )
    atomic_json(output_dir / "resolved_config.json", config)

    device = torch.device(args.device)
    model, load_report = build_model(
        args.method,
        checkpoint_path,
        slot_specs,
        args.input_size,
        args.fusion_mode,
        allow_fusion_override=args.allow_fusion_override,
    )
    model.to(device).eval()
    atomic_json(output_dir / "checkpoint_load.json", load_report)
    dataset = build_dataset(
        test_csv,
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
    modes = ("direct_raw", "direct_post", "indirect_raw", "indirect_post")
    counters = {
        mode: {class_id: {} for class_id in class_ids}
        for mode in modes
    }
    id_to_slot_index = {
        int(spec["raw_class_id"]): index for index, spec in enumerate(slot_specs)
    }
    started = time.time()
    processed = 0
    amp_enabled = not args.no_amp
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True)
            labels = batch["full_label"][:, 0].numpy()
            batch_size = image.shape[0]
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if args.method == "orgslot":
                    encoded, _ = model.forward_encoder(image)
                    canvases = slot_canvases(model, encoded)
                else:
                    organ_tokens = encode_owt_organ_tokens(model, image)
                predictions = {mode: {} for mode in modes}
                for class_id in class_ids:
                    if args.method == "orgslot":
                        slot_index = id_to_slot_index[class_id]
                        direct_keep = torch.zeros(
                            batch_size,
                            len(slot_specs),
                            dtype=torch.bool,
                            device=device,
                        )
                        direct_keep[:, slot_index] = True
                        direct = reconstruct_from_keep(model, canvases, direct_keep)
                    else:
                        direct = reconstruct_owt_from_keep(
                            model, organ_tokens, (class_id,), len(slot_specs)
                        )
                    direct_binary = direct.mean(dim=1) > args.threshold
                    predictions["direct_raw"][class_id] = direct_binary.cpu().numpy()

                    if args.method == "orgslot":
                        indirect_keep = torch.ones_like(direct_keep)
                        indirect_keep[:, slot_index] = False
                        complement = reconstruct_from_keep(
                            model, canvases, indirect_keep
                        )
                    else:
                        complement = reconstruct_owt_from_keep(
                            model,
                            organ_tokens,
                            tuple(
                                value for value in range(len(slot_specs))
                                if value != class_id
                            ),
                            len(slot_specs),
                        )
                    residual = (image - complement).clamp_min_(0)
                    indirect_binary = residual.mean(dim=1) > args.threshold
                    predictions["indirect_raw"][class_id] = (
                        indirect_binary.cpu().numpy()
                    )

            for sample_index, case_id in enumerate(batch["case_id"]):
                for class_id in class_ids:
                    target = labels[sample_index] == class_id
                    direct_raw = predictions["direct_raw"][class_id][sample_index]
                    indirect_raw = predictions["indirect_raw"][class_id][sample_index]
                    sample_predictions = {
                        "direct_raw": direct_raw,
                        "direct_post": postprocess_slice(
                            direct_raw, args.min_size, args.opening_radius
                        ),
                        "indirect_raw": indirect_raw,
                        "indirect_post": postprocess_slice(
                            indirect_raw, args.min_size, args.opening_radius
                        ),
                    }
                    for mode, prediction in sample_predictions.items():
                        counter = counters[mode][class_id].setdefault(
                            str(case_id), new_counter()
                        )
                        update_counter(counter, prediction, target)
            processed += batch_size
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

    method_name = (
        "Legacy OWT reconstruction threshold"
        if args.method == "owt"
        else "OrganSlotBank reconstruction threshold"
    )
    metrics, rows, records = summarize(
        counters, class_ids, class_names, method_name
    )
    with open(output_dir / "per_case.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(output_dir / "per_case.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    result = {
        "method": method_name,
        "dataset": "WORD",
        "samples": len(dataset),
        "complete_test_set": args.max_samples is None,
        "elapsed_seconds": time.time() - started,
        "primary_mode": PRIMARY_MODE,
        "checkpoint_load": load_report,
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
