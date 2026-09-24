#!/usr/bin/env python3
"""Evaluate Common8 OWT reconstruction and reconstruction-derived segmentation.

This keeps the original Step 2/Step 3 idea while fixing two protocol issues:
images retain the fixed HU-window /255 scaling used for training, and Dice/NSD
are computed against the actual class labels rather than thresholded CT targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import OWT_models


CLASS_NAMES = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "pancreas",
    7: "liver",
    8: "stomach",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="mae_vit_base_patch16-LA")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--token-factor", type=int, default=20)
    parser.add_argument("--arch-version", default="v11")
    parser.add_argument("--training-version", default="v01")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--intensity-norm", choices=("fixed_255", "per_sample"),
                        default="fixed_255")
    parser.add_argument("--class-names", nargs="+", default=None,
                        help="Optional foreground names in label order.")
    parser.add_argument("--recon-threshold", type=float, default=0.02)
    parser.add_argument("--seg-threshold", type=float, default=0.15)
    parser.add_argument("--min-component-size", type=int, default=100)
    parser.add_argument("--spacing", type=float, nargs=3,
                        default=(3.0, 512.0 / 224.0, 512.0 / 224.0),
                        metavar=("Z", "Y", "X"))
    parser.add_argument("--nsd-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--max-cases", type=int, default=0,
                        help="0 evaluates every test case")
    parser.add_argument("--case-start", type=int, default=0)
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def natural_key(path: str) -> list[object]:
    import re

    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path)]


def read_cases(csv_path: str) -> dict[str, list[tuple[str, str]]]:
    cases: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_pth", "mask_pth"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}: {csv_path}")
        for row in reader:
            case_id = Path(row["image_pth"]).parent.name
            cases[case_id].append((row["image_pth"], row["mask_pth"]))

    ordered: dict[str, list[tuple[str, str]]] = {}
    for case_id in sorted(cases, key=natural_key):
        ordered[case_id] = sorted(cases[case_id], key=lambda pair: natural_key(pair[0]))
    return ordered


def load_case(
    rows: list[tuple[str, str]],
    input_size: int,
    intensity_norm: str = "fixed_255",
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    labels = []
    for image_path, mask_path in rows:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or label is None:
            raise FileNotFoundError(f"Could not read {image_path} or {mask_path}")
        if label.ndim == 3:
            label = label[..., 0]
        expected = (input_size, input_size)
        if image.shape != expected or label.shape != expected:
            raise ValueError(
                f"Expected {expected}, got image={image.shape}, label={label.shape}: {image_path}"
            )
        image = image.astype(np.float32)
        if intensity_norm == "per_sample":
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        else:
            image = image / 255.0
        images.append(image)
        labels.append(label.astype(np.int64))
    return np.stack(images), np.stack(labels)


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model_name = args.model
    use_la = "-LA" in model_name
    if use_la:
        model_name = model_name.split("-LA")[0].split("-")[0]

    model_args = SimpleNamespace(
        arch_version=args.arch_version,
        training_version=args.training_version,
        dataset_type="2D",
        fix_frame=0,
        temp_stride=0,
        LA=use_la,
        num_classes=args.num_classes,
        num_classes_with_bg=args.num_classes + 1,
        token_factor=args.token_factor,
        organ_token_total=(args.num_classes + 1) * args.token_factor,
        cls_num=1,
        loss_version=["L2", "LPIPS"],
        text_encoding="None",
    )
    model = OWT_models.__dict__[model_name](
        img_size=args.input_size,
        norm_pix_loss=False,
        model_args=model_args,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    message = model.load_state_dict(checkpoint["model"], strict=True)
    print(f"Checkpoint load: {message}", flush=True)
    del checkpoint
    model.to(device).eval()
    return model


@torch.inference_mode()
def reconstruct(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    selected_classes: list[int],
    batch_size: int,
    device: torch.device,
    compute_lpips: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    predictions = []
    targets = []
    lpips_sum = 0.0
    lpips_count = 0

    for start in range(0, len(images), batch_size):
        end = min(start + batch_size, len(images))
        image = torch.from_numpy(images[start:end]).to(device)
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        label = torch.from_numpy(labels[start:end]).to(device)
        target = image.clone()
        for class_id in selected_classes:
            target = target.masked_fill((label == class_id).unsqueeze(1), 0.0)

        middle = {
            "image_target": target,
            "random_selected_class": selected_classes,
        }
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            restored, cls_tokens, middle_output = model.forward_encoder(
                image, mask_ratio=1.0, middle=middle
            )
            patch_prediction = model.forward_decoder(restored, cls_tokens, middle_output)
            prediction = model.unpatchify(patch_prediction)
            if compute_lpips:
                lpips = model.perceptual_loss(target.contiguous(), prediction.contiguous())
                lpips_sum += float(lpips.sum().item())
                lpips_count += int(lpips.numel())

        predictions.append(prediction[:, 0].float().cpu().numpy())
        targets.append(target[:, 0].float().cpu().numpy())

    mean_lpips = lpips_sum / max(lpips_count, 1) if compute_lpips else float("nan")
    return np.concatenate(predictions), np.concatenate(targets), mean_lpips


def ssim_2d(image_a: np.ndarray, image_b: np.ndarray) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_a = cv2.GaussianBlur(image_a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(image_b, (11, 11), 1.5)
    sigma_a = cv2.GaussianBlur(image_a * image_a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b = cv2.GaussianBlur(image_b * image_b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(image_a * image_b, (11, 11), 1.5) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def reconstruction_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    lpips: float,
    threshold: float,
) -> dict[str, float]:
    error = prediction - target
    per_slice_mse = np.mean(error * error, axis=(1, 2))
    psnr_values = 10.0 * np.log10(1.0 / np.maximum(per_slice_mse, 1e-12))
    ssim_values = [ssim_2d(a, b) for a, b in zip(prediction, target)]

    prediction_t = prediction.copy()
    target_t = target.copy()
    prediction_t[prediction_t < threshold] = 0.0
    target_t[target_t < threshold] = 0.0
    error_t = prediction_t - target_t
    per_slice_mse_t = np.mean(error_t * error_t, axis=(1, 2))
    psnr_values_t = 10.0 * np.log10(1.0 / np.maximum(per_slice_mse_t, 1e-12))
    ssim_values_t = [ssim_2d(a, b) for a, b in zip(prediction_t, target_t)]

    return {
        "l1": float(np.mean(np.abs(error))),
        "l2": float(np.mean(error * error)),
        "lpips": float(lpips),
        "psnr": float(np.mean(psnr_values)),
        "ssim": float(np.mean(ssim_values)),
        "thresholded_l1": float(np.mean(np.abs(error_t))),
        "thresholded_l2": float(np.mean(error_t * error_t)),
        "thresholded_psnr": float(np.mean(psnr_values_t)),
        "thresholded_ssim": float(np.mean(ssim_values_t)),
    }


def clean_mask(mask: np.ndarray, min_size: int) -> np.ndarray:
    structure = ndimage.generate_binary_structure(3, 1)
    components, count = ndimage.label(mask, structure=structure)
    if count:
        sizes = np.bincount(components.ravel())
        keep = sizes >= min_size
        keep[0] = False
        mask = keep[components]
    return ndimage.binary_opening(mask, structure=structure)


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = int(prediction.sum() + target.sum())
    if denominator == 0:
        return 1.0
    return float(2.0 * np.logical_and(prediction, target).sum() / denominator)


def surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> tuple[float, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 1.0, 0.0
    if not prediction.any() or not target.any():
        return 0.0, float("inf")

    structure = ndimage.generate_binary_structure(3, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    target_surface = target ^ ndimage.binary_erosion(target, structure=structure)
    distance_to_target = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    pred_distances = distance_to_target[pred_surface]
    target_distances = distance_to_pred[target_surface]
    nsd = (
        np.count_nonzero(pred_distances <= tolerance_mm)
        + np.count_nonzero(target_distances <= tolerance_mm)
    ) / max(len(pred_distances) + len(target_distances), 1)
    hd95 = np.percentile(np.concatenate([pred_distances, target_distances]), 95)
    return float(nsd), float(hd95)


def segmentation_metrics(
    score_volume: np.ndarray,
    target_mask: np.ndarray,
    threshold: float,
    min_size: int,
    spacing: tuple[float, float, float],
    tolerance_mm: float,
) -> dict[str, float | int]:
    raw = score_volume > threshold
    post = clean_mask(raw, min_size)
    raw_nsd, raw_hd95 = surface_metrics(raw, target_mask, spacing, tolerance_mm)
    post_nsd, post_hd95 = surface_metrics(post, target_mask, spacing, tolerance_mm)
    return {
        "gt_voxels": int(target_mask.sum()),
        "pred_voxels_raw": int(raw.sum()),
        "pred_voxels_post": int(post.sum()),
        "dice_raw": dice_score(raw, target_mask),
        "dice_post": dice_score(post, target_mask),
        "nsd_raw": raw_nsd,
        "nsd_post": post_nsd,
        "hd95_raw_mm": raw_hd95,
        "hd95_post_mm": post_hd95,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("inf")


def summarize_reconstruction(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    modes = sorted({str(row["mode"]) for row in rows}, key=natural_key)
    metrics = [
        "l1", "l2", "lpips", "psnr", "ssim",
        "thresholded_l1", "thresholded_l2", "thresholded_psnr", "thresholded_ssim",
    ]
    for mode in modes:
        subset = [row for row in rows if row["mode"] == mode]
        summary: dict[str, object] = {"mode": mode, "cases": len(subset)}
        for metric in metrics:
            summary[metric] = finite_mean([float(row[metric]) for row in subset])
        output.append(summary)
    return output


def summarize_segmentation(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    keys = sorted({(str(row["method"]), int(row["class_id"])) for row in rows})
    metrics = ["dice_raw", "dice_post", "nsd_raw", "nsd_post", "hd95_raw_mm", "hd95_post_mm"]
    for method, class_id in keys:
        subset = [row for row in rows if row["method"] == method and row["class_id"] == class_id]
        present = [row for row in subset if int(row["gt_voxels"]) > 0]
        for scope, scoped_rows in (("all_cases", subset), ("gt_present", present)):
            summary: dict[str, object] = {
                "method": method,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "scope": scope,
                "cases": len(scoped_rows),
            }
            for metric in metrics:
                summary[metric] = finite_mean([float(row[metric]) for row in scoped_rows])
            output.append(summary)
    return output


def main() -> None:
    args = parse_args()
    if args.class_names is not None:
        if len(args.class_names) != args.num_classes:
            raise ValueError(
                f"Expected {args.num_classes} class names, got {len(args.class_names)}"
            )
        CLASS_NAMES.clear()
        CLASS_NAMES[0] = "background"
        CLASS_NAMES.update({
            index: name for index, name in enumerate(args.class_names, start=1)
        })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cases = read_cases(args.test_csv)
    case_items = list(cases.items())[args.case_start:]
    if args.max_cases > 0:
        case_items = case_items[:args.max_cases]
    cases = dict(case_items)
    print(f"Cases: {len(cases)}, slices: {sum(len(rows) for rows in cases.values())}", flush=True)

    model = build_model(args, device)
    all_classes = list(range(args.num_classes + 1))
    reconstruction_modes = {
        "whole": [],
        "organs_only": [0],
        "background_only": list(range(1, args.num_classes + 1)),
    }
    for class_id in range(1, args.num_classes + 1):
        reconstruction_modes[f"class_{class_id}_only"] = [
            value for value in all_classes if value != class_id
        ]

    recon_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    started = time.time()

    for case_index, (case_id, case_rows) in enumerate(cases.items(), start=1):
        images, labels = load_case(case_rows, args.input_size, args.intensity_norm)
        print(
            f"[{case_index}/{len(cases)}] {case_id}: {len(images)} slices, labels={np.unique(labels).tolist()}",
            flush=True,
        )

        direct_predictions: dict[int, np.ndarray] = {}
        for mode, selected_classes in reconstruction_modes.items():
            prediction, target, lpips = reconstruct(
                model,
                images,
                labels,
                selected_classes,
                args.batch_size,
                device,
                compute_lpips=not args.skip_lpips,
            )
            row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "mode": mode,
                "selected_classes": ",".join(map(str, selected_classes)),
                "slices": len(images),
            }
            row.update(reconstruction_metrics(prediction, target, lpips, args.recon_threshold))
            recon_rows.append(row)
            if mode.startswith("class_"):
                class_id = int(mode.split("_")[1])
                direct_predictions[class_id] = prediction

        for class_id in range(1, args.num_classes + 1):
            target_mask = labels == class_id
            direct_score = direct_predictions[class_id]
            direct_row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "method": "direct_only_class_token",
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
            direct_row.update(segmentation_metrics(
                direct_score,
                target_mask,
                args.seg_threshold,
                args.min_component_size,
                tuple(args.spacing),
                args.nsd_tolerance_mm,
            ))
            seg_rows.append(direct_row)

            leave_one_out, _, _ = reconstruct(
                model,
                images,
                labels,
                [class_id],
                args.batch_size,
                device,
                compute_lpips=False,
            )
            indirect_score = np.maximum(images - leave_one_out, 0.0)
            indirect_row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "method": "indirect_input_minus_without_class",
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
            indirect_row.update(segmentation_metrics(
                indirect_score,
                target_mask,
                args.seg_threshold,
                args.min_component_size,
                tuple(args.spacing),
                args.nsd_tolerance_mm,
            ))
            seg_rows.append(indirect_row)

        write_csv(output_dir / "step2_reconstruction_per_case.csv", recon_rows)
        write_csv(output_dir / "step3_segmentation_per_case.csv", seg_rows)

    recon_summary = summarize_reconstruction(recon_rows)
    seg_summary = summarize_segmentation(seg_rows)
    write_csv(output_dir / "step2_reconstruction_summary.csv", recon_summary)
    write_csv(output_dir / "step3_segmentation_summary.csv", seg_summary)

    protocol = vars(args).copy()
    protocol.update({
        "device": str(device),
        "cases": len(cases),
        "slices": sum(len(rows) for rows in cases.values()),
        "class_names": CLASS_NAMES,
        "elapsed_seconds": time.time() - started,
        "notes": [
            f"Input normalization: {args.intensity_norm}.",
            "Step 3 is reconstruction-derived pseudo-segmentation, not a learned segmentation head.",
            "Dice/NSD use true class masks. Both raw and morphology-postprocessed metrics are reported.",
            "NSD is a voxel-surface approximation at the configured physical tolerance.",
        ],
    })
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=True)

    print(f"Finished in {(time.time() - started) / 60:.1f} minutes", flush=True)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
