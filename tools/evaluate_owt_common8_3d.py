#!/usr/bin/env python3
"""Evaluate a Fixfr4 Common8 OWT model with overlapping 3D windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import OWT_models
from evaluate_owt_common8 import (
    CLASS_NAMES,
    load_case,
    natural_key,
    read_cases,
    reconstruction_metrics,
    segmentation_metrics,
    summarize_reconstruction,
    summarize_segmentation,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True,
                        help="Use the 2D slice CSV so every slice is available.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="mae_vit_basefix16_patch16-LA")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--token-factor", type=int, default=20)
    parser.add_argument("--arch-version", default="v11")
    parser.add_argument("--training-version", default="v01")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--fix-frame", type=int, default=4)
    parser.add_argument("--temp-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--recon-threshold", type=float, default=0.02)
    parser.add_argument("--seg-threshold", type=float, default=0.15)
    parser.add_argument("--min-component-size", type=int, default=100)
    parser.add_argument("--spacing", type=float, nargs=3,
                        default=(3.0, 512.0 / 224.0, 512.0 / 224.0),
                        metavar=("Z", "Y", "X"))
    parser.add_argument("--nsd-tolerance-mm", type=float, default=3.0)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model_name = args.model
    use_la = "-LA" in model_name
    if use_la:
        model_name = model_name.split("-LA")[0].split("-")[0]

    model_args = SimpleNamespace(
        arch_version=args.arch_version,
        training_version=args.training_version,
        dataset_type="3D",
        fix_frame=args.fix_frame,
        temp_stride=args.temp_stride,
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
    return model.to(device).eval()


@torch.inference_mode()
def volume_lpips(
    model: torch.nn.Module,
    prediction: np.ndarray,
    target: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(prediction), batch_size):
        pred = torch.from_numpy(prediction[start:start + batch_size]).to(device)
        truth = torch.from_numpy(target[start:start + batch_size]).to(device)
        pred = pred.unsqueeze(1).repeat(1, 3, 1, 1)
        truth = truth.unsqueeze(1).repeat(1, 3, 1, 1)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            values = model.perceptual_loss(truth.contiguous(), pred.contiguous())
        total += float(values.sum().item())
        count += int(values.numel())
    return total / max(count, 1)


@torch.inference_mode()
def reconstruct_volume(
    model: torch.nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    selected_classes: list[int],
    fix_frame: int,
    batch_size: int,
    device: torch.device,
    compute_lpips: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    depth, height, width = images.shape
    if depth < fix_frame:
        raise ValueError(f"Volume depth {depth} is smaller than Fixfr{fix_frame}")

    target = images.copy()
    for class_id in selected_classes:
        target[labels == class_id] = 0.0

    prediction_sum = np.zeros((depth, height, width), dtype=np.float32)
    prediction_count = np.zeros((depth, 1, 1), dtype=np.float32)
    starts = list(range(depth - fix_frame + 1))

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        image_np = np.stack([images[s:s + fix_frame] for s in batch_starts])
        label_np = np.stack([labels[s:s + fix_frame] for s in batch_starts])
        image = torch.from_numpy(image_np).to(device)
        image = image.unsqueeze(1).repeat(1, 3, 1, 1, 1)
        label = torch.from_numpy(label_np).to(device)
        target_window = image.clone()
        for class_id in selected_classes:
            target_window = target_window.masked_fill(
                (label == class_id).unsqueeze(1), 0.0
            )

        middle = {
            "image_target": target_window,
            "random_selected_class": selected_classes,
        }
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            restored, cls_tokens, middle_output = model.forward_encoder(
                image, mask_ratio=1.0, middle=middle
            )
            patch_prediction = model.forward_decoder(
                restored, cls_tokens, middle_output
            )
            prediction = model.unpatchify3D(patch_prediction)[:, 0]
        prediction_np = prediction.float().cpu().numpy()

        for local_index, start in enumerate(batch_starts):
            prediction_sum[start:start + fix_frame] += prediction_np[local_index]
            prediction_count[start:start + fix_frame] += 1.0

    prediction = prediction_sum / np.maximum(prediction_count, 1.0)
    lpips = (
        volume_lpips(model, prediction, target, batch_size, device)
        if compute_lpips else float("nan")
    )
    return prediction, target, lpips


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cases = read_cases(args.test_csv)
    if args.max_cases > 0:
        cases = dict(list(cases.items())[:args.max_cases])
    print(
        f"Cases: {len(cases)}, slices: {sum(len(rows) for rows in cases.values())}",
        flush=True,
    )
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
        images, labels = load_case(case_rows, args.input_size)
        print(
            f"[{case_index}/{len(cases)}] {case_id}: {len(images)} slices, "
            f"labels={np.unique(labels).tolist()}",
            flush=True,
        )
        direct_predictions: dict[int, np.ndarray] = {}

        for mode, selected_classes in reconstruction_modes.items():
            prediction, target, lpips = reconstruct_volume(
                model, images, labels, selected_classes, args.fix_frame,
                args.batch_size, device, compute_lpips=not args.skip_lpips,
            )
            row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "mode": mode,
                "selected_classes": ",".join(map(str, selected_classes)),
                "slices": len(images),
                "windows": len(images) - args.fix_frame + 1,
            }
            row.update(
                reconstruction_metrics(
                    prediction, target, lpips, args.recon_threshold
                )
            )
            recon_rows.append(row)
            if mode.startswith("class_"):
                direct_predictions[int(mode.split("_")[1])] = prediction

        for class_id in range(1, args.num_classes + 1):
            target_mask = labels == class_id
            direct_row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "method": "direct_only_class_token_3d",
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
            direct_row.update(segmentation_metrics(
                direct_predictions[class_id], target_mask, args.seg_threshold,
                args.min_component_size, tuple(args.spacing),
                args.nsd_tolerance_mm,
            ))
            seg_rows.append(direct_row)

            leave_one_out, _, _ = reconstruct_volume(
                model, images, labels, [class_id], args.fix_frame,
                args.batch_size, device, compute_lpips=False,
            )
            indirect_score = np.maximum(images - leave_one_out, 0.0)
            indirect_row: dict[str, object] = {
                "dataset": args.dataset,
                "case_id": case_id,
                "method": "indirect_input_minus_without_class_3d",
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
            }
            indirect_row.update(segmentation_metrics(
                indirect_score, target_mask, args.seg_threshold,
                args.min_component_size, tuple(args.spacing),
                args.nsd_tolerance_mm,
            ))
            seg_rows.append(indirect_row)

        write_csv(output_dir / "step2_reconstruction_per_case.csv", recon_rows)
        write_csv(output_dir / "step3_segmentation_per_case.csv", seg_rows)

    write_csv(
        output_dir / "step2_reconstruction_summary.csv",
        summarize_reconstruction(recon_rows),
    )
    write_csv(
        output_dir / "step3_segmentation_summary.csv",
        summarize_segmentation(seg_rows),
    )

    protocol = vars(args).copy()
    protocol.update({
        "device": str(device),
        "cases": len(cases),
        "slices": sum(len(rows) for rows in cases.values()),
        "class_names": CLASS_NAMES,
        "elapsed_seconds": time.time() - started,
        "notes": [
            "Inputs use JPEG/255; no per-volume min-max normalization.",
            "Fixfr4 predictions use stride-1 windows and overlap averaging.",
            "Step 3 is reconstruction-derived pseudo-segmentation.",
            "Dice/NSD use true masks only for evaluation.",
        ],
    })
    with (output_dir / "protocol.json").open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2, ensure_ascii=True)

    print(f"Finished in {(time.time() - started) / 60:.1f} minutes", flush=True)
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
