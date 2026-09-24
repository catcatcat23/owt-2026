#!/usr/bin/env python3
"""Build paper-aligned WORD/BTCV Common8 slices for the OWT loaders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import pil_cv2_compat as cv2
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output


COMMON8_NAMES = {
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

LABEL_MAPS = {
    "word": {2: 1, 4: 2, 3: 3, 6: 4, 7: 5, 8: 6, 1: 7, 5: 8},
    "btcv": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 11: 6, 6: 7, 7: 8},
}

SPLIT_SIZES = {
    "word": {"Training": 96, "Test": 24},
    "btcv": {"Training": 24, "Test": 6},
}


@dataclass(frozen=True)
class CasePair:
    case_id: str
    image: str
    mask: str


@dataclass(frozen=True)
class JobConfig:
    dataset: str
    split: str
    output_root: str
    spacing: tuple[float, float, float]
    hu_min: float
    hu_max: float
    canvas_size: int
    image_size: int
    jpeg_quality: int
    overwrite: bool


@dataclass
class CaseResult:
    case_id: str
    split: str
    image_rows: list[dict[str, str]]
    raw_shape: tuple[int, int, int]
    raw_spacing: tuple[float, float, float]
    resampled_shape: tuple[int, int, int]
    labels: list[int]
    foreground_voxels_before_canvas: int
    foreground_voxels_after_canvas: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("word", "btcv"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 3.0))
    parser.add_argument("--hu-min", type=float, default=-175.0)
    parser.add_argument("--hu-max", type=float, default=250.0)
    parser.add_argument("--canvas-size", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--fix-frame", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--splits", nargs="+", choices=("Training", "Test"), default=("Training", "Test")
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Debug limit per split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def nifti_stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def discover_word(root: Path) -> list[CasePair]:
    pairs: list[CasePair] = []
    for image_dir, mask_dir in ((root / "imagesTr", root / "labelsTr"),
                                (root / "imagesVal", root / "labelsVal")):
        for image in sorted(image_dir.glob("*.nii.gz")):
            mask = mask_dir / image.name
            if not mask.exists():
                mask = mask_dir / image.name.replace("word_", "label_")
            if not mask.exists():
                raise FileNotFoundError(f"Missing WORD mask for {image}")
            pairs.append(CasePair(nifti_stem(image), str(image), str(mask)))
    return sorted(pairs, key=lambda item: item.case_id)


def discover_btcv(root: Path) -> list[CasePair]:
    image_dir = root / "Training" / "img"
    mask_dir = root / "Training" / "label"
    pairs: list[CasePair] = []
    for image in sorted(image_dir.glob("img*.nii.gz")):
        image_stem = nifti_stem(image)
        number = image_stem[len("img") :] if image_stem.startswith("img") else image_stem
        mask = mask_dir / f"label{number}.nii.gz"
        if not mask.exists():
            raise FileNotFoundError(f"Missing BTCV mask for {image}")
        pairs.append(CasePair(nifti_stem(image), str(image), str(mask)))
    return pairs


def discover_cases(dataset: str, root: Path) -> list[CasePair]:
    return discover_word(root) if dataset == "word" else discover_btcv(root)


def split_cases(dataset: str, cases: Sequence[CasePair], seed: int) -> dict[str, list[CasePair]]:
    expected = sum(SPLIT_SIZES[dataset].values())
    if len(cases) != expected:
        raise ValueError(f"Expected {expected} labeled {dataset.upper()} cases, found {len(cases)}")
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    test_count = SPLIT_SIZES[dataset]["Test"]
    return {
        "Training": sorted(shuffled[test_count:], key=lambda item: item.case_id),
        "Test": sorted(shuffled[:test_count], key=lambda item: item.case_id),
    }


def normalize_ct(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = np.asarray(volume, dtype=np.float32)
    volume = np.nan_to_num(volume, nan=hu_min, posinf=hu_max, neginf=hu_min)
    return (np.clip(volume, hu_min, hu_max) - hu_min) / (hu_max - hu_min)


def map_labels(mask: np.ndarray, dataset: str) -> np.ndarray:
    rounded = np.rint(mask).astype(np.int16)
    if np.any(rounded < 0):
        raise ValueError("Negative source labels are not supported")
    output = np.zeros(rounded.shape, dtype=np.uint8)
    for source, target in LABEL_MAPS[dataset].items():
        output[rounded == source] = target
    return output


def center_crop_or_pad_xy(volume: np.ndarray, size: int, fill: float = 0) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D array, got {volume.shape}")
    output = np.full((size, size, volume.shape[2]), fill, dtype=volume.dtype)
    src_slices = []
    dst_slices = []
    for length in volume.shape[:2]:
        copied = min(length, size)
        src_start = max((length - size) // 2, 0)
        dst_start = max((size - length) // 2, 0)
        src_slices.append(slice(src_start, src_start + copied))
        dst_slices.append(slice(dst_start, dst_start + copied))
    output[dst_slices[0], dst_slices[1], :] = volume[src_slices[0], src_slices[1], :]
    return output


def write_jpeg(path: Path, gray: np.ndarray, quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if existing is None or existing.shape[:2] != gray.shape:
            raise IOError(f"Invalid existing image: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise IOError(f"Failed to write {path}")


def write_png(path: Path, mask: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if existing is None or existing.shape != mask.shape:
            raise IOError(f"Invalid existing mask: {path}")
        if not set(np.unique(existing)).issubset(set(range(9))):
            raise ValueError(f"Existing mask has invalid labels: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise IOError(f"Failed to write {path}")


def process_case(pair: CasePair, config: JobConfig) -> CaseResult:
    image_nii = nib.as_closest_canonical(nib.load(pair.image))
    mask_nii = nib.as_closest_canonical(nib.load(pair.mask))
    if len(image_nii.shape) != 3 or len(mask_nii.shape) != 3:
        raise ValueError(f"{pair.case_id}: expected 3D image and mask")

    raw_shape = tuple(int(value) for value in image_nii.shape)
    raw_spacing = tuple(float(value) for value in image_nii.header.get_zooms()[:3])
    image_normalized = normalize_ct(
        image_nii.get_fdata(dtype=np.float32), config.hu_min, config.hu_max
    )
    normalized_nii = nib.Nifti1Image(image_normalized, image_nii.affine)
    image_resampled_nii = resample_to_output(
        normalized_nii,
        voxel_sizes=config.spacing,
        order=1,
        mode="constant",
        cval=0.0,
    )
    mask_resampled_nii = resample_from_to(
        mask_nii,
        (image_resampled_nii.shape, image_resampled_nii.affine),
        order=0,
        mode="constant",
        cval=0.0,
    )

    image_resampled = np.clip(
        image_resampled_nii.get_fdata(dtype=np.float32), 0.0, 1.0
    )
    mask_common8 = map_labels(mask_resampled_nii.get_fdata(dtype=np.float32), config.dataset)
    foreground_before = int(np.count_nonzero(mask_common8))
    image_canvas = center_crop_or_pad_xy(image_resampled, config.canvas_size, fill=0.0)
    mask_canvas = center_crop_or_pad_xy(mask_common8, config.canvas_size, fill=0)
    foreground_after = int(np.count_nonzero(mask_canvas))
    if foreground_after != foreground_before:
        raise ValueError(
            f"{pair.case_id}: {foreground_before - foreground_after} Common8 foreground "
            f"voxels would be cropped by the {config.canvas_size}x{config.canvas_size} canvas"
        )

    output_root = Path(config.output_root)
    image_dir = output_root / config.split / "image" / pair.case_id
    mask_dir = output_root / config.split / "mask" / pair.case_id
    rows: list[dict[str, str]] = []
    for index in range(image_canvas.shape[2]):
        image_slice = cv2.resize(
            image_canvas[:, :, index],
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_slice = cv2.resize(
            mask_canvas[:, :, index],
            (config.image_size, config.image_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.uint8)
        image_u8 = np.rint(np.clip(image_slice, 0.0, 1.0) * 255.0).astype(np.uint8)
        stem = f"{pair.case_id}_{index}"
        image_path = image_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"
        write_jpeg(image_path, image_u8, config.jpeg_quality, config.overwrite)
        write_png(mask_path, mask_slice, config.overwrite)
        present_classes = [int(value) for value in np.unique(mask_slice) if value > 0]
        rows.append({
            "image_pth": str(image_path),
            "mask_pth": str(mask_path),
            "present_classes": "|".join(str(value) for value in present_classes),
        })

    return CaseResult(
        case_id=pair.case_id,
        split=config.split,
        image_rows=rows,
        raw_shape=raw_shape,
        raw_spacing=raw_spacing,
        resampled_shape=tuple(int(value) for value in image_resampled.shape),
        labels=[int(value) for value in np.unique(mask_canvas)],
        foreground_voxels_before_canvas=foreground_before,
        foreground_voxels_after_canvas=foreground_after,
    )


def write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("image_pth", "mask_pth", "present_classes"),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def make_window_rows(results: Sequence[CaseResult], fix_frame: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result in results:
        if len(result.image_rows) >= fix_frame:
            for index in range(len(result.image_rows) - fix_frame + 1):
                row = dict(result.image_rows[index])
                present = set()
                for window_row in result.image_rows[index:index + fix_frame]:
                    present.update(
                        int(value)
                        for value in window_row["present_classes"].split('|')
                        if value
                    )
                row["present_classes"] = "|".join(str(value) for value in sorted(present))
                rows.append(row)
    return rows


def validate_result(result: CaseResult, image_size: int) -> None:
    if len(result.image_rows) != result.resampled_shape[2]:
        raise ValueError(f"{result.case_id}: slice count does not match resampled depth")
    sample_indices = sorted({0, len(result.image_rows) // 2, len(result.image_rows) - 1})
    for index in sample_indices:
        row = result.image_rows[index]
        image = cv2.imread(row["image_pth"], cv2.IMREAD_COLOR)
        mask = cv2.imread(row["mask_pth"], cv2.IMREAD_UNCHANGED)
        if image is None or image.shape != (image_size, image_size, 3):
            raise ValueError(f"Bad image output: {row['image_pth']}")
        if mask is None or mask.shape != (image_size, image_size):
            raise ValueError(f"Bad mask output: {row['mask_pth']}")
        if not set(np.unique(mask)).issubset(set(range(9))):
            raise ValueError(f"Bad mask labels: {row['mask_pth']}")


def run_split(
    args: argparse.Namespace, split: str, cases: Sequence[CasePair]
) -> tuple[list[CaseResult], list[dict[str, str]], list[dict[str, str]]]:
    selected = list(cases[: args.max_cases]) if args.max_cases > 0 else list(cases)
    config = JobConfig(
        dataset=args.dataset,
        split=split,
        output_root=str(args.out),
        spacing=tuple(args.spacing),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        canvas_size=args.canvas_size,
        image_size=args.image_size,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    print(f"[{split}] processing {len(selected)} cases with {args.workers} workers", flush=True)
    results: list[CaseResult] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(process_case, pair, config): pair for pair in selected}
        for completed, future in enumerate(as_completed(futures), 1):
            pair = futures[future]
            result = future.result()
            validate_result(result, args.image_size)
            results.append(result)
            print(
                f"  {completed:03d}/{len(selected):03d} {pair.case_id}: "
                f"shape={result.resampled_shape}, labels={result.labels}",
                flush=True,
            )
    results.sort(key=lambda item: item.case_id)
    rows_2d = [row for result in results for row in result.image_rows]
    rows_fix = make_window_rows(results, args.fix_frame)
    return results, rows_2d, rows_fix


def main() -> None:
    args = parse_args()
    if args.hu_max <= args.hu_min:
        raise ValueError("--hu-max must be greater than --hu-min")
    args.out = args.out.resolve()
    cases = discover_cases(args.dataset, args.root)
    split_map = split_cases(args.dataset, cases, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    csv_dir = args.out / "csv"
    metadata_dir = args.out / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    split_manifest = {
        "dataset": args.dataset.upper(),
        "seed": args.seed,
        "source_root": str(args.root.resolve()),
        "splits": {
            name: [asdict(pair) for pair in split_cases_list]
            for name, split_cases_list in split_map.items()
        },
    }
    (metadata_dir / f"{args.dataset.upper()}_split_seed{args.seed}.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n"
    )

    summary = {
        "dataset": args.dataset.upper(),
        "common8_names": COMMON8_NAMES,
        "source_to_common8": LABEL_MAPS[args.dataset],
        "config": {
            "orientation": "RAS",
            "spacing_mm": list(args.spacing),
            "hu_clip": [args.hu_min, args.hu_max],
            "normalization": [0.0, 1.0],
            "canvas_xy": [args.canvas_size, args.canvas_size],
            "output_xy": [args.image_size, args.image_size],
            "image_format": f"JPEG quality {args.jpeg_quality}, 3 channels",
            "mask_format": "lossless uint8 PNG",
            "fix_frame": args.fix_frame,
            "all_slices_kept": True,
        },
        "splits": {},
    }

    for split in args.splits:
        results, rows_2d, rows_fix = run_split(args, split, split_map[split])
        prefix = args.dataset.upper()
        write_csv(csv_dir / f"{prefix}_{split}_2D_{args.image_size}.csv", rows_2d)
        write_csv(
            csv_dir / f"{prefix}_{split}_Fixfr{args.fix_frame}_{args.image_size}.csv",
            rows_fix,
        )
        labels = sorted({label for result in results for label in result.labels})
        summary["splits"][split] = {
            "cases": len(results),
            "slices_2d": len(rows_2d),
            f"windows_fixfr{args.fix_frame}": len(rows_fix),
            "labels": labels,
            "case_details": [
                {key: value for key, value in asdict(result).items() if key != "image_rows"}
                for result in results
            ],
        }

    summary_path = metadata_dir / f"{args.dataset.upper()}_preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.out), "splits": summary["splits"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
