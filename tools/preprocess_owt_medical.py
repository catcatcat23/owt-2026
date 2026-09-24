#!/usr/bin/env python3
"""Convert WORD or BTCV NIfTI volumes to the 2D slice format used by OWT.

The OWT dataset loader reads image/mask pairs from a CSV with columns:
    image_pth,mask_pth

For 3D training, the loader still reads consecutive 2D files from the
starting slice listed in the CSV. This script therefore writes both per-slice
2D CSVs and optional fix-frame window CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - clear runtime message
    raise SystemExit(
        "nibabel is required for NIfTI preprocessing. Install it in abdpet with:\n"
        "  /gpfs/work/aac/bolinren19/.conda/envs/abdpet/bin/python -m pip install nibabel"
    ) from exc


@dataclass(frozen=True)
class CasePair:
    image: Path
    mask: Path | None
    case_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("word", "btcv"), required=True)
    parser.add_argument("--root", type=Path, required=True, help="Raw dataset root.")
    parser.add_argument("--out", type=Path, required=True, help="Output processed root.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hu-min", type=float, default=-1000.0)
    parser.add_argument("--hu-max", type=float, default=1000.0)
    parser.add_argument(
        "--keep-background-every",
        type=int,
        default=0,
        help="Keep every Nth all-background slice. 0 drops pure-background slices.",
    )
    parser.add_argument(
        "--fix-frame",
        type=int,
        default=4,
        help="Window length for generated 3D CSV. Set 0 to skip 3D CSV.",
    )
    parser.add_argument("--max-cases", type=int, default=0, help="Debug limit per split.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split names to process. WORD: train val. BTCV: train.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_nifti(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {arr.shape}: {path}")
    return arr


def normalize_ct(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    volume = volume.astype(np.float32)
    volume = np.nan_to_num(volume, nan=hu_min, posinf=hu_max, neginf=hu_min)
    volume = np.clip(volume, hu_min, hu_max)
    volume = (volume - hu_min) / max(hu_max - hu_min, 1e-6)
    return (volume * 255.0).round().astype(np.uint8)


def resize_image(slice_2d: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(slice_2d, (image_size, image_size), interpolation=cv2.INTER_LINEAR)


def resize_mask(slice_2d: np.ndarray, image_size: int) -> np.ndarray:
    out = cv2.resize(slice_2d, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.uint8)


def write_image(path: Path, gray: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ok = cv2.imwrite(str(path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), mask)
    if not ok:
        raise IOError(f"Failed to write mask: {path}")


def word_pairs(root: Path, split: str) -> list[CasePair]:
    if split == "train":
        image_dir, mask_dir = root / "imagesTr", root / "labelsTr"
    elif split in {"val", "valid", "validation"}:
        image_dir, mask_dir = root / "imagesVal", root / "labelsVal"
    else:
        raise ValueError(f"WORD split with labels must be train or val, got {split}")

    pairs: list[CasePair] = []
    for image in sorted(image_dir.glob("*.nii.gz")):
        mask = mask_dir / image.name.replace("word_", "label_")
        if not mask.exists():
            mask = mask_dir / image.name
        if not mask.exists():
            raise FileNotFoundError(f"No WORD label found for {image}")
        pairs.append(CasePair(image=image, mask=mask, case_id=image.name))
    return pairs


def btcv_pairs(root: Path, split: str) -> list[CasePair]:
    raw = root / "RawData"
    if split != "train":
        raise ValueError("BTCV has no public RawData test labels here; use split=train.")
    image_dir = raw / "Training" / "img"
    mask_dir = raw / "Training" / "label"
    pairs: list[CasePair] = []
    for image in sorted(image_dir.glob("img*.nii.gz")):
        number = image.name[len("img") : -len(".nii.gz")]
        mask = mask_dir / f"label{number}.nii.gz"
        if not mask.exists():
            raise FileNotFoundError(f"No BTCV label found for {image}")
        pairs.append(CasePair(image=image, mask=mask, case_id=image.name))
    return pairs


def get_pairs(dataset: str, root: Path, split: str) -> list[CasePair]:
    if dataset == "word":
        return word_pairs(root, split)
    if dataset == "btcv":
        return btcv_pairs(root, split)
    raise ValueError(dataset)


def iter_slices(volume: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    # NIfTI arrays are usually (H, W, D). Treat axis 2 as axial slice index.
    for idx in range(volume.shape[2]):
        yield idx, volume[:, :, idx]


def convert_pair(
    pair: CasePair,
    split_out: Path,
    image_size: int,
    hu_min: float,
    hu_max: float,
    keep_background_every: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    image_vol = load_nifti(pair.image)
    if pair.mask is None:
        raise ValueError("Current OWT training requires masks.")
    mask_vol = load_nifti(pair.mask)
    if image_vol.shape != mask_vol.shape:
        raise ValueError(f"Shape mismatch for {pair.case_id}: image {image_vol.shape}, mask {mask_vol.shape}")

    image_u8 = normalize_ct(image_vol, hu_min=hu_min, hu_max=hu_max)
    mask_u8 = np.rint(mask_vol).astype(np.uint8)
    image_dir = split_out / "image" / pair.case_id
    mask_dir = split_out / "mask" / pair.case_id
    rows: list[dict[str, str]] = []

    for slice_idx, image_slice in iter_slices(image_u8):
        mask_slice = mask_u8[:, :, slice_idx]
        has_fg = bool(np.any(mask_slice > 0))
        if not has_fg:
            if keep_background_every <= 0 or slice_idx % keep_background_every != 0:
                continue

        stem = f"{pair.case_id}_{slice_idx}"
        image_path = image_dir / f"{stem}.jpg"
        mask_path = mask_dir / f"{stem}.png"
        if overwrite or not image_path.exists():
            write_image(image_path, resize_image(image_slice, image_size))
        if overwrite or not mask_path.exists():
            write_mask(mask_path, resize_mask(mask_slice, image_size))
        rows.append({"image_pth": str(image_path), "mask_pth": str(mask_path)})

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_pth", "mask_pth"])
        writer.writeheader()
        writer.writerows(rows)


def make_3d_rows(rows: list[dict[str, str]], fix_frame: int) -> list[dict[str, str]]:
    if fix_frame <= 0:
        return []
    by_case: dict[tuple[str, str], list[tuple[int, dict[str, str]]]] = {}
    for row in rows:
        image = Path(row["image_pth"])
        mask = Path(row["mask_pth"])
        try:
            idx = int(image.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            continue
        by_case.setdefault((str(image.parent), str(mask.parent)), []).append((idx, row))

    window_rows: list[dict[str, str]] = []
    for (_image_parent, _mask_parent), items in by_case.items():
        items = sorted(items, key=lambda x: x[0])
        present = {idx: row for idx, row in items}
        for idx, row in items:
            if all((idx + off) in present for off in range(fix_frame)):
                window_rows.append(row)
    return window_rows


def main() -> None:
    args = parse_args()
    default_splits = ["train", "val"] if args.dataset == "word" else ["train"]
    splits = args.splits or default_splits
    args.out.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}
    for split in splits:
        pairs = get_pairs(args.dataset, args.root, split)
        if args.max_cases > 0:
            pairs = pairs[: args.max_cases]

        split_name = "Training" if split == "train" else "Test"
        split_out = args.out / split_name
        all_rows: list[dict[str, str]] = []
        print(f"[{split}] cases={len(pairs)} output={split_out}", flush=True)
        for i, pair in enumerate(pairs, 1):
            rows = convert_pair(
                pair=pair,
                split_out=split_out,
                image_size=args.image_size,
                hu_min=args.hu_min,
                hu_max=args.hu_max,
                keep_background_every=args.keep_background_every,
                overwrite=args.overwrite,
            )
            all_rows.extend(rows)
            print(f"  {i:04d}/{len(pairs):04d} {pair.case_id}: slices={len(rows)}", flush=True)

        csv_path = args.out / f"{args.dataset.upper()}_{split_name}_2D_{args.image_size}.csv"
        write_csv(csv_path, all_rows)
        window_rows = make_3d_rows(all_rows, args.fix_frame)
        if args.fix_frame > 0:
            csv3d_path = args.out / f"{args.dataset.upper()}_{split_name}_Fixfr{args.fix_frame}_{args.image_size}.csv"
            write_csv(csv3d_path, window_rows)
        summary[split_name] = {
            "cases": len(pairs),
            "slices_2d": len(all_rows),
            f"windows_fixfr{args.fix_frame}": len(window_rows),
        }

    summary_path = args.out / f"{args.dataset.upper()}_preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
