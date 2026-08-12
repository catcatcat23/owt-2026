#!/usr/bin/env python3
"""Build native-matrix 1x1x2 Common8 slices for crop-to-448 training.

No fixed canvas or offline resize is applied.  The paired runtime transform
crops the resampled native matrix and performs the only resize, to 448x448.
"""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Tuple

import cv2
import nibabel as nib
from nibabel.processing import resample_from_to
from nibabel.spaces import vox2out_vox
import numpy as np


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

SOURCE_TO_COMMON8 = {
    "WORD": {2: 1, 4: 2, 3: 3, 6: 4, 7: 5, 8: 6, 1: 7, 5: 8},
    "BTCV": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 11: 6, 6: 7, 7: 8},
}


@dataclass(frozen=True)
class CaseRecord:
    split: str
    case_id: str
    image: Path
    mask: Path


def _json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _resolve_source_path(source_root: Path, stale_path: str) -> Path:
    stale = Path(stale_path)
    direct_candidates = []
    for suffix_length in range(2, min(len(stale.parts), 5) + 1):
        direct_candidates.append(source_root.joinpath(*stale.parts[-suffix_length:]))
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    candidates = [
        candidate
        for candidate in source_root.rglob(stale.name)
        if candidate.is_file() and candidate.parent.name == stale.parent.name
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "could not uniquely resolve {} below {}: {}".format(
                stale_path, source_root, [str(path) for path in candidates]
            )
        )
    return candidates[0]


def load_case_records(
    dataset: str,
    source_root: Path,
    split_json: Path,
    max_cases: int = None,
) -> List[CaseRecord]:
    with open(split_json, "r", encoding="utf-8") as handle:
        split_data = json.load(handle)
    if split_data.get("dataset") != dataset:
        raise ValueError(
            "split dataset {} does not match requested {}".format(
                split_data.get("dataset"), dataset
            )
        )

    records = []
    for split, entries in split_data["splits"].items():
        for entry in entries:
            records.append(
                CaseRecord(
                    split=split,
                    case_id=entry["case_id"],
                    image=_resolve_source_path(source_root, entry["image"]),
                    mask=_resolve_source_path(source_root, entry["mask"]),
                )
            )
    if len({record.case_id for record in records}) != len(records):
        raise ValueError("case IDs are not unique across splits")
    if max_cases is not None:
        records = records[: int(max_cases)]
    return records


def remap_common8(label: np.ndarray, mapping: Mapping[int, int]) -> np.ndarray:
    rounded = np.rint(label).astype(np.int16, copy=False)
    output = np.zeros(rounded.shape, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        output[rounded == int(source_id)] = int(target_id)
    return output


def target_grid(image: nib.spatialimages.SpatialImage, spacing: Sequence[float]):
    canonical = nib.as_closest_canonical(image)
    shape, affine = vox2out_vox(canonical, voxel_sizes=tuple(spacing))
    return canonical, tuple(int(value) for value in shape), affine


def validate_case_geometry(
    record: CaseRecord,
    dataset: str,
    spacing: Sequence[float],
) -> dict:
    image = nib.load(str(record.image))
    mask = nib.as_closest_canonical(nib.load(str(record.mask)))
    canonical_image, output_shape, output_affine = target_grid(image, spacing)
    resampled_mask = resample_from_to(
        mask,
        (output_shape, output_affine),
        order=0,
        mode="constant",
        cval=0,
    )
    common8 = remap_common8(
        np.asarray(resampled_mask.dataobj), SOURCE_TO_COMMON8[dataset]
    )
    return {
        "case_id": record.case_id,
        "split": record.split,
        "image": str(record.image),
        "mask": str(record.mask),
        "raw_shape": list(canonical_image.shape[:3]),
        "raw_spacing": [float(value) for value in canonical_image.header.get_zooms()[:3]],
        "resampled_shape": list(output_shape),
        "target_spacing": [float(value) for value in spacing],
        "labels_after_mapping": [int(value) for value in np.unique(common8)],
        "common8_foreground_voxels": int(np.count_nonzero(common8)),
    }


def _normalize_ct(image: np.ndarray, hu_clip: Sequence[float]) -> np.ndarray:
    lower, upper = [float(value) for value in hu_clip]
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image.astype(np.float32, copy=False)


def _write_csv(path: Path, rows: Iterable[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_pth", "mask_pth"])
        writer.writerows(rows)


def write_case(
    record: CaseRecord,
    dataset: str,
    output_root: Path,
    spacing: Sequence[float],
    hu_clip: Sequence[float],
    image_order: int,
    jpeg_quality: int,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    image = nib.load(str(record.image))
    mask = nib.as_closest_canonical(nib.load(str(record.mask)))
    canonical_image, output_shape, output_affine = target_grid(image, spacing)
    resampled_image = resample_from_to(
        canonical_image,
        (output_shape, output_affine),
        order=int(image_order),
        mode="constant",
        cval=float(hu_clip[0]),
    )
    resampled_mask = resample_from_to(
        mask,
        (output_shape, output_affine),
        order=0,
        mode="constant",
        cval=0,
    )

    image_array = _normalize_ct(np.asarray(resampled_image.dataobj), hu_clip)
    label_array = remap_common8(
        np.asarray(resampled_mask.dataobj), SOURCE_TO_COMMON8[dataset]
    )

    image_dir = output_root / dataset / record.split / "image" / record.case_id
    mask_dir = output_root / dataset / record.split / "mask" / record.case_id
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows_2d = []
    image_paths = []
    mask_paths = []
    for index in range(image_array.shape[2]):
        image_path = image_dir / "{}_{}.jpg".format(record.case_id, index)
        mask_path = mask_dir / "{}_{}.png".format(record.case_id, index)
        gray = np.rint(image_array[:, :, index] * 255.0).astype(np.uint8)
        three_channel = np.repeat(gray[:, :, None], 3, axis=2)
        label = label_array[:, :, index]
        if not cv2.imwrite(
            str(image_path),
            three_channel,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        ):
            raise OSError("failed to write {}".format(image_path))
        if not cv2.imwrite(str(mask_path), label):
            raise OSError("failed to write {}".format(mask_path))
        image_paths.append(image_path)
        mask_paths.append(mask_path)
        rows_2d.append((str(image_path), str(mask_path)))

    rows_fixfr4 = [
        (str(image_paths[index]), str(mask_paths[index]))
        for index in range(max(0, len(image_paths) - 3))
    ]
    return rows_2d, rows_fixfr4


def parse_args():
    parser = argparse.ArgumentParser(
        "Preprocess WORD/BTCV to native-matrix Common8 at configurable spacing"
    )
    parser.add_argument("--dataset", choices=("WORD", "BTCV"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 2.0))
    parser.add_argument(
        "--runtime-input-size",
        type=int,
        default=None,
        help="Optional intended runtime square input size recorded in metadata.",
    )
    parser.add_argument("--manifest-tag", default="native112")
    parser.add_argument("--hu-clip", nargs=2, type=float, default=(-175.0, 250.0))
    parser.add_argument("--image-order", choices=(1, 3), type=int, default=3)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not all(value > 0 for value in args.spacing):
        raise ValueError("spacing values must be positive")
    if args.runtime_input_size is not None and args.runtime_input_size <= 0:
        raise ValueError("runtime-input-size must be positive")
    if not args.manifest_tag or any(char.isspace() for char in args.manifest_tag):
        raise ValueError("manifest-tag must be non-empty and contain no whitespace")
    dataset_output_root = args.output_root / args.dataset
    if (
        dataset_output_root.exists()
        and any(dataset_output_root.iterdir())
        and not args.validate_only
    ):
        raise FileExistsError(
            "refusing to write into non-empty dataset root: {}".format(
                dataset_output_root
            )
        )

    records = load_case_records(
        args.dataset,
        args.source_root,
        args.split_json,
        max_cases=args.max_cases,
    )
    print("Resolved {} {} cases".format(len(records), args.dataset), flush=True)

    details = []
    for index, record in enumerate(records, start=1):
        detail = validate_case_geometry(record, args.dataset, args.spacing)
        details.append(detail)
        print(
            "geometry {}/{} {} native_resampled_shape={}".format(
                index,
                len(records),
                record.case_id,
                detail["resampled_shape"],
            ),
            flush=True,
        )

    geometry_report = {
        "dataset": args.dataset,
        "config": {
            "orientation": "RAS",
            "spacing_mm": list(args.spacing),
            "offline_spatial_matrix": "native_after_resampling",
            "offline_resize": None,
            "runtime_final_input_xy": (
                [args.runtime_input_size, args.runtime_input_size]
                if args.runtime_input_size is not None
                else None
            ),
            "manifest_tag": args.manifest_tag,
            "hu_clip": list(args.hu_clip),
            "normalization": [0.0, 1.0],
            "image_interpolation_order": args.image_order,
            "label_interpolation_order": 0,
            "jpeg_quality": args.jpeg_quality,
        },
        "common8_names": COMMON8_NAMES,
        "source_to_common8": SOURCE_TO_COMMON8[args.dataset],
        "case_details": details,
    }
    report_path = args.output_root / args.dataset / "metadata" / "geometry_preflight.json"
    _json_dump(report_path, geometry_report)
    if args.validate_only:
        print("Geometry validation passed; report: {}".format(report_path))
        return

    rows_by_split = {
        split: {"2d": [], "fixfr4": []} for split in sorted({r.split for r in records})
    }
    for index, record in enumerate(records, start=1):
        rows_2d, rows_fixfr4 = write_case(
            record,
            args.dataset,
            args.output_root,
            args.spacing,
            args.hu_clip,
            args.image_order,
            args.jpeg_quality,
        )
        rows_by_split[record.split]["2d"].extend(rows_2d)
        rows_by_split[record.split]["fixfr4"].extend(rows_fixfr4)
        print(
            "write {}/{} {} slices={} windows={}".format(
                index, len(records), record.case_id, len(rows_2d), len(rows_fixfr4)
            ),
            flush=True,
        )

    csv_dir = args.output_root / args.dataset / "csv"
    for split, rows in rows_by_split.items():
        _write_csv(
            csv_dir / "{}_{}_2D_{}.csv".format(args.dataset, split, args.manifest_tag),
            rows["2d"],
        )
        _write_csv(
            csv_dir / "{}_{}_Fixfr4_{}.csv".format(args.dataset, split, args.manifest_tag),
            rows["fixfr4"],
        )
    geometry_report["splits"] = {
        split: {
            "cases": sum(record.split == split for record in records),
            "slices_2d": len(rows["2d"]),
            "windows_fixfr4": len(rows["fixfr4"]),
        }
        for split, rows in rows_by_split.items()
    }
    _json_dump(
        args.output_root / args.dataset / "metadata" / "preprocess_summary.json",
        geometry_report,
    )


if __name__ == "__main__":
    main()
