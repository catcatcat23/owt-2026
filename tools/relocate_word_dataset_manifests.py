#!/usr/bin/env python3
"""Relocate absolute paths in processed WORD CSV manifests safely."""

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument(
        "--skip-existence-check",
        action="store_true",
        help="Do not require every relocated image and mask path to exist.",
    )
    return parser.parse_args()


def relocate_path(value: str, old_root: Path, new_root: Path) -> str:
    old = str(old_root).rstrip("/")
    new = str(new_root).rstrip("/")
    if value == old:
        return new
    prefix = old + "/"
    if not value.startswith(prefix):
        raise ValueError("path is not below old root: {}".format(value))
    return new + value[len(old) :]


def read_relocated_rows(
    manifest: Path,
    old_root: Path,
    new_root: Path,
    check_exists: bool,
) -> List[List[str]]:
    rows: List[List[str]] = []
    with manifest.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["image_pth", "mask_pth"]:
            raise ValueError("unexpected header in {}: {}".format(manifest, header))
        rows.append(header)
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    "expected two columns in {} line {}".format(manifest, line_number)
                )
            relocated = [relocate_path(value, old_root, new_root) for value in row]
            if check_exists:
                missing = [value for value in relocated if not Path(value).is_file()]
                if missing:
                    raise FileNotFoundError(
                        "missing relocated path in {} line {}: {}".format(
                            manifest, line_number, missing[0]
                        )
                    )
            rows.append(relocated)
    return rows


def atomic_write_csv(manifest: Path, rows: Sequence[Sequence[str]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=manifest.name + ".", suffix=".tmp", dir=str(manifest.parent)
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, manifest)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    args = parse_args()
    csv_dir = args.dataset_root / "csv"
    manifests = sorted(csv_dir.glob("*.csv"))
    if not manifests:
        raise FileNotFoundError("no CSV manifests found in {}".format(csv_dir))

    relocated = []
    for manifest in manifests:
        rows = read_relocated_rows(
            manifest,
            args.old_root,
            args.new_root,
            check_exists=not args.skip_existence_check,
        )
        relocated.append((manifest, rows))

    # No files are changed until every manifest has passed validation.
    for manifest, rows in relocated:
        atomic_write_csv(manifest, rows)
        print("relocated {} rows in {}".format(len(rows) - 1, manifest))


if __name__ == "__main__":
    main()
