"""Create deterministic, case-disjoint OrganSlot train/validation CSV files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.orgslot_manifest import parse_case_and_slice


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_records(records, validation_fraction=0.1, seed=0):
    by_case = {}
    for record in records:
        case_id, _ = parse_case_and_slice(record["image_pth"])
        by_case.setdefault(case_id, []).append(record)
    case_ids = sorted(by_case)
    if len(case_ids) < 2:
        raise ValueError("at least two cases are required for a disjoint split")
    random.Random(seed).shuffle(case_ids)
    validation_count = max(1, round(len(case_ids) * validation_fraction))
    validation_count = min(validation_count, len(case_ids) - 1)
    validation_cases = set(case_ids[:validation_count])
    train_cases = set(case_ids[validation_count:])
    train = [record for record in records if parse_case_and_slice(
        record["image_pth"]
    )[0] in train_cases]
    validation = [record for record in records if parse_case_and_slice(
        record["image_pth"]
    )[0] in validation_cases]
    return train, validation, tuple(sorted(train_cases)), tuple(sorted(validation_cases))


def read_manifest(path):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def write_manifest(path, fieldnames, records):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("validation-fraction must be strictly between 0 and 1")

    fieldnames, records = read_manifest(args.source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, validation, train_cases, validation_cases = split_records(
        records, args.validation_fraction, args.seed
    )
    write_manifest(output_dir / "train.csv", fieldnames, train)
    write_manifest(output_dir / "validation.csv", fieldnames, validation)
    report = {
        "source": str(Path(args.source).resolve()),
        "source_sha256": file_sha256(args.source),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "train_cases": list(train_cases),
        "validation_cases": list(validation_cases),
        "train_records": len(train),
        "validation_records": len(validation),
    }
    with open(output_dir / "split_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
