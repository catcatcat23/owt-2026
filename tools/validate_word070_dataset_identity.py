"""Validate account-independent identity of the fixed WORD 0.7 protocol."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

EXPECTED = {
    "preprocess_summary_sha256": "272182e4bc09ee825db431ab6d41255a14df9b081c8ea2d1b07be6f8966e635b",
    "roi_core_sha256": "3a2596a6c7e53ae53c7640af6da4b900e61b7e63bb1ece1caa8df479828bfb8b",
    "train_count": 28586,
    "train_sha256": "ffbe5e71da4609d8ff17c88e875376b361db3e52a294e92c12f81709e571fc8c",
    "test_count": 6990,
    "test_sha256": "871a31790b6d3152b5358f8830a2cffebeec8e0c4cfcbd8da3d09538b9b0a9f1",
}


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    return _sha256_bytes(path.read_bytes())


def _manifest_signature(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        values = []
        for _, value in sorted(row.items()):
            if isinstance(value, str) and "/WORD/" in value:
                value = value.split("/WORD/", 1)[1]
            values.append(str(value))
        normalized.append("|".join(values))
    return len(rows), _sha256_bytes("\n".join(normalized).encode())


def validate(processed_root):
    root = Path(processed_root)
    summary = root / "metadata/preprocess_summary.json"
    roi_path = root / "metadata/small_organ_roi_index_verified.json"
    train_path = root / "csv/WORD_Training_2D_native07072.csv"
    test_path = root / "csv/WORD_Test_2D_native07072.csv"
    for path in (summary, roi_path, train_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    roi = json.loads(roi_path.read_text(encoding="utf-8"))
    roi_core = {
        key: value
        for key, value in roi.items()
        if key not in ("manifest", "manifest_sha256")
    }
    observed = {
        "preprocess_summary_sha256": _sha256_file(summary),
        "roi_core_sha256": _sha256_bytes(
            json.dumps(
                roi_core, sort_keys=True, separators=(",", ":")
            ).encode()
        ),
    }
    observed["train_count"], observed["train_sha256"] = _manifest_signature(
        train_path
    )
    observed["test_count"], observed["test_sha256"] = _manifest_signature(
        test_path
    )
    if observed != EXPECTED:
        raise RuntimeError(
            "WORD 0.7 dataset identity mismatch: observed={} expected={}".format(
                observed, EXPECTED
            )
        )
    return observed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.processed_root), sort_keys=True))
