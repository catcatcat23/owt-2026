"""Exploratory, outcome-selected case replacement. NOT a new SOTA protocol."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def case_ids(data):
    return sorted({Path(r["image_pth"]).parent.name for r in data})


def prepare(args):
    plan = json.loads(args.plan.read_text())
    old_csv = args.old_data / "csv"
    train_path = old_csv / "WORD_Training_2D_native07072.csv"
    test_path = old_csv / "WORD_Test_2D_native07072.csv"
    train, test = rows(train_path), rows(test_path)
    saved = json.loads((args.old_run / "dataset_manifest_checksums.json").read_text())
    assert sha(train_path) == saved["train"] and sha(test_path) == saved["validation"]
    assert case_ids(train) == plan["original_training_cases"]
    assert case_ids(test) == plan["original_test_cases"]
    assert sorted(random.Random(42).sample(plan["eligible_pool"], 5)) == plan["replacement_cases"]
    assert len(set(plan["removed_cases"])) == 5
    assert set(plan["removed_cases"]) <= set(case_ids(test))
    assert not set(plan["replacement_cases"]) & (set(case_ids(train)) | set(case_ids(test)))
    result = [r for r in test if Path(r["image_pth"]).parent.name not in plan["removed_cases"]]
    for case in plan["replacement_cases"]:
        images = args.replacement_data / "image" / case
        masks = args.replacement_data / "mask" / case
        files = sorted(images.glob("*.jpg"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        assert files and len(files) == len(list(masks.glob("*.png")))
        assert [int(p.stem.rsplit("_", 1)[1]) for p in files] == list(range(len(files)))
        result += [{"image_pth": str(p), "mask_pth": str(masks / (p.stem + ".png"))} for p in files]
    assert len(case_ids(result)) == 24
    assert not set(case_ids(result)) & set(case_ids(train))
    assert all(Path(p).is_file() for r in result for p in r.values())
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "test.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_pth", "mask_pth"])
        writer.writeheader()
        writer.writerows(result)
    report = dict(plan, samples=len(result), test_manifest_sha256=sha(args.output / "test.csv"),
                  old_thresholds_sha256=sha(args.thresholds), old_results_sha256=sha(args.old_results))
    # The shared cohort plan was first used for E; identify the actual model
    # being evaluated rather than inheriting that historical model label.
    report["cohort_plan_original_result"] = plan.get("original_result")
    report["original_result"] = str(args.old_results.resolve())
    report["evaluated_checkpoint"] = str((args.old_run / "checkpoint-802.pth").resolve())
    (args.output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def compare(args):
    old = json.loads(args.old_results.read_text())
    new = json.loads((args.output / "heads/results.json").read_text())
    protocol = json.loads((args.output / "protocol.json").read_text())
    assert new["complete_split"] and new["samples"] == protocol["samples"]
    assert new["checkpoint_load"]["exact"] and new["checkpoint_load"]["epoch"] == 802
    assert sha(args.thresholds) == protocol["old_thresholds_sha256"]
    assert sha(args.old_results) == protocol["old_results_sha256"]
    lines = ["# Exploratory replacement evaluation — not SOTA", "",
             "Five failures were deliberately removed. Different cohorts; not evidence of model improvement.",
             "Original thresholds/postprocessing unchanged; no retraining or recalibration.", "",
             "| Readout | Organ | Original 24 Dice (%) | Replacement 24 Dice (%) |", "|---|---|---:|---:|"]
    for mode in old["metrics"]:
        if mode not in new["metrics"]:
            continue
        before, after = [], []
        for c in map(str, range(1, 9)):
            a, b = old["metrics"][mode][c], new["metrics"][mode][c]
            assert b["case_count"] == 24
            av, bv = a["case_dice_presence_mean"] * 100, b["case_dice_presence_mean"] * 100
            before.append(av)
            after.append(bv)
            lines.append("| %s | %s | %.2f | %.2f |" % (mode, a["class_name"], av, bv))
        lines.append("| %s | Mean | %.2f | %.2f |" % (mode, sum(before)/8, sum(after)/8))
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage", choices=("prepare", "compare"))
    for name in ("plan", "old-data", "old-run", "replacement-data", "output", "thresholds", "old-results"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    (prepare if args.stage == "prepare" else compare)(args)
