"""Report matched-cohort Offline/Stage1/Stage2 Old/New/All case Dice (%)."""
import argparse
import csv
import json
import math
from pathlib import Path


OLD = (1, 2, 3, 4)
NEW = (5, 6, 7, 8)
MODES = ("binary_raw", "binary_post", "selected_raw", "selected_post")


def group_scores(metrics, stage):
    expected = OLD if stage == "stage1" else OLD + NEW
    actual = {int(k) for k in metrics if str(k).isdigit() and int(k) != 0}
    if actual != set(expected):
        raise ValueError(f"{stage}: expected foreground classes {expected}, got {actual}")
    values = {c: float(metrics[str(c)]["case_dice_presence_mean"]) for c in expected}
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        raise ValueError("Invalid case Dice")
    def mean(ids):
        return 100 * sum(values[c] for c in ids) / len(ids)
    return {
        "old": mean(OLD),
        "new": None if stage == "stage1" else mean(NEW),
        "all": None if stage == "stage1" else mean(OLD + NEW),
        "new_status": "unseen_no_output" if stage == "stage1" else "learned",
        "per_organ": {str(c): 100 * values[c] for c in expected},
    }


def load_report(path, stage):
    path = Path(path)
    result = json.loads(path.read_text())
    if result["split"] != "test" or not result["complete_split"] or not result["checkpoint_load"]["exact"]:
        raise ValueError("Require complete test evaluation with exact checkpoint loading")
    with path.with_name("per_case.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    expected = OLD if stage == "stage1" else OLD + NEW
    reference = None
    scores = {}
    for mode in MODES:
        if mode not in result["metrics"]:
            continue
        scores[mode] = group_scores(result["metrics"][mode], stage)
        for c in expected:
            selected = [r for r in rows if r["mode"] == mode and int(r["class_id"]) == c]
            cohort = {r["case_id"]: int(r["slices"]) for r in selected}
            if len(selected) != len(cohort) or len(cohort) != 24:
                raise ValueError("Expected exactly 24 unique cases per class/mode")
            if reference is not None and reference != cohort:
                raise ValueError("Case identities or slice counts differ across classes/modes")
            reference = cohort
    if "binary_post" not in scores:
        raise ValueError("Fixed-0.5 postprocessed reference is missing")
    config = json.loads(path.with_name("resolved_config.json").read_text())
    if float(config["binary_threshold"]) != 0.5:
        raise ValueError("Require fixed threshold 0.5 for forgetting comparisons")
    protocol = {k: config[k] for k in ("input_size", "global_crop_size", "min_size", "opening_radius", "fusion_mode")}
    return {"source": str(path), "scores": scores}, reference, protocol


def main():
    parser = argparse.ArgumentParser(__doc__)
    for stage in ("offline", "stage1", "stage2"):
        parser.add_argument("--" + stage, type=Path)
    args = parser.parse_args()
    reports, cohort, protocol = {}, None, None
    for stage in ("offline", "stage1", "stage2"):
        path = getattr(args, stage)
        if path is None:
            continue
        report, cases, settings = load_report(path, stage)
        if cohort is not None and (cohort != cases or protocol != settings):
            raise ValueError("Cannot compare different test cohorts or evaluation settings")
        reports[stage], cohort, protocol = report, cases, settings
    if not reports:
        parser.error("Provide at least one results.json")
    forgetting = None
    if "stage1" in reports and "stage2" in reports:
        forgetting = reports["stage1"]["scores"]["binary_post"]["old"] - reports["stage2"]["scores"]["binary_post"]["old"]
    print(json.dumps({"unit": "percent", "reports": reports,
                     "old_forgetting_fixed_post_pp": forgetting,
                     "stage1_unseen_policy": "New/All are null: new organ outputs do not exist; background excluded",
                     "pcdd_comparison": "reported-number reference; split/preprocessing not reproduced"}, indent=2))


if __name__ == "__main__":
    main()
