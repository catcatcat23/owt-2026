"""Build a fresh split from existing identical preprocessed cases, without copying pixels."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

from tools.prepare_word_official96 import CONFIG, COUNTS, check_plan, digest, dump, validate


def build(plan_path, sources, output):
    plan = json.loads(plan_path.read_text())
    check_plan(plan)
    if output.exists():
        raise FileExistsError(output)
    cases = {}
    for source in sources:
        summary = json.loads((source / "metadata/preprocess_summary.json").read_text())
        assert all(summary["config"].get(k) == v for k, v in CONFIG.items())
        for detail in summary["case_details"]:
            cases[detail["case_id"]] = (source, detail)
    needed = {e["case_id"] for entries in plan["splits"].values() for e in entries}
    assert needed <= cases.keys(), needed - cases.keys()
    root = output / "WORD"
    summary = copy.deepcopy(summary)
    summary.update(case_details=[], splits={}, protocol=plan["protocol"])
    for split, entries in plan["splits"].items():
        rows = []
        for entry in entries:
            case = entry["case_id"]
            source, detail = cases[case]
            count = int(detail["resampled_shape"][2])
            for kind, ext in (("image", "jpg"), ("mask", "png")):
                target = (source / detail["split"] / kind / case).resolve()
                assert len(list(target.glob("*." + ext))) == count, target
                link = root / split / kind / case
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target, target_is_directory=True)
            rows.extend([(str(root / split / "image" / case / (case + "_%d.jpg" % z)),
                          str(root / split / "mask" / case / (case + "_%d.png" % z))) for z in range(count)])
            summary["case_details"].append(dict(detail, split=split, official_split=entry["official_split"]))
        manifest = root / "csv" / ("WORD_%s_2D_native07072.csv" % split)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_pth", "mask_pth"])
            writer.writerows(rows)
        summary["splits"][split] = {"cases": len(entries), "slices_2d": len(rows)}
    dump(root / "metadata/split.json", plan)
    dump(root / "metadata/preprocess_summary.json", summary)
    subprocess.run([sys.executable, "tools/build_small_organ_roi_index.py", "--manifest",
                    str(root / "csv/WORD_Training_2D_native07072.csv"), "--output",
                    str(root / "metadata/small_organ_roi_index.json")], check=True)
    validate(root, plan_path)
    dump(root / "metadata/READY.json", {"protocol": plan["protocol"], "splits": summary["splits"],
         "plan_sha256": digest(plan_path), "manifests": {
             s: digest(root / "csv" / ("WORD_%s_2D_native07072.csv" % s)) for s in COUNTS}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.plan.resolve(), [p.resolve() for p in args.sources], args.output.resolve())
