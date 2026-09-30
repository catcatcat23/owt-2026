"""Independent WORD official-pool 96/26/24 protocol; never mutate old data."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import zipfile

COUNTS = {"Training": 96, "Validation": 26, "Test": 24}
CONFIG = {"spacing_mm": [0.7, 0.7, 2.0], "orientation": "RAS",
          "hu_clip": [-175.0, 250.0], "image_interpolation_order": 3,
          "label_interpolation_order": 0, "jpeg_quality": 95,
          "offline_resize": None, "runtime_final_input_xy": [448, 448]}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(raw):
    pools = {s: sorted(p.name[:-7] for p in (raw / ("images" + s)).glob("*.nii.gz"))
             for s in ("Tr", "Val", "Ts")}
    assert {s: len(v) for s, v in pools.items()} == {"Tr": 100, "Val": 20, "Ts": 30}
    assert len(set(sum(pools.values(), []))) == 150
    train = sorted(random.Random(42).sample(pools["Tr"], 96))
    holdout = sorted(pools["Val"] + pools["Ts"])
    test = sorted(random.Random(42).sample(holdout, 24))
    groups = {"Training": train, "Test": test,
              "Validation": sorted(set(holdout) - set(test))}
    origin = {c: s for s, ids in pools.items() for c in ids}
    return {"dataset": "WORD", "protocol": "official_pool_96_26_24_seed42",
            "seed": 42, "sampling": "independent random.Random(42).sample on sorted pools",
            "official_pools": pools, "unused_train": sorted(set(pools["Tr"]) - set(train)),
            "splits": {s: [{"case_id": c, "official_split": origin[c],
                            "image": "images%s/%s.nii.gz" % (origin[c], c),
                            "mask": "labels%s/%s.nii.gz" % (origin[c], c)} for c in ids]
                       for s, ids in groups.items()}}


def check_plan(plan):
    ids = {s: {r["case_id"] for r in rows} for s, rows in plan["splits"].items()}
    assert {s: len(v) for s, v in ids.items()} == COUNTS
    assert sum(len(v) for v in ids.values()) == len(set.union(*ids.values())) == 146
    assert ids["Training"] <= set(plan["official_pools"]["Tr"])
    assert ids["Test"] | ids["Validation"] == set(plan["official_pools"]["Val"] + plan["official_pools"]["Ts"])


def build(args):
    plan = json.loads(args.plan.read_text())
    assert plan == make_plan(args.raw), "Official files or frozen sampling plan changed"
    check_plan(plan)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; old protocols are never overwritten")
    args.output.mkdir(parents=True)
    old = json.loads((args.old / "metadata/preprocess_summary.json").read_text())
    assert all(old["config"].get(k) == v for k, v in CONFIG.items())
    overlay = args.output / "raw"
    overlay.mkdir()
    for name in ("imagesTr", "imagesVal", "imagesTs", "labelsTr", "labelsVal"):
        (overlay / name).symlink_to(args.raw / name, target_is_directory=True)
    (overlay / "labelsTs").mkdir()
    with zipfile.ZipFile(args.labels_zip) as archive:
        # Extract ONLY the exact expected labels; no arbitrary ZIP paths.
        for case in plan["official_pools"]["Ts"]:
            name = "labelsTs/%s.nii.gz" % case
            with archive.open(name) as src, (overlay / name).open("wb") as dst:
                import shutil
                shutil.copyfileobj(src, dst)
    ts_plan = {"dataset": "WORD", "splits": {"Test": [
        {"case_id": c, "image": "imagesTs/%s.nii.gz" % c,
         "mask": "labelsTs/%s.nii.gz" % c} for c in plan["official_pools"]["Ts"]]}}
    dump(args.output / "official_ts.json", ts_plan)
    subprocess.run([sys.executable, "tools/preprocess_common8_112_448.py",
                    "--dataset", "WORD", "--source-root", str(overlay),
                    "--split-json", str(args.output / "official_ts.json"),
                    "--output-root", str(args.output / "resampled_ts"),
                    "--spacing", "0.7", "0.7", "2", "--runtime-input-size", "448",
                    "--manifest-tag", "native07072", "--workers", "2",
                    "--hu-clip", "-175", "250", "--image-order", "3", "--jpeg-quality", "95"], check=True)
    ts_root = args.output / "resampled_ts/WORD"
    ts = json.loads((ts_root / "metadata/preprocess_summary.json").read_text())
    assert all(ts["config"].get(k) == v for k, v in CONFIG.items())
    sources = {d["case_id"]: (args.old, d) for d in old["case_details"]}
    sources.update({d["case_id"]: (ts_root, d) for d in ts["case_details"]})
    root = args.output / "WORD"
    summary = copy.deepcopy(old)
    summary["case_details"], summary["splits"] = [], {}
    summary["protocol"] = plan["protocol"]
    for split, entries in plan["splits"].items():
        rows = []
        for entry in entries:
            case = entry["case_id"]
            source, detail = sources[case]
            count = int(detail["resampled_shape"][2])
            for kind, extension in (("image", "jpg"), ("mask", "png")):
                target = source / detail["split"] / kind / case
                assert len(list(target.glob("*." + extension))) == count, target
                link = root / split / kind / case
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(target, target_is_directory=True)
            rows.extend([(str(root / split / "image" / case / (case + "_%d.jpg" % z)),
                          str(root / split / "mask" / case / (case + "_%d.png" % z))) for z in range(count)])
            copied = dict(detail, split=split, official_split=entry["official_split"])
            summary["case_details"].append(copied)
        csv_path = root / "csv" / ("WORD_%s_2D_native07072.csv" % split)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image_pth", "mask_pth"])
            writer.writerows(rows)
        summary["splits"][split] = {"cases": len(entries), "slices_2d": len(rows)}
    dump(root / "metadata/split.json", plan)
    dump(root / "metadata/preprocess_summary.json", summary)
    subprocess.run([sys.executable, "tools/build_small_organ_roi_index.py", "--manifest",
                    str(root / "csv/WORD_Training_2D_native07072.csv"), "--output",
                    str(root / "metadata/small_organ_roi_index.json")], check=True)
    validate(root, args.plan)
    dump(root / "metadata/READY.json", {"protocol": plan["protocol"], "splits": summary["splits"],
         "plan_sha256": digest(args.plan), "manifests": {
             s: digest(root / "csv" / ("WORD_%s_2D_native07072.csv" % s)) for s in COUNTS}})


def validate(root, plan_path):
    plan = json.loads(plan_path.read_text())
    check_plan(plan)
    assert json.loads((root / "metadata/split.json").read_text()) == plan
    summary = json.loads((root / "metadata/preprocess_summary.json").read_text())
    assert all(summary["config"].get(k) == v for k, v in CONFIG.items())
    details = {d["case_id"]: d for d in summary["case_details"]}
    for split, entries in plan["splits"].items():
        with (root / "csv" / ("WORD_%s_2D_native07072.csv" % split)).open() as handle:
            rows = list(csv.DictReader(handle))
        expected = ["%s_%d" % (e["case_id"], z) for e in entries
                    for z in range(int(details[e["case_id"]]["resampled_shape"][2]))]
        assert [Path(r["image_pth"]).stem for r in rows] == expected
        assert [Path(r["mask_pth"]).stem for r in rows] == expected
        assert all(Path(p).is_file() for r in rows for p in r.values())
        assert summary["splits"][split] == {"cases": len(entries), "slices_2d": len(rows)}
    roi = json.loads((root / "metadata/small_organ_roi_index.json").read_text())
    assert roi["manifest_sha256"] == digest(root / "csv/WORD_Training_2D_native07072.csv")
    assert roi["sample_count"] == summary["splits"]["Training"]["slices_2d"]
    print(json.dumps(summary["splits"], indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("plan", "build", "validate"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--labels-zip", type=Path)
    parser.add_argument("--old", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "plan":
        if args.plan.exists():
            raise FileExistsError(args.plan)
        dump(args.plan, make_plan(args.raw))
    elif args.stage == "build":
        build(args)
    else:
        validate(args.output / "WORD", args.plan)
