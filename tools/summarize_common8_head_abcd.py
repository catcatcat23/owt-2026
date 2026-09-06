#!/usr/bin/env python3
"""Build a fixed-protocol A/B/C/D WORD head comparison table."""

import argparse
import csv
import json
from pathlib import Path


CLASS_IDS = tuple(range(1, 9))
MODE_ORDER = ("binary_post", "selected_post")
MODE_LABELS = {
    "binary_post": "fixed_0.5_post",
    "selected_post": "train_calibrated_post",
}


def parse_args():
    parser = argparse.ArgumentParser("Summarize Common8 OrganSlot head arms")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="LABEL=/absolute/path/to/heads_test/results.json",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_arm(value):
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise ValueError("--arm must be LABEL=/path/results.json")
    return label, Path(path).resolve()


def mean(values):
    values = tuple(values)
    return sum(values) / len(values)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    arms = [parse_arm(value) for value in args.arm]
    labels = [label for label, _ in arms]
    if len(labels) != len(set(labels)):
        raise ValueError("arm labels must be unique")

    rows = []
    provenance = {}
    class_names = None
    for arm_label, result_path in arms:
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        result = json.loads(result_path.read_text())
        if result.get("complete_split") is not True:
            raise ValueError("{} is not a complete test split".format(result_path))
        if int(result.get("samples", -1)) != 6990:
            raise ValueError("{} does not contain 6990 test slices".format(result_path))
        load = result.get("checkpoint_load", {})
        if int(load.get("epoch", -1)) != 802 or load.get("exact") is not True:
            raise ValueError("{} did not exactly load checkpoint-802".format(result_path))
        metrics = result.get("metrics", {})
        available_modes = [mode for mode in MODE_ORDER if mode in metrics]
        if "binary_post" not in available_modes:
            raise ValueError("{} lacks fixed-0.5 post metrics".format(result_path))
        names = {
            class_id: metrics["binary_post"][str(class_id)]["class_name"]
            for class_id in CLASS_IDS
        }
        if class_names is None:
            class_names = names
        elif class_names != names:
            raise ValueError("class-name mapping differs across arms")
        provenance[arm_label] = {
            "result_path": str(result_path),
            "checkpoint_load": load,
            "selected_thresholds": result.get("selected_thresholds"),
        }
        for mode in available_modes:
            dice = {
                class_id: float(
                    metrics[mode][str(class_id)]["case_dice_presence_mean"]
                )
                for class_id in CLASS_IDS
            }
            iou = {
                class_id: float(
                    metrics[mode][str(class_id)]["case_iou_presence_mean"]
                )
                for class_id in CLASS_IDS
            }
            volume_ratio = {
                class_id: float(
                    metrics[mode][str(class_id)]["prediction_to_target_volume_ratio"]
                )
                for class_id in CLASS_IDS
            }
            row = {
                "arm": arm_label,
                "head_type": load.get("slot_head_type"),
                "mode": MODE_LABELS[mode],
                "mean_dice": mean(dice.values()),
                "small_organ_mean_dice": mean(dice[class_id] for class_id in (4, 5, 6)),
                "mean_iou": mean(iou.values()),
                "mean_prediction_to_target_volume_ratio": mean(volume_ratio.values()),
            }
            row.update(
                {"dice_{}_{}".format(class_id, class_names[class_id]): dice[class_id]
                 for class_id in CLASS_IDS}
            )
            rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / "abcd_head_comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = ["arm", "head_type", "mode"] + [
        class_names[class_id] for class_id in CLASS_IDS
    ] + ["mean", "small-organ mean"]
    markdown = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = [row["arm"], str(row["head_type"]), row["mode"]]
        values.extend(
            "{:.2f}".format(100.0 * row["dice_{}_{}".format(class_id, class_names[class_id])])
            for class_id in CLASS_IDS
        )
        values.extend([
            "{:.2f}".format(100.0 * row["mean_dice"]),
            "{:.2f}".format(100.0 * row["small_organ_mean_dice"]),
        ])
        markdown.append("| " + " | ".join(values) + " |")
    (output_dir / "abcd_head_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    (output_dir / "abcd_head_comparison.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "dataset": "WORD Common8",
                    "spacing_mm": [0.7, 0.7, 2.0],
                    "test_slices": 6990,
                    "test_cases": 24,
                    "checkpoint_epoch": 802,
                    "modes": MODE_LABELS,
                },
                "class_names": class_names,
                "provenance": provenance,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(output_dir / "abcd_head_comparison.md")


if __name__ == "__main__":
    main()
