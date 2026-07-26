"""Head-based OrganSlotBank metrics and case aggregation."""

import re

import torch


def binary_dice(prediction, target, eps=1e-8, empty_policy="nan"):
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum().float()
    denominator = prediction.sum().float() + target.sum().float()
    if denominator == 0:
        if empty_policy == "one":
            return torch.tensor(1.0)
        if empty_policy == "zero":
            return torch.tensor(0.0)
        if empty_policy == "nan":
            return torch.tensor(float("nan"))
        raise ValueError(f"unsupported empty_policy: {empty_policy}")
    return (2 * intersection + eps) / (denominator + eps)


def calibrated_multiclass_prediction(calibrated_logits, slot_order):
    logits = torch.cat([calibrated_logits[name] for name in slot_order], dim=1)
    return logits.argmax(dim=1)


def numeric_slice_index(value):
    match = re.search(r"_(\d+)(?:\.[^.]+)?$", str(value))
    if match is None:
        raise ValueError(f"cannot parse numeric slice index from {value}")
    return int(match.group(1))


def aggregate_slices_by_case(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record["case_id"], []).append(record)
    volumes = {}
    for case_id, case_records in grouped.items():
        ordered = sorted(case_records, key=lambda item: int(item["slice_index"]))
        volumes[case_id] = {
            "prediction": torch.stack(
                [item["prediction"] for item in ordered], dim=0
            ),
            "target": torch.stack(
                [item["target"] for item in ordered], dim=0
            ),
            "slice_indices": [int(item["slice_index"]) for item in ordered],
        }
    return volumes


def old_new_all_summary(per_class_dice, old_names, new_names):
    def mean_for(names):
        values = [float(per_class_dice[name]) for name in names]
        return sum(values) / len(values) if values else float("nan")

    all_names = tuple(old_names) + tuple(new_names)
    return {
        "old_dsc": mean_for(old_names),
        "new_dsc": mean_for(new_names),
        "all_dsc": mean_for(all_names),
    }


def forgetting(old_before, old_after):
    return float(old_before) - float(old_after)
