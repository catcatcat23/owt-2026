"""Strict stage-wise label visibility for OrganSlotBank training."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClassSpec:
    name: str
    raw_id: int
    is_background: bool = False


@dataclass(frozen=True)
class StageSpec:
    name: str
    stage: str
    visible_slots: Tuple[str, ...]
    new_slots: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.stage not in {"base", "incremental", "evaluation"}:
            raise ValueError(f"unsupported stage: {self.stage}")
        if not self.visible_slots:
            raise ValueError("visible_slots must not be empty")
        if self.stage == "incremental":
            if not self.new_slots:
                raise ValueError("incremental stage requires new_slots")
            if set(self.visible_slots) != set(self.new_slots):
                raise ValueError(
                    "incremental training may expose only current new slots"
                )


def load_visibility_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    classes = tuple(ClassSpec(**item) for item in config["classes"])
    stages = {
        name: StageSpec(
            name=name,
            stage=value["stage"],
            visible_slots=tuple(value["visible_slots"]),
            new_slots=tuple(value.get("new_slots", ())),
        )
        for name, value in config["stages"].items()
    }
    return classes, stages


def _label_map(raw_label):
    if not torch.is_tensor(raw_label):
        raw_label = torch.as_tensor(raw_label)
    if raw_label.ndim not in (3, 4, 5):
        raise ValueError(
            "raw_label must be [C,H,W], [C,T,H,W], or batched equivalent"
        )
    # Legacy OWT repeats an integer mask over three channels.
    if raw_label.ndim in (3, 4):
        return raw_label[0].long()
    return raw_label[:, 0].long()


def validate_raw_label_ids(raw_label, class_specs):
    label_map = _label_map(raw_label)
    allowed = {int(spec.raw_id) for spec in class_specs}
    observed = {int(value) for value in torch.unique(label_map).tolist()}
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise ValueError(f"unexpected raw label ids: {unexpected}")
    return observed


def build_provisional_background(visible_organ_masks):
    masks = list(visible_organ_masks.values())
    if not masks:
        raise ValueError("at least one visible organ mask is required")
    union = torch.zeros_like(masks[0], dtype=torch.bool)
    for mask in masks:
        if mask.shape != union.shape:
            raise ValueError("visible organ masks must share a shape")
        union |= mask.bool()
    return ~union


def build_visible_binary_masks(raw_label, class_specs, stage_spec):
    """Convert a full raw map inside the wrapper, returning visible masks only."""
    validate_raw_label_ids(raw_label, class_specs)
    label_map = _label_map(raw_label)
    by_name = {spec.name: spec for spec in class_specs}
    unknown = set(stage_spec.visible_slots) - set(by_name)
    if unknown:
        raise ValueError(f"stage references unknown slots: {sorted(unknown)}")

    masks = {}
    for name in stage_spec.visible_slots:
        spec = by_name[name]
        if spec.is_background:
            continue
        masks[name] = (label_map == spec.raw_id).unsqueeze(0).float()

    if stage_spec.stage == "base":
        backgrounds = [
            spec for spec in class_specs if spec.is_background
            and spec.name in stage_spec.visible_slots
        ]
        if len(backgrounds) != 1:
            raise ValueError("base stage requires exactly one visible background")
        masks[backgrounds[0].name] = build_provisional_background(masks).float()
    elif any(by_name[name].is_background for name in stage_spec.visible_slots):
        raise ValueError("incremental training must not expose background GT")

    return {name: masks[name] for name in stage_spec.visible_slots}


def assert_case_splits_disjoint(split_case_ids):
    names = list(split_case_ids)
    sets = {name: set(split_case_ids[name]) for name in names}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise ValueError(
                    f"case leakage between {left} and {right}: {sorted(overlap)}"
                )


class StrictVisibilityDataset(Dataset):
    """Hide the full label map at the dataset boundary used by training."""

    def __init__(
        self,
        dataset,
        class_specs,
        stage_spec,
        sample_indices: Optional[Sequence[int]] = None,
    ):
        self.dataset = dataset
        self.class_specs = tuple(class_specs)
        self.stage_spec = stage_spec
        self.sample_indices = (
            tuple(sample_indices) if sample_indices is not None else None
        )

    def __len__(self):
        return (
            len(self.sample_indices)
            if self.sample_indices is not None
            else len(self.dataset)
        )

    def __getitem__(self, index):
        source_index = (
            self.sample_indices[index]
            if self.sample_indices is not None
            else index
        )
        sample = self.dataset[source_index]
        if "label" not in sample:
            raise KeyError("wrapped dataset sample must contain label")
        visible_masks = build_visible_binary_masks(
            sample["label"], self.class_specs, self.stage_spec
        )
        output = {
            "image": sample["image"],
            "visible_masks": visible_masks,
            "sample_index": int(sample.get("sample_index", source_index)),
        }
        case_name = sample.get("case_name")
        if case_name is not None:
            if isinstance(case_name, (tuple, list)):
                output["case_id"] = str(case_name[0])
            else:
                output["case_id"] = str(case_name)
        if "slice_index" in sample:
            output["slice_index"] = int(sample["slice_index"])
        for key in ("focus_class_id", "roi_applied", "crop_box"):
            if key in sample:
                output[key] = sample[key]
        return output


class EvaluationVisibilityDataset(Dataset):
    """Evaluation-only wrapper that may expose the complete raw label map."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        if "label" not in sample:
            raise KeyError("evaluation sample must contain label")
        sample["full_label"] = sample.pop("label")
        return sample
