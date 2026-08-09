"""LossBalance-v3 supervision for strictly isolated OrganSlot masks."""

from typing import Dict, Mapping, Sequence

import torch


def build_class_frequency_weights(
    positive_sample_counts: Sequence[int],
    dataset_size: int,
    alpha: float = 0.5,
    max_weight_ratio: float = 4.0,
) -> torch.Tensor:
    """Return background-zeroed foreground weights with unit expected mass."""
    counts = torch.as_tensor(positive_sample_counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("positive_sample_counts must be a non-empty 1D sequence")
    if torch.any(counts <= 0):
        raise ValueError("every foreground class must have a positive sample count")
    if dataset_size <= 0 or torch.any(counts > dataset_size):
        raise ValueError("positive sample counts must be within the dataset size")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be at least 1")

    raw_weights = counts.pow(-alpha)
    raw_weights = (raw_weights / raw_weights.min()).clamp(max=max_weight_ratio)
    expected_mass = (counts * raw_weights).sum() / float(dataset_size)
    foreground_weights = raw_weights / expected_mass
    return torch.cat((foreground_weights.new_zeros(1), foreground_weights)).float()


def _summarize(
    per_sample_class_loss: torch.Tensor,
    active: torch.Tensor,
    class_weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    dtype = per_sample_class_loss.dtype
    weights = class_weights.to(device=per_sample_class_loss.device, dtype=dtype)
    active_weights = active.to(dtype) * weights.unsqueeze(0)
    batch_size = per_sample_class_loss.shape[0]
    class_counts = active.sum(dim=0)
    return {
        "loss": (per_sample_class_loss * active_weights).sum() / batch_size,
        "class_losses": (
            (per_sample_class_loss * active).sum(dim=0)
            / class_counts.clamp_min(1).to(dtype)
        ),
        "class_counts": class_counts,
        "valid_samples": (active_weights.sum(dim=1) > 0).sum(),
        "mass": active_weights.sum() / batch_size,
    }


def state_separated_mask_roi_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_masks: Mapping[str, torch.Tensor],
    slot_names: Sequence[str],
    class_keep_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compute LossBalance-v3 from visible masks without exposing raw labels.

    Only present and kept foreground slots contribute to ``positive.loss``.
    Present removed foreground slots are reported for monitoring and receive no
    gradient from the returned positive objective.
    """
    if pred.shape != target.shape:
        raise ValueError("pred and target shapes differ")
    batch_size = pred.shape[0]
    slot_count = len(slot_names)
    if class_keep_mask.shape != (batch_size, slot_count):
        raise ValueError("class_keep_mask has the wrong shape")
    if class_keep_mask.dtype != torch.bool:
        raise ValueError("class_keep_mask must be boolean")
    if class_weights.shape != (slot_count,):
        raise ValueError("class_weights must align with slot_names")
    if torch.any(class_weights < 0):
        raise ValueError("class_weights must be non-negative")
    if set(visible_masks) != set(slot_names):
        raise ValueError("visible_masks must exactly match slot_names")

    voxel_error = (pred.float() - target.float()).pow(2).mean(dim=1)
    per_class_losses = []
    per_class_present = []
    for name in slot_names:
        mask = visible_masks[name].to(device=pred.device, dtype=voxel_error.dtype)
        if mask.shape[0] != batch_size or mask.shape[1] != 1:
            raise ValueError("visible mask {} has an incompatible shape".format(name))
        mask = mask[:, 0]
        if mask.shape != voxel_error.shape:
            raise ValueError("visible mask {} does not align with prediction".format(name))
        counts = mask.reshape(batch_size, -1).sum(dim=1)
        error_sums = (voxel_error * mask).reshape(batch_size, -1).sum(dim=1)
        per_class_losses.append(error_sums / counts.clamp_min(1.0))
        per_class_present.append(counts > 0)

    per_sample_class_loss = torch.stack(per_class_losses, dim=1)
    present = torch.stack(per_class_present, dim=1)
    keep = class_keep_mask.to(device=pred.device)
    foreground = class_weights.to(device=pred.device) > 0
    positive_active = present & keep & foreground.unsqueeze(0)
    removed_active = present & (~keep) & foreground.unsqueeze(0)

    positive = _summarize(per_sample_class_loss, positive_active, class_weights)
    monitor_weights = class_weights.new_ones(slot_count)
    monitor_weights[~foreground.to(device=monitor_weights.device)] = 0
    removed = _summarize(per_sample_class_loss, removed_active, monitor_weights)
    return {"positive": positive, "removed": removed}
