"""Organ-balanced reconstruction losses for 2D and short-window 3D OWT."""

from typing import Tuple

import torch


def organ_balanced_roi_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    background_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average L2 inside each present class, then across samples.

    Returns the scalar ROI loss, per-class losses, per-class sample counts, and
    the number of samples containing at least one weighted class.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
    if label.ndim != pred.ndim or label.shape[0] != pred.shape[0]:
        raise ValueError(f"label shape {label.shape} is incompatible with pred {pred.shape}")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if background_weight < 0:
        raise ValueError("background_weight must be non-negative")

    # Labels are repeated over the image-channel dimension by the legacy loader.
    labels = label[:, 0].long().reshape(label.shape[0], -1)
    voxel_error = (pred.float() - target.float()).pow(2).mean(dim=1)
    voxel_error = voxel_error.reshape(pred.shape[0], -1)

    valid = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(0, num_classes - 1)
    class_error_sums = voxel_error.new_zeros((pred.shape[0], num_classes))
    class_counts = voxel_error.new_zeros((pred.shape[0], num_classes))
    class_error_sums.scatter_add_(1, safe_labels, voxel_error * valid)
    class_counts.scatter_add_(1, safe_labels, valid.to(voxel_error.dtype))

    present = class_counts > 0
    per_sample_class_loss = class_error_sums / class_counts.clamp_min(1.0)

    class_weights = voxel_error.new_ones(num_classes)
    class_weights[0] = background_weight
    active_weights = present.to(voxel_error.dtype) * class_weights.unsqueeze(0)
    sample_denominator = active_weights.sum(dim=1)
    valid_samples = sample_denominator > 0
    sample_losses = (
        (per_sample_class_loss * active_weights).sum(dim=1)
        / sample_denominator.clamp_min(1.0)
    )

    if valid_samples.any():
        roi_loss = sample_losses[valid_samples].mean()
    else:
        roi_loss = pred.float().sum() * 0.0

    per_class_sample_count = present.sum(dim=0)
    per_class_loss = (
        (per_sample_class_loss * present).sum(dim=0)
        / per_class_sample_count.clamp_min(1).to(voxel_error.dtype)
    )
    return roi_loss, per_class_loss, per_class_sample_count, valid_samples.sum()
