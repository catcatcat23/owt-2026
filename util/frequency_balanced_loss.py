"""Area- and sample-frequency-balanced reconstruction losses for OWT."""

from typing import Sequence, Tuple

import torch


def build_class_frequency_weights(
    positive_sample_counts: Sequence[int],
    dataset_size: int,
    alpha: float = 0.5,
    max_weight_ratio: float = 4.0,
) -> torch.Tensor:
    """Build foreground weights with unit expected mass per uniform sample.

    ``alpha=1`` gives exact inverse positive-sample-frequency weighting.
    ``alpha=0`` reduces to equal pair weighting. The final normalization keeps
    the expected weighted foreground mass at one, so the ROI-loss scale remains
    comparable across alpha values.
    """
    counts = torch.as_tensor(positive_sample_counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("positive_sample_counts must be a non-empty 1D sequence")
    if torch.any(counts <= 0):
        raise ValueError("every foreground class must have a positive sample count")
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if torch.any(counts > dataset_size):
        raise ValueError("positive sample counts cannot exceed dataset_size")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if max_weight_ratio < 1.0:
        raise ValueError("max_weight_ratio must be at least 1")

    raw_weights = counts.pow(-alpha)
    raw_weights = raw_weights / raw_weights.min()
    raw_weights = raw_weights.clamp(max=max_weight_ratio)

    expected_mass = (counts * raw_weights).sum() / float(dataset_size)
    foreground_weights = raw_weights / expected_mass

    # Class 0 is background and remains excluded from the ROI objective.
    return torch.cat((foreground_weights.new_zeros(1), foreground_weights)).float()


def frequency_balanced_roi_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    class_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average pixels per sample/class and frequency-weight occurrences.

    Returns the scalar ROI loss, unweighted per-class losses, per-class sample
    counts, number of samples containing a weighted class, and observed weighted
    mass per sample.
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
    if label.ndim != pred.ndim or label.shape[0] != pred.shape[0]:
        raise ValueError(f"label shape {label.shape} is incompatible with pred {pred.shape}")
    if class_weights.ndim != 1 or class_weights.numel() < 2:
        raise ValueError("class_weights must contain background and foreground entries")
    if torch.any(class_weights < 0):
        raise ValueError("class_weights must be non-negative")

    num_classes = class_weights.numel()
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
    weights = class_weights.to(device=pred.device, dtype=voxel_error.dtype)
    active_weights = present.to(voxel_error.dtype) * weights.unsqueeze(0)

    # Dataset-frequency weights carry a fixed normalization. Dividing by B
    # yields an unbiased estimate of the class-balanced full-dataset objective.
    roi_loss = (per_sample_class_loss * active_weights).sum() / pred.shape[0]
    weighted_mass = active_weights.sum() / pred.shape[0]

    per_class_sample_count = present.sum(dim=0)
    per_class_loss = (
        (per_sample_class_loss * present).sum(dim=0)
        / per_class_sample_count.clamp_min(1).to(voxel_error.dtype)
    )
    valid_samples = (active_weights.sum(dim=1) > 0).sum()
    return roi_loss, per_class_loss, per_class_sample_count, valid_samples, weighted_mass
