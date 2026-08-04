"""State-separated, area- and frequency-balanced ROI losses for OWT."""

from typing import Dict

import torch


def _state_summary(
    per_sample_class_loss: torch.Tensor,
    active: torch.Tensor,
    class_weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Summarize one class state without renormalizing the minibatch."""
    dtype = per_sample_class_loss.dtype
    weights = class_weights.to(
        device=per_sample_class_loss.device, dtype=dtype
    )
    active_weights = active.to(dtype) * weights.unsqueeze(0)
    batch_size = per_sample_class_loss.shape[0]

    loss = (per_sample_class_loss * active_weights).sum() / batch_size
    mass = active_weights.sum() / batch_size
    class_counts = active.sum(dim=0)
    class_losses = (
        (per_sample_class_loss * active).sum(dim=0)
        / class_counts.clamp_min(1).to(dtype)
    )
    valid_samples = (active_weights.sum(dim=1) > 0).sum()
    return {
        "loss": loss,
        "class_losses": class_losses,
        "class_counts": class_counts,
        "valid_samples": valid_samples,
        "mass": mass,
    }


def state_separated_roi_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    label: torch.Tensor,
    class_keep_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split present foreground classes into kept and removed states.

    ``target`` is the standard OWT masked reconstruction target: pixels of
    removed classes are zero, while pixels of kept classes retain the original
    image. The positive state is frequency weighted and optimized. The removed
    state is reported with equal foreground weights for monitoring only.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shapes differ: {pred.shape} vs {target.shape}"
        )
    if label.ndim != pred.ndim or label.shape[0] != pred.shape[0]:
        raise ValueError(
            f"label shape {label.shape} is incompatible with pred {pred.shape}"
        )
    if class_weights.ndim != 1 or class_weights.numel() < 2:
        raise ValueError(
            "class_weights must contain background and foreground entries"
        )
    if torch.any(class_weights < 0):
        raise ValueError("class_weights must be non-negative")

    batch_size = pred.shape[0]
    num_classes = class_weights.numel()
    if class_keep_mask.shape != (batch_size, num_classes):
        raise ValueError(
            "class_keep_mask must have shape "
            f"{(batch_size, num_classes)}, got {tuple(class_keep_mask.shape)}"
        )
    if class_keep_mask.dtype != torch.bool:
        raise ValueError("class_keep_mask must be boolean")

    labels = label[:, 0].long().reshape(batch_size, -1)
    voxel_error = (pred.float() - target.float()).pow(2).mean(dim=1)
    voxel_error = voxel_error.reshape(batch_size, -1)

    valid = (labels >= 0) & (labels < num_classes)
    safe_labels = labels.clamp(0, num_classes - 1)
    class_error_sums = voxel_error.new_zeros((batch_size, num_classes))
    class_counts = voxel_error.new_zeros((batch_size, num_classes))
    class_error_sums.scatter_add_(1, safe_labels, voxel_error * valid)
    class_counts.scatter_add_(1, safe_labels, valid.to(voxel_error.dtype))

    present = class_counts > 0
    per_sample_class_loss = class_error_sums / class_counts.clamp_min(1.0)

    foreground = torch.ones(
        num_classes, dtype=torch.bool, device=pred.device
    )
    foreground[0] = False
    keep = class_keep_mask.to(device=pred.device)
    positive_active = present & keep & foreground.unsqueeze(0)
    removed_active = present & (~keep) & foreground.unsqueeze(0)

    positive = _state_summary(
        per_sample_class_loss,
        positive_active,
        class_weights,
    )
    monitor_weights = class_weights.new_ones(num_classes)
    monitor_weights[0] = 0
    removed = _state_summary(
        per_sample_class_loss,
        removed_active,
        monitor_weights,
    )
    return {"positive": positive, "removed": removed}
