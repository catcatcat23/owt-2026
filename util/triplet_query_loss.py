"""Loss helpers for PSEM-v3 anchor-and-context triplets."""

from typing import Dict

import torch

from util.state_separated_balanced_loss import state_separated_roi_l2


def triplet_delta_loss(
    pred_context: torch.Tensor,
    pred_plus: torch.Tensor,
    target_direct: torch.Tensor,
    label: torch.Tensor,
    direct_mask: torch.Tensor,
    class_weights: torch.Tensor,
    positive_roi_loss_weight: float,
) -> Dict[str, torch.Tensor]:
    """Supervise the image increment caused by adding the anchor query."""
    if pred_context.shape != pred_plus.shape:
        raise ValueError("context and plus predictions must have the same shape")
    if pred_plus.shape != target_direct.shape:
        raise ValueError("delta prediction and direct target shapes must match")

    pred_delta = pred_plus - pred_context
    global_loss = (pred_delta - target_direct).pow(2).mean()
    state_stats = state_separated_roi_l2(
        pred_delta,
        target_direct,
        label,
        class_keep_mask=direct_mask,
        class_weights=class_weights,
    )
    positive = state_stats["positive"]
    loss = global_loss + positive_roi_loss_weight * positive["loss"]
    return {
        "loss": loss,
        "prediction": pred_delta,
        "global_loss": global_loss,
        "positive_roi_loss": positive["loss"],
        "positive_class_losses": positive["class_losses"],
        "positive_class_counts": positive["class_counts"],
        "positive_valid_samples": positive["valid_samples"],
        "positive_weighted_mass": positive["mass"],
    }
