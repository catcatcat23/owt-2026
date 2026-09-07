"""Independently testable OrganSlotBank losses."""

import math

import torch
import torch.nn.functional as F


def masked_mean(values, mask, eps=1e-8):
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(eps)


def soft_dice_loss(logits, target, eps=1e-6):
    target = target.to(device=logits.device, dtype=logits.dtype)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    intersection = (probabilities * target).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target.sum(dim=dims)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def dice_bce_loss(logits, target):
    target = target.to(device=logits.device, dtype=logits.dtype)
    return soft_dice_loss(logits, target) + F.binary_cross_entropy_with_logits(
        logits, target
    )


def sigmoid_focal_loss(logits, target, alpha=0.75, gamma=2.0):
    """Standard binary focal loss with ``alpha`` weighting the positive class."""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("focal alpha must be in [0, 1]")
    if float(gamma) < 0.0:
        raise ValueError("focal gamma must be non-negative")
    target = target.to(device=logits.device, dtype=logits.dtype)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    probabilities = torch.sigmoid(logits)
    probability_target = probabilities * target + (1.0 - probabilities) * (
        1.0 - target
    )
    alpha_target = float(alpha) * target + (1.0 - float(alpha)) * (
        1.0 - target
    )
    return (
        alpha_target
        * (1.0 - probability_target).pow(float(gamma))
        * cross_entropy
    ).mean()


def _topk_mean(values, ratio):
    if not 0.0 < float(ratio) <= 1.0:
        raise ValueError("hard-negative ratio must be in (0, 1]")
    values = values.reshape(-1)
    if values.numel() == 0:
        raise ValueError("hard-negative mining requires at least one value")
    count = max(1, int(math.ceil(values.numel() * float(ratio))))
    return torch.topk(values, count, sorted=False).values.mean()


def tversky_loss(logits, target, alpha_fp=0.3, beta_fn=0.7, eps=1e-6):
    """Per-sample binary Tversky loss with explicit FP/FN coefficients."""
    logits = logits.float()
    target = target.to(device=logits.device, dtype=logits.dtype)
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, logits.ndim))
    true_positive = (probabilities * target).sum(dim=dims)
    false_positive = (probabilities * (1.0 - target)).sum(dim=dims)
    false_negative = ((1.0 - probabilities) * target).sum(dim=dims)
    score = (true_positive + float(eps)) / (
        true_positive
        + float(alpha_fp) * false_positive
        + float(beta_fn) * false_negative
        + float(eps)
    )
    return 1.0 - score.mean()


def small_organ_segmentation_loss(
    logits,
    target,
    focal_alpha=0.75,
    focal_gamma=2.0,
    tversky_alpha_fp=0.3,
    tversky_beta_fn=0.7,
    tversky_eps=1e-6,
    balanced_focal_weight=0.5,
    hard_negative_ratio=0.02,
    negative_slice_weight=0.1,
):
    """Small-organ loss for one sample, separating positive and empty slices."""
    logits = logits.float()
    target = target.to(device=logits.device, dtype=logits.dtype)
    if logits.shape[0] != 1:
        raise ValueError("small-organ loss expects one sample at a time")
    target_positive = target > 0.5
    zero = logits.sum() * 0.0

    if not torch.any(target_positive):
        empty_bce = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )
        hard_negative_bce = _topk_mean(empty_bce, hard_negative_ratio)
        return float(negative_slice_weight) * hard_negative_bce, {
            "is_positive": False,
            "tversky_loss": zero,
            "positive_focal_loss": zero,
            "hard_negative_focal_loss": zero,
            "empty_negative_loss": hard_negative_bce,
        }

    probabilities = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    positive_focal = (
        float(focal_alpha)
        * (1.0 - probabilities[target_positive]).pow(float(focal_gamma))
        * cross_entropy[target_positive]
    ).mean()
    background = ~target_positive
    if torch.any(background):
        negative_focal_values = (
            (1.0 - float(focal_alpha))
            * probabilities[background].pow(float(focal_gamma))
            * cross_entropy[background]
        )
        hard_negative_focal = _topk_mean(
            negative_focal_values, hard_negative_ratio
        )
    else:
        hard_negative_focal = zero
    overlap = tversky_loss(
        logits,
        target,
        alpha_fp=tversky_alpha_fp,
        beta_fn=tversky_beta_fn,
        eps=tversky_eps,
    )
    loss = overlap + float(balanced_focal_weight) * (
        positive_focal + hard_negative_focal
    )
    return loss, {
        "is_positive": True,
        "tversky_loss": overlap,
        "positive_focal_loss": positive_focal,
        "hard_negative_focal_loss": hard_negative_focal,
        "empty_negative_loss": zero,
    }


def binary_segmentation_loss(
    logits,
    target,
    loss_type,
    focal_alpha,
    focal_gamma,
    tversky_alpha_fp=0.3,
    tversky_beta_fn=0.7,
    tversky_eps=1e-6,
    balanced_focal_weight=0.5,
    hard_negative_ratio=0.02,
    negative_slice_weight=0.1,
):
    if loss_type == "dice_bce":
        return dice_bce_loss(logits, target)
    if loss_type == "focal":
        return sigmoid_focal_loss(
            logits,
            target,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    if loss_type == "small_organ":
        loss, _ = small_organ_segmentation_loss(
            logits,
            target,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            tversky_alpha_fp=tversky_alpha_fp,
            tversky_beta_fn=tversky_beta_fn,
            tversky_eps=tversky_eps,
            balanced_focal_weight=balanced_focal_weight,
            hard_negative_ratio=hard_negative_ratio,
            negative_slice_weight=negative_slice_weight,
        )
        return loss
    raise ValueError("unknown segmentation loss type: {}".format(loss_type))


def base_reconstruction_loss(reconstruction, target):
    return F.mse_loss(reconstruction, target)


def _mean_or_zero(values, zero):
    return torch.stack(values).mean() if values else zero


def base_segmentation_loss(
    slot_logits,
    visible_masks,
    slot_names,
    slot_keep_mask,
    background_name="background",
    background_weight=0.25,
    loss_type="dice_bce",
    focal_alpha=0.75,
    focal_gamma=2.0,
    tversky_alpha_fp=0.3,
    tversky_beta_fn=0.7,
    tversky_eps=1e-6,
    balanced_focal_weight=0.5,
    hard_negative_ratio=0.02,
    negative_slice_weight=0.1,
    diagnostics=None,
    segmentation_unit="slab",
):
    """Average losses selected by an explicit per-sample/per-slot mask."""
    batch_size = slot_keep_mask.shape[0]
    total = next(iter(slot_logits.values())).sum() * 0.0
    weight_total = 0.0
    per_slot = {}
    for slot_index, name in enumerate(slot_names):
        logits = slot_logits[name]
        target = visible_masks[name].to(logits.device)
        keep_for_slot = slot_keep_mask[:, slot_index]
        # Preserve 2D behavior; independently supervise temporal planes in 3D.
        if segmentation_unit not in ("slab", "slice"):
            raise ValueError("segmentation_unit must be slab or slice")
        if loss_type == "small_organ" and logits.ndim == 5 and segmentation_unit == "slice":
            frames = logits.shape[2]
            logits = logits.permute(0, 2, 1, 3, 4).reshape(
                -1, logits.shape[1], logits.shape[3], logits.shape[4]
            )
            target = target.permute(0, 2, 1, 3, 4).reshape_as(logits)
            keep_for_slot = keep_for_slot.repeat_interleave(frames)
        zero = logits.sum() * 0.0
        sample_losses = []
        positive_losses = []
        negative_losses = []
        positive_details = {
            "tversky_loss": [],
            "positive_focal_loss": [],
            "hard_negative_focal_loss": [],
        }
        empty_negative_losses = []
        if loss_type == "small_organ" and diagnostics is not None:
            diagnostics[name] = {
                "positive_samples": zero,
                "negative_samples": zero,
                "tversky_loss": zero,
                "positive_focal_loss": zero,
                "hard_negative_focal_loss": zero,
                "empty_negative_loss": zero,
            }
        for sample_index in range(logits.shape[0]):
            if not bool(keep_for_slot[sample_index]):
                continue
            sample_logits = logits[sample_index:sample_index + 1]
            sample_target = target[sample_index:sample_index + 1]
            if loss_type != "small_organ":
                sample_losses.append(
                    binary_segmentation_loss(
                        sample_logits,
                        sample_target,
                        loss_type,
                        focal_alpha,
                        focal_gamma,
                    )
                )
                continue
            sample_loss, details = small_organ_segmentation_loss(
                sample_logits,
                sample_target,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
                tversky_alpha_fp=tversky_alpha_fp,
                tversky_beta_fn=tversky_beta_fn,
                tversky_eps=tversky_eps,
                balanced_focal_weight=balanced_focal_weight,
                hard_negative_ratio=hard_negative_ratio,
                negative_slice_weight=negative_slice_weight,
            )
            if details["is_positive"]:
                positive_losses.append(sample_loss)
                for key in positive_details:
                    positive_details[key].append(details[key])
            else:
                negative_losses.append(sample_loss)
                empty_negative_losses.append(details["empty_negative_loss"])

        if loss_type == "small_organ":
            if not positive_losses and not negative_losses:
                per_slot[name] = zero
                continue
            slot_loss = _mean_or_zero(positive_losses, zero)
            slot_loss = slot_loss + _mean_or_zero(negative_losses, zero)
            if diagnostics is not None:
                diagnostics[name] = {
                    "positive_samples": zero + len(positive_losses),
                    "negative_samples": zero + len(negative_losses),
                    "tversky_loss": _mean_or_zero(
                        positive_details["tversky_loss"], zero
                    ),
                    "positive_focal_loss": _mean_or_zero(
                        positive_details["positive_focal_loss"], zero
                    ),
                    "hard_negative_focal_loss": _mean_or_zero(
                        positive_details["hard_negative_focal_loss"], zero
                    ),
                    "empty_negative_loss": _mean_or_zero(
                        empty_negative_losses, zero
                    ),
                }
        else:
            if not sample_losses:
                per_slot[name] = zero
                continue
            slot_loss = torch.stack(sample_losses).mean()
        weight = background_weight if name == background_name else 1.0
        per_slot[name] = slot_loss
        total = total + weight * slot_loss
        weight_total += weight
    if weight_total == 0:
        return total, per_slot
    return total / weight_total, per_slot


def new_region_reconstruction_loss(reconstruction, image, new_mask):
    channel_mask = new_mask.to(reconstruction.device)
    return masked_mean((reconstruction - image).pow(2), channel_mask)


def old_confidence_suppression_loss(
    new_logits,
    frozen_old_logits,
    new_mask,
    threshold=0.7,
):
    with torch.no_grad():
        old_union = torch.zeros_like(new_mask, dtype=torch.bool)
        for logits in frozen_old_logits.values():
            old_union |= torch.sigmoid(logits.detach()) > threshold
        suppress_mask = old_union & ~new_mask.bool()
    if not torch.any(suppress_mask):
        return new_logits.sum() * 0.0, suppress_mask
    zeros = torch.zeros_like(new_logits)
    loss = masked_mean(
        F.binary_cross_entropy_with_logits(
            new_logits, zeros, reduction="none"
        ),
        suppress_mask,
    )
    return loss, suppress_mask


def background_new_complementarity_loss(background_logits, new_mask):
    zeros = torch.zeros_like(background_logits)
    return masked_mean(
        F.binary_cross_entropy_with_logits(
            background_logits, zeros, reduction="none"
        ),
        new_mask,
    )
