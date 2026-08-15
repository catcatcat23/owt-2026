"""Independently testable OrganSlotBank losses."""

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


def base_reconstruction_loss(reconstruction, target):
    return F.mse_loss(reconstruction, target)


def base_segmentation_loss(
    slot_logits,
    visible_masks,
    slot_names,
    slot_keep_mask,
    background_name="background",
    background_weight=0.25,
):
    """Average losses selected by an explicit per-sample/per-slot mask."""
    batch_size = slot_keep_mask.shape[0]
    total = next(iter(slot_logits.values())).sum() * 0.0
    weight_total = 0.0
    per_slot = {}
    for slot_index, name in enumerate(slot_names):
        logits = slot_logits[name]
        target = visible_masks[name].to(logits.device)
        sample_losses = []
        for sample_index in range(batch_size):
            if bool(slot_keep_mask[sample_index, slot_index]):
                sample_losses.append(
                    dice_bce_loss(
                        logits[sample_index:sample_index + 1],
                        target[sample_index:sample_index + 1],
                    )
                )
        if not sample_losses:
            per_slot[name] = logits.sum() * 0.0
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
