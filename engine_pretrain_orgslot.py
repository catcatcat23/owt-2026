"""Stage-explicit training helpers for OrganSlotBank."""

import torch

from eval_orgslot import binary_dice
from losses_orgslot import (
    base_reconstruction_loss,
    base_segmentation_loss,
    dice_bce_loss,
    new_region_reconstruction_loss,
    old_confidence_suppression_loss,
)


def perceptual_reconstruction_loss(model, reconstruction, target):
    """Match legacy OWT LPIPS behavior for 2D and slice-wise 3D inputs."""
    if not hasattr(model, "perceptual_loss"):
        return reconstruction.sum() * 0.0
    if reconstruction.ndim == 4:
        return model.perceptual_loss(
            target.contiguous(), reconstruction.contiguous()
        ).mean()
    losses = []
    for slice_index in range(reconstruction.shape[2]):
        losses.append(model.perceptual_loss(
            target[:, :, slice_index].contiguous(),
            reconstruction[:, :, slice_index].contiguous(),
        ).mean())
    return torch.stack(losses).mean()


from util.slot_tgr import (
    build_base_reconstruction_target,
    sample_base_slot_keep_mask,
    sample_retain_new_keep_mask,
)


def compute_base_objective(
    model,
    batch,
    epoch,
    lambda_seg=1.0,
    lambda_lpips=1.0,
    background_weight=0.25,
    tgr_seed=0,
):
    image = batch["image"]
    visible_masks = batch["visible_masks"]
    sample_indices = batch["sample_index"].to(image.device)
    keep = sample_base_slot_keep_mask(
        sample_indices,
        epoch,
        len(model.slot_names),
        seed=tgr_seed,
    )
    target = build_base_reconstruction_target(
        image, visible_masks, model.slot_names, keep
    )
    output = model(image, slot_keep_mask=keep)
    reconstruction = base_reconstruction_loss(
        output["reconstruction"], target
    )
    perceptual = perceptual_reconstruction_loss(
        model, output["reconstruction"], target
    )
    segmentation, per_slot = base_segmentation_loss(
        output["calibrated_logits"],
        visible_masks,
        model.slot_names,
        keep,
        background_weight=background_weight,
    )
    total = (
        reconstruction
        + float(lambda_lpips) * perceptual
        + float(lambda_seg) * segmentation
    )
    return total, {
        "total_loss": total,
        "reconstruction_loss": reconstruction,
        "perceptual_loss": perceptual,
        "segmentation_loss": segmentation,
        "per_slot_loss": per_slot,
        "slot_keep_mask": keep,
        "output": output,
        "target": target,
    }


def compute_incremental_minimal_objective(
    model,
    batch,
    new_slot,
    epoch,
    lambda_rec=0.1,
    lambda_sup=0.0,
    frozen_old_logits=None,
    old_slot_names=(),
    old_confidence_threshold=0.7,
    tgr_seed=0,
):
    """Incremental API accepts visible new mask only, never a full/old label."""
    image = batch["image"]
    visible_masks = batch["visible_masks"]
    if set(visible_masks) != {new_slot}:
        raise ValueError(
            "incremental batch must expose exactly the current new slot"
        )
    new_mask = visible_masks[new_slot]
    sample_indices = batch["sample_index"].to(image.device)
    keep = sample_retain_new_keep_mask(
        sample_indices,
        epoch,
        list(model.slot_names),
        new_slot,
        seed=tgr_seed,
        minimal=True,
    )
    output = model(image, slot_keep_mask=keep)
    if lambda_sup and frozen_old_logits is None:
        frozen_old_logits = {
            name: output["calibrated_logits"][name].detach()
            for name in old_slot_names
        }
    segmentation = dice_bce_loss(
        output["calibrated_logits"][new_slot], new_mask
    )
    reconstruction = new_region_reconstruction_loss(
        output["reconstruction"], image, new_mask
    )
    suppression = image.sum() * 0.0
    suppress_mask = torch.zeros_like(new_mask, dtype=torch.bool)
    if lambda_sup and frozen_old_logits:
        suppression, suppress_mask = old_confidence_suppression_loss(
            output["calibrated_logits"][new_slot],
            frozen_old_logits,
            new_mask,
            threshold=old_confidence_threshold,
        )
    total = (
        segmentation
        + float(lambda_rec) * reconstruction
        + float(lambda_sup) * suppression
    )
    return total, {
        "total_loss": total,
        "segmentation_loss": segmentation,
        "new_region_reconstruction_loss": reconstruction,
        "suppression_loss": suppression,
        "suppression_mask": suppress_mask,
        "slot_keep_mask": keep,
        "output": output,
    }


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    stage,
    new_slot=None,
    lambda_seg=1.0,
    lambda_lpips=1.0,
    lambda_rec=0.1,
    lambda_sup=0.0,
    background_weight=0.25,
    old_slot_names=(),
    old_confidence_threshold=0.7,
    max_steps=None,
):
    model.train()
    totals = {}
    processed_steps = 0
    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        batch["image"] = batch["image"].to(device)
        batch["visible_masks"] = {
            name: mask.to(device)
            for name, mask in batch["visible_masks"].items()
        }
        optimizer.zero_grad()
        if stage == "base":
            loss, stats = compute_base_objective(
                model,
                batch,
                epoch,
                lambda_seg=lambda_seg,
                lambda_lpips=lambda_lpips,
                background_weight=background_weight,
            )
        elif stage in {"incremental_min", "incremental_suppress"}:
            loss, stats = compute_incremental_minimal_objective(
                model,
                batch,
                new_slot,
                epoch,
                lambda_rec=lambda_rec,
                lambda_sup=(lambda_sup if stage == "incremental_suppress" else 0.0),
                old_slot_names=old_slot_names,
                old_confidence_threshold=old_confidence_threshold,
            )
        else:
            raise ValueError(f"unsupported training stage: {stage}")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        scalar_stats = {
            name: float(value.detach())
            for name, value in stats.items()
            if torch.is_tensor(value) and value.ndim == 0
        }
        for name, value in scalar_stats.items():
            totals[name] = totals.get(name, 0.0) + value
        processed_steps += 1
    count = max(processed_steps, 1)
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def validate_visible_heads(model, data_loader, device, threshold=0.5):
    """Validate only masks explicitly exposed by the selected stage loader."""
    model.eval()
    dice_values = {}
    reconstruction_losses = []
    for batch in data_loader:
        image = batch["image"].to(device)
        visible_masks = {
            name: mask.to(device)
            for name, mask in batch["visible_masks"].items()
        }
        output = model(image)
        reconstruction_losses.append(float(
            base_reconstruction_loss(output["reconstruction"], image)
        ))
        for name, target in visible_masks.items():
            probability = torch.sigmoid(output["calibrated_logits"][name])
            value = binary_dice(
                probability >= threshold,
                target.bool(),
                empty_policy="nan",
            )
            if torch.isfinite(value):
                dice_values.setdefault(name, []).append(float(value))
    per_slot = {
        name: sum(values) / len(values)
        for name, values in dice_values.items()
    }
    return {
        "reconstruction_mse": (
            sum(reconstruction_losses) / len(reconstruction_losses)
            if reconstruction_losses else float("nan")
        ),
        "per_slot_dice": per_slot,
        "mean_visible_dice": (
            sum(per_slot.values()) / len(per_slot) if per_slot else float("nan")
        ),
    }
