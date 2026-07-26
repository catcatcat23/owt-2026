"""Per-sample slot-aware TGR schedules and targets."""

import torch


def _validate_indices(sample_indices, batch_size):
    if sample_indices.ndim != 1 or sample_indices.shape[0] != batch_size:
        raise ValueError("sample_indices must be [B]")
    return sample_indices.long()


def sample_base_slot_keep_mask(
    sample_indices,
    epoch,
    slot_count,
    seed=0,
):
    """Keep a deterministic random-looking non-empty subset per sample."""
    if slot_count < 1:
        raise ValueError("slot_count must be positive")
    sample_indices = _validate_indices(sample_indices, sample_indices.shape[0])
    device = sample_indices.device
    base = (
        sample_indices * 1103515245
        + int(epoch) * 12345
        + int(seed) * 2654435761
    ).bitwise_and(0x7FFFFFFF)
    retained_count = torch.remainder(base, slot_count) + 1
    positions = torch.arange(slot_count, device=device, dtype=torch.long)
    scores = (
        base[:, None]
        + (positions[None] + 1) * 1664525
        + (sample_indices[:, None] + 1) * (positions[None] + 3) * 1013904223
    ).bitwise_and(0x7FFFFFFF)
    permutation = scores.argsort(dim=1)
    ranks = torch.empty_like(permutation)
    ranks.scatter_(
        1,
        permutation,
        positions.unsqueeze(0).expand_as(permutation),
    )
    keep = ranks < retained_count[:, None]
    if torch.any(keep.sum(dim=1) == 0):
        raise RuntimeError("base TGR produced an all-dropped sample")
    return keep


def sample_retain_new_keep_mask(
    sample_indices,
    epoch,
    slot_names,
    new_slot,
    seed=0,
    minimal=False,
):
    if new_slot not in slot_names:
        raise ValueError("new_slot is not present")
    batch_size = sample_indices.shape[0]
    if minimal:
        return torch.ones(
            batch_size,
            len(slot_names),
            dtype=torch.bool,
            device=sample_indices.device,
        )
    existing = [name for name in slot_names if name != new_slot]
    existing_keep = sample_base_slot_keep_mask(
        sample_indices,
        epoch,
        len(existing),
        seed=seed,
    )
    keep = torch.zeros(
        batch_size,
        len(slot_names),
        dtype=torch.bool,
        device=sample_indices.device,
    )
    keep[:, slot_names.index(new_slot)] = True
    for index, name in enumerate(existing):
        keep[:, slot_names.index(name)] = existing_keep[:, index]
    return keep


def _pixel_mask(mask, image):
    mask = mask.to(device=image.device, dtype=torch.bool)
    if mask.ndim != image.ndim:
        raise ValueError("visible masks and image must have equal rank")
    if mask.shape[0] != image.shape[0] or mask.shape[2:] != image.shape[2:]:
        raise ValueError("visible mask spatial shape does not match image")
    if mask.shape[1] != 1:
        raise ValueError("visible masks must have one channel")
    return mask.expand(-1, image.shape[1], *([-1] * (image.ndim - 2)))


def build_base_reconstruction_target(
    image,
    visible_masks,
    slot_names,
    slot_keep_mask,
):
    if slot_keep_mask.shape != (image.shape[0], len(slot_names)):
        raise ValueError("slot_keep_mask has incompatible shape")
    target = image.clone()
    for slot_index, name in enumerate(slot_names):
        if name not in visible_masks:
            raise KeyError(f"missing visible mask for slot {name}")
        dropped = ~slot_keep_mask[:, slot_index]
        if not torch.any(dropped):
            continue
        region = _pixel_mask(visible_masks[name], image)
        region = region & dropped.reshape(
            image.shape[0], 1, *([1] * (image.ndim - 2))
        )
        target = target.masked_fill(region, 0)
    return target


def build_incremental_pseudo_target(
    image,
    new_mask,
    frozen_slot_logits,
    slot_names,
    slot_keep_mask,
    threshold=0.7,
):
    """Build formal targets without accepting any old ground-truth argument."""
    target = image.clone()
    new_region = _pixel_mask(new_mask, image)
    for slot_index, name in enumerate(slot_names):
        dropped = ~slot_keep_mask[:, slot_index]
        if not torch.any(dropped) or name not in frozen_slot_logits:
            continue
        with torch.no_grad():
            pseudo = torch.sigmoid(frozen_slot_logits[name].detach()) > threshold
            pseudo = pseudo & ~new_mask.bool()
        region = _pixel_mask(pseudo, image)
        region = region & dropped.reshape(
            image.shape[0], 1, *([1] * (image.ndim - 2))
        )
        target = target.masked_fill(region, 0)
    # The current new organ is always retained and never zeroed.
    return torch.where(new_region, image, target)
