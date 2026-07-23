import torch


def label_map_from_channels(labels):
    """Return one integer label map from OWT's repeated label channels."""
    if labels.ndim not in (4, 5):
        raise ValueError(f"expected 2D or 3D batched labels, got {labels.shape}")
    return labels[:, 0].long()


def present_class_mask(labels, num_classes_with_bg):
    """Build [B, K] presence flags from the current transformed labels."""
    label_map = label_map_from_channels(labels)
    if label_map.numel() == 0:
        raise ValueError("labels must not be empty")
    if label_map.min() < 0 or label_map.max() >= num_classes_with_bg:
        raise ValueError(
            f"label ids must be in [0, {num_classes_with_bg - 1}], "
            f"got [{label_map.min().item()}, {label_map.max().item()}]"
        )

    present = torch.zeros(
        label_map.shape[0], num_classes_with_bg,
        dtype=torch.bool, device=label_map.device
    )
    return present.scatter_(1, label_map.flatten(1), True)


def exhaustive_present_label_schedule(present_mask, sample_indices, epoch):
    """Enumerate valid deletion subsets independently for every sample.

    For m present classes, codes 0 .. 2**m - 2 enumerate every deletion subset
    except deleting all classes. Code 0 means no class is deleted.
    """
    if present_mask.ndim != 2 or present_mask.dtype != torch.bool:
        raise ValueError("present_mask must be a boolean [B, K] tensor")
    if sample_indices.ndim != 1 or sample_indices.shape[0] != present_mask.shape[0]:
        raise ValueError("sample_indices must be [B]")

    present_counts = present_mask.sum(dim=1)
    if torch.any(present_counts == 0):
        raise ValueError("every sample must contain at least one class")

    cycle_lengths = torch.bitwise_left_shift(
        torch.ones_like(present_counts), present_counts
    ) - 1
    sample_indices = sample_indices.to(
        device=present_mask.device, dtype=torch.long
    )
    mixed_indices = (
        sample_indices * 1103515245 + 12345
    ).bitwise_and(0x7FFFFFFF)
    offsets = torch.remainder(mixed_indices, cycle_lengths)
    combination_codes = torch.remainder(
        offsets + int(epoch), cycle_lengths
    )

    present_ranks = present_mask.long().cumsum(dim=1) - 1
    bit_positions = present_ranks.clamp_min(0)
    drop_bits = torch.bitwise_right_shift(
        combination_codes.unsqueeze(1), bit_positions
    ).bitwise_and(1).bool()
    drop_mask = present_mask & drop_bits
    keep_mask = present_mask & ~drop_mask

    if torch.any(keep_mask.sum(dim=1) == 0):
        raise RuntimeError("PSEM attempted to delete every present class")
    return keep_mask, drop_mask, combination_codes, cycle_lengths


def masked_reconstruction_target(images, labels, class_keep_mask):
    """Zero pixels whose class token group is not retained for that sample."""
    label_map = label_map_from_channels(labels)
    flat_labels = label_map.flatten(1)
    pixel_keep = torch.gather(
        class_keep_mask, dim=1, index=flat_labels
    ).reshape_as(label_map)
    return images * pixel_keep.unsqueeze(1).to(images.dtype)
