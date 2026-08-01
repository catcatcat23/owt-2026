import torch


QUERY_CYCLE_LENGTH = 20

MODE_DIRECT_POSITIVE = 0
MODE_DIRECT_NEGATIVE = 1
MODE_LEAVE_ONE_OUT = 2
MODE_WHOLE = 3
MODE_BACKGROUND_ONLY = 4
MODE_ORGANS_ONLY = 5
MODE_RANDOM_SUBSET = 6

QUERY_MODE_NAMES = {
    MODE_DIRECT_POSITIVE: "direct_positive",
    MODE_DIRECT_NEGATIVE: "direct_negative",
    MODE_LEAVE_ONE_OUT: "leave_one_out",
    MODE_WHOLE: "whole",
    MODE_BACKGROUND_ONLY: "background_only",
    MODE_ORGANS_ONLY: "organs_only",
    MODE_RANDOM_SUBSET: "random_subset",
}


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


def query_matched_schedule(present_mask, sample_indices, epoch):
    """Build deterministic per-sample masks that match inference queries.

    The 20-slot cycle allocates 20% direct-positive, 20% direct-negative,
    25% leave-one-out, 10% whole, 5% background-only, 5% organs-only,
    and 15% full-bank random-subset queries. GT presence is used only to
    choose positive versus negative direct queries; it never limits which
    token groups are allowed to enter the model.
    """
    if present_mask.ndim != 2 or present_mask.dtype != torch.bool:
        raise ValueError("present_mask must be a boolean [B, K] tensor")
    if sample_indices.ndim != 1 or sample_indices.shape[0] != present_mask.shape[0]:
        raise ValueError("sample_indices must be [B]")
    if present_mask.shape[1] < 2:
        raise ValueError("query schedule requires background and a foreground class")
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")

    batch_size, class_count = present_mask.shape
    if class_count > 30:
        raise ValueError("random-subset bit codes support at most 30 classes")

    sample_indices = sample_indices.to(
        device=present_mask.device, dtype=torch.long
    )
    # 17 is coprime with 20, so every 20 consecutive sample indices map to
    # every slot exactly once. The +13 only rotates the starting point.
    offsets = torch.remainder(sample_indices * 17 + 13, QUERY_CYCLE_LENGTH)
    epoch_positions = offsets + int(epoch)
    slots = torch.remainder(epoch_positions, QUERY_CYCLE_LENGTH)
    cycle_numbers = torch.div(
        epoch_positions, QUERY_CYCLE_LENGTH, rounding_mode="floor"
    )
    # When a slot repeats after 20 epochs, this selector advances by 23,
    # which is coprime with all current foreground class counts (4 and 8).
    selectors = sample_indices * 17 + int(epoch) + cycle_numbers * 3

    keep_mask = torch.zeros_like(present_mask)
    mode_ids = torch.empty(batch_size, dtype=torch.long, device=present_mask.device)
    query_classes = torch.full(
        (batch_size,), -1, dtype=torch.long, device=present_mask.device
    )
    foreground_present = present_mask[:, 1:]
    foreground_absent = ~foreground_present
    present_counts = foreground_present.sum(dim=1)
    absent_counts = foreground_absent.sum(dim=1)

    nominal_positive = slots <= 3
    nominal_negative = (slots >= 4) & (slots <= 7)
    direct_query = nominal_positive | nominal_negative
    use_present = (
        (nominal_positive & (present_counts > 0))
        | (nominal_negative & (absent_counts == 0) & (present_counts > 0))
    )
    use_absent = (
        (nominal_negative & (absent_counts > 0))
        | (nominal_positive & (present_counts == 0) & (absent_counts > 0))
    )
    background_fallback = direct_query & ~use_present & ~use_absent

    direct_candidates = torch.where(
        use_present.unsqueeze(1), foreground_present, foreground_absent
    )
    direct_counts = direct_candidates.sum(dim=1).clamp_min(1)
    selected_ranks = torch.remainder(selectors, direct_counts)
    selected_candidates = direct_candidates & (
        direct_candidates.long().cumsum(dim=1)
        == selected_ranks.unsqueeze(1) + 1
    )
    selected_classes = selected_candidates.long().argmax(dim=1) + 1
    selected_direct = use_present | use_absent
    selected_rows = torch.nonzero(selected_direct, as_tuple=False).squeeze(1)
    keep_mask[selected_rows, selected_classes[selected_rows]] = True
    query_classes[selected_direct] = selected_classes[selected_direct]
    keep_mask[background_fallback, 0] = True
    mode_ids[use_present] = MODE_DIRECT_POSITIVE
    mode_ids[use_absent] = MODE_DIRECT_NEGATIVE
    mode_ids[background_fallback] = MODE_BACKGROUND_ONLY

    leave_one_out = (slots >= 8) & (slots <= 12)
    leave_classes = 1 + torch.remainder(selectors, class_count - 1)
    keep_mask[leave_one_out] = True
    leave_rows = torch.nonzero(leave_one_out, as_tuple=False).squeeze(1)
    keep_mask[leave_rows, leave_classes[leave_rows]] = False
    query_classes[leave_one_out] = leave_classes[leave_one_out]
    mode_ids[leave_one_out] = MODE_LEAVE_ONE_OUT

    whole = (slots >= 13) & (slots <= 14)
    keep_mask[whole] = True
    mode_ids[whole] = MODE_WHOLE

    background_only = slots == 15
    keep_mask[background_only, 0] = True
    mode_ids[background_only] = MODE_BACKGROUND_ONLY

    organs_only = slots == 16
    keep_mask[organs_only, 1:] = True
    mode_ids[organs_only] = MODE_ORGANS_ONLY

    random_subset = slots >= 17
    max_code = (1 << class_count) - 1
    random_codes = torch.remainder(
        selectors + slots * 13, max_code
    ) + 1
    bit_positions = torch.arange(class_count, device=present_mask.device)
    random_keep = torch.bitwise_right_shift(
        random_codes.unsqueeze(1), bit_positions.unsqueeze(0)
    ).bitwise_and(1).bool()
    keep_mask[random_subset] = random_keep[random_subset]
    mode_ids[random_subset] = MODE_RANDOM_SUBSET

    if torch.any(keep_mask.sum(dim=1) == 0):
        raise RuntimeError("PSEM-v2 attempted to build an empty query")

    drop_mask = ~keep_mask
    return keep_mask, drop_mask, slots, mode_ids, query_classes


def masked_reconstruction_target(images, labels, class_keep_mask):
    """Zero pixels whose class token group is not retained for that sample."""
    label_map = label_map_from_channels(labels)
    flat_labels = label_map.flatten(1)
    pixel_keep = torch.gather(
        class_keep_mask, dim=1, index=flat_labels
    ).reshape_as(label_map)
    return images * pixel_keep.unsqueeze(1).to(images.dtype)
