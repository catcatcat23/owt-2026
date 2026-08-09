"""Deterministic anchor-and-context queries for PSEM-v3."""

import torch


def triplet_query_schedule(
    sample_indices: torch.Tensor,
    epoch: int,
    num_classes_with_bg: int,
):
    """Build Direct, Context, and Context-plus-anchor class masks.

    The anchor is always a foreground class and is selected without looking at
    GT labels. Half of sample/epoch pairs use the full complement as context so
    they exactly match Whole/Without inference. The other half use a
    deterministic non-empty subset of the full class bank, excluding the
    anchor, to retain compositional coverage.
    """
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must be a 1D tensor")
    if num_classes_with_bg < 2:
        raise ValueError("at least background and one foreground class are required")
    if num_classes_with_bg > 30:
        raise ValueError("binary context codes support at most 30 classes")
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")

    device = sample_indices.device
    sample_indices = sample_indices.to(dtype=torch.long)
    batch_size = sample_indices.shape[0]
    foreground_count = num_classes_with_bg - 1

    mixed = (sample_indices * 1103515245 + 12345).bitwise_and(0x7FFFFFFF)
    anchor_classes = 1 + torch.remainder(
        mixed + int(epoch), foreground_count
    )

    direct_mask = torch.zeros(
        batch_size,
        num_classes_with_bg,
        dtype=torch.bool,
        device=device,
    )
    rows = torch.arange(batch_size, device=device)
    direct_mask[rows, anchor_classes] = True

    available = torch.ones_like(direct_mask)
    available[rows, anchor_classes] = False
    full_context = torch.remainder(sample_indices + int(epoch), 2) == 0

    # Encode a non-empty subset over the K-1 non-anchor classes. Ranks make
    # the bit positions contiguous even though the anchor column is skipped.
    max_context_code = (1 << (num_classes_with_bg - 1)) - 1
    context_codes = torch.remainder(
        mixed + int(epoch) * 104729, max_context_code
    ) + 1
    available_ranks = available.long().cumsum(dim=1) - 1
    subset_bits = torch.bitwise_right_shift(
        context_codes.unsqueeze(1), available_ranks.clamp_min(0)
    ).bitwise_and(1).bool()
    random_context = available & subset_bits

    context_mask = torch.where(
        full_context.unsqueeze(1), available, random_context
    )
    if torch.any(context_mask.sum(dim=1) == 0):
        raise RuntimeError("triplet schedule produced an empty context")

    plus_mask = context_mask | direct_mask
    return {
        "anchor_classes": anchor_classes,
        "direct_mask": direct_mask,
        "context_mask": context_mask,
        "plus_mask": plus_mask,
        "full_context": full_context,
        "context_codes": context_codes,
    }
