"""Explicit checkpoint, migration, and freeze-audit utilities."""

import hashlib
import json
from pathlib import Path

import torch

from OrganSlotEmbed import sanitize_slot_name


def _tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def hash_frozen_parameters(model):
    return {
        name: _tensor_hash(parameter)
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }


def compare_parameter_hashes(before, after):
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    changed = sorted(
        name for name in set(before) & set(after)
        if before[name] != after[name]
    )
    return {
        "match": not missing and not added and not changed,
        "missing": missing,
        "added": added,
        "changed": changed,
    }


def save_orgslot_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    extra=None,
):
    payload = {
        "model": model.state_dict(),
        "slot_metadata": model.slot_bank.metadata(),
        "slot_names": model.slot_names,
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return payload


def _save_report(report, report_path):
    if report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
    return report


def load_orgslot_base_checkpoint(
    model,
    checkpoint_or_path,
    optimizer=None,
    report_path=None,
):
    checkpoint = (
        torch.load(checkpoint_or_path, map_location="cpu")
        if isinstance(checkpoint_or_path, (str, Path))
        else checkpoint_or_path
    )
    expected_slots = tuple(checkpoint.get("slot_names", ()))
    if expected_slots and expected_slots != model.slot_names:
        raise ValueError(
            f"checkpoint slots {expected_slots} != model slots {model.slot_names}"
        )
    result = model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    report = {
        "loaded_keys": sorted(checkpoint["model"]),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "shape_mismatches": [],
        "slots_loaded": list(model.slot_names),
        "slots_intentionally_skipped": [],
    }
    _save_report(report, report_path)
    return checkpoint, report


def append_and_initialize_new_slots(
    model,
    slots,
    init_from="background",
):
    added = []
    for name, raw_class_id in slots:
        model.append_slot(name, raw_class_id, init_from=init_from)
        added.append(name)
    return added


def visible_slot_transfer(
    target_model,
    source_checkpoint,
    visible_slots,
    report_path=None,
):
    source_state = source_checkpoint["model"]
    target_state = target_model.state_dict()
    visible_keys = {sanitize_slot_name(name) for name in visible_slots}
    selected = {}
    skipped_slots = set()
    for key, tensor in source_state.items():
        if key.startswith("slot_bank.slots."):
            slot_key = key.split(".")[2]
            if slot_key not in visible_keys:
                skipped_slots.add(slot_key)
                continue
        if key in target_state and target_state[key].shape == tensor.shape:
            selected[key] = tensor
    result = target_model.load_state_dict(selected, strict=False)
    report = {
        "loaded_keys": sorted(selected),
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": sorted(result.unexpected_keys),
        "shape_mismatches": sorted(
            key for key, tensor in source_state.items()
            if key in target_state and target_state[key].shape != tensor.shape
        ),
        "slots_loaded": sorted(visible_slots),
        "slots_intentionally_skipped": sorted(skipped_slots),
    }
    _save_report(report, report_path)
    return report


def load_original_owt_initialization(
    model,
    original_checkpoint_or_state,
    slot_order,
    report_path=None,
):
    """Migrate a joint OWT state into visible OrganSlotBank slots."""
    source = original_checkpoint_or_state
    if isinstance(source, (str, Path)):
        source = torch.load(source, map_location="cpu")
    if "model" in source:
        source = source["model"]
    target = model.state_dict()
    selected = {}
    loaded_source_keys = set()
    shape_mismatches = []

    direct_prefixes = (
        "patch_embed.",
        "cls_token",
        "pos_embed",
        "pos_embed_spatial",
        "pos_embed_temporal",
        "decoder_pos_embed",
        "decoder_blocks.",
        "decoder_norm.",
        "decoder_pred.",
    )
    for key, tensor in source.items():
        target_key = key
        if key.startswith("norm."):
            target_key = "encoder_norm." + key[len("norm."):]
        if key.startswith("blocks1.") or key.startswith(direct_prefixes):
            if target_key in target:
                if target[target_key].shape == tensor.shape:
                    selected[target_key] = tensor
                    loaded_source_keys.add(key)
                else:
                    shape_mismatches.append(target_key)

    token_factor = model.model_args.token_factor
    for slot_index, slot_name in enumerate(slot_order):
        if slot_name not in model.slot_names:
            continue
        slot_key = sanitize_slot_name(slot_name)
        prefix = f"slot_bank.slots.{slot_key}."
        for suffix in ("conv1.weight", "conv3.weight"):
            source_key = f"organ_embed.{suffix}"
            target_key = prefix + f"collector.{suffix}"
            if source_key in source and target_key in target:
                selected[target_key] = source[source_key]
                loaded_source_keys.add(source_key)
        source_key = "organ_embed.conv2.weight"
        target_key = prefix + "collector.conv2.weight"
        if source_key in source and target_key in target:
            start = slot_index * token_factor
            stop = start + token_factor
            sliced = source[source_key][start:stop]
            if sliced.shape == target[target_key].shape:
                selected[target_key] = sliced
                loaded_source_keys.add(source_key)
            else:
                shape_mismatches.append(target_key)

        for block_index in range(model.slot_tg_depth):
            source_prefix = f"blocks2.{block_index}."
            target_prefix = prefix + f"tg_encoder.{block_index}."
            for key, tensor in source.items():
                if key.startswith(source_prefix):
                    target_key = target_prefix + key[len(source_prefix):]
                    if target_key in target and target[target_key].shape == tensor.shape:
                        selected[target_key] = tensor
                        loaded_source_keys.add(key)
                    elif target_key in target:
                        shape_mismatches.append(target_key)
        for key, tensor in source.items():
            if key.startswith("decoder_embed."):
                target_key = prefix + "aher." + key[len("decoder_embed."):]
                if target_key in target and target[target_key].shape == tensor.shape:
                    selected[target_key] = tensor
                    loaded_source_keys.add(key)
                elif target_key in target:
                    shape_mismatches.append(target_key)
        for suffix in ("weight", "bias"):
            source_key = f"norm.{suffix}"
            target_key = prefix + f"token_norm.{suffix}"
            if source_key in source and target_key in target:
                selected[target_key] = source[source_key]
                loaded_source_keys.add(source_key)

    result = model.load_state_dict(selected, strict=False)
    unexpected_source = sorted(
        key for key in source
        if key not in loaded_source_keys
        and not key.startswith(("organ_embed.", "blocks2.", "decoder_embed."))
    )
    report = {
        "loaded_keys": sorted(selected),
        "loaded_source_keys": sorted(loaded_source_keys),
        "missing_keys": sorted(result.missing_keys),
        "unexpected_keys": unexpected_source,
        "shape_mismatches": sorted(set(shape_mismatches)),
        "slots_loaded": [name for name in slot_order if name in model.slot_names],
        "slots_intentionally_skipped": [
            name for name in slot_order if name not in model.slot_names
        ],
    }
    _save_report(report, report_path)
    return report
