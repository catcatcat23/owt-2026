"""Strict partial loading from architecture-matched MAE to OrganSlot."""

import hashlib
from pathlib import Path

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_mapping(source_state, scope):
    if scope not in ("encoder", "encoder_decoder"):
        raise ValueError("scope must be encoder or encoder_decoder")
    mapping = {
        "cls_token": "cls_token",
    }
    for source_key in source_state:
        if source_key.startswith("patch_embed."):
            mapping[source_key] = source_key
        elif source_key.startswith("blocks."):
            mapping[source_key] = "blocks1." + source_key[len("blocks.") :]
        elif source_key.startswith("encoder_norm."):
            mapping[source_key] = source_key
        elif scope == "encoder_decoder" and source_key.startswith(
            ("decoder_blocks.", "decoder_norm.", "decoder_pred.")
        ):
            mapping[source_key] = source_key
    return mapping


def load_mae_transfer_checkpoint(model, checkpoint_path, scope):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if "model" not in checkpoint:
        raise ValueError("MAE checkpoint has no model state")
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    mapping = _build_mapping(source_state, scope)
    loaded = []
    shape_errors = []
    missing_targets = []
    with torch.no_grad():
        for source_key, target_key in sorted(mapping.items()):
            if target_key not in target_state:
                missing_targets.append((source_key, target_key))
                continue
            source_value = source_state[source_key]
            target_value = target_state[target_key]
            if tuple(source_value.shape) != tuple(target_value.shape):
                shape_errors.append(
                    {
                        "source": source_key,
                        "target": target_key,
                        "source_shape": list(source_value.shape),
                        "target_shape": list(target_value.shape),
                    }
                )
                continue
            target_value.copy_(source_value)
            loaded.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "shape": list(source_value.shape),
                }
            )
    if missing_targets or shape_errors:
        raise RuntimeError(
            "invalid MAE transfer: missing_targets={}, shape_errors={}".format(
                missing_targets, shape_errors
            )
        )

    required_prefixes = ["patch_embed.", "blocks1.", "encoder_norm."]
    if scope == "encoder_decoder":
        required_prefixes.extend(
            ["decoder_blocks.", "decoder_norm.", "decoder_pred."]
        )
    expected_targets = {"cls_token"}
    expected_targets.update(
        key
        for key in target_state
        if any(key.startswith(prefix) for prefix in required_prefixes)
    )
    loaded_targets = {item["target"] for item in loaded}
    uncovered_targets = sorted(expected_targets - loaded_targets)
    if uncovered_targets:
        raise RuntimeError(
            "MAE checkpoint did not cover every required target tensor: {}".format(
                uncovered_targets
            )
        )
    ignored_source = sorted(set(source_state) - set(mapping))
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "scope": scope,
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_optimizer_updates": int(
            checkpoint.get("optimizer_updates", -1)
        ),
        "source_input_size": int(checkpoint.get("input_size", -1)),
        "target_input_size": int(getattr(model, "img_size", -1)),
        "position_embedding_policy": "target regenerated; source pos embeddings ignored",
        "loaded_tensor_count": len(loaded),
        "loaded": loaded,
        "ignored_source_keys": ignored_source,
        "slot_policy": "all slot collectors/TGEnc/AHER/heads remain target initialization",
        "optimizer_policy": "not loaded",
    }
