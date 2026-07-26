"""Tiny deterministic base overfit plus strict incremental one-step audit."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from OWT_models_orgslot import OrganSlotMaskedAutoencoderViT
from engine_pretrain_orgslot import (
    compute_base_objective,
    compute_incremental_minimal_objective,
)
from eval_orgslot import binary_dice
from util.checkpoint_orgslot import (
    compare_parameter_hashes,
    hash_frozen_parameters,
    load_orgslot_base_checkpoint,
    save_orgslot_checkpoint,
)


def build_model():
    args = SimpleNamespace(
        LA=True, arch_version="v1", dataset_type="2D", token_factor=2,
        organ_token_total=4, fix_frame=4, temp_stride=1,
        loss_version="L2", text_encoding="None",
    )
    return OrganSlotMaskedAutoencoderViT(
        img_size=32, patch_size=16, embed_dim=32, depth=2, num_heads=4,
        decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
        mlp_ratio=2, norm_layer=nn.LayerNorm, model_args=args,
        slot_specs=[
            {"name": "background", "raw_class_id": 0},
            {"name": "organ", "raw_class_id": 1},
        ],
        slot_tg_depth=1,
    )


def make_base_batch():
    mask = torch.zeros(2, 1, 32, 32)
    # One repeated, patch-aligned sample is the canonical memorization test.
    # With patch_size=16 this target is exactly representable on a 2x2 canvas.
    mask[:, :, :16, :16] = 1
    image = 0.1 * torch.ones(2, 3, 32, 32)
    image = image + mask.repeat(1, 3, 1, 1) * 0.8
    return {
        "image": image,
        "visible_masks": {"background": 1 - mask, "organ": mask},
        # Odd indices yield retained_count=2 at epoch=0 for S=2.
        "sample_index": torch.tensor([1, 3]),
    }


def organ_dice(model, batch, name="organ"):
    with torch.no_grad():
        logits = model(batch["image"])["slot_logits"][name]
        return float(binary_dice(logits > 0, batch["visible_masks"][name], empty_policy="zero"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(17)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = build_model().to(device)
    batch = make_base_batch()
    batch = {
        "image": batch["image"].to(device),
        "visible_masks": {key: value.to(device) for key, value in batch["visible_masks"].items()},
        "sample_index": batch["sample_index"].to(device),
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0)
    initial_loss = None
    for step in range(args.steps):
        loss, _ = compute_base_objective(
            model, batch, epoch=0, lambda_lpips=0, background_weight=0.25
        )
        if initial_loss is None:
            initial_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss, _ = compute_base_objective(
        model, batch, epoch=0, lambda_lpips=0, background_weight=0.25
    )
    final_loss = float(final_loss.detach())
    final_dice = organ_dice(model, batch)

    checkpoint_path = output_dir / "tiny_base_checkpoint.pth"
    save_orgslot_checkpoint(checkpoint_path, model, optimizer, args.steps - 1)
    reloaded = build_model().to(device)
    _, load_report = load_orgslot_base_checkpoint(reloaded, checkpoint_path)
    for key, value in model.state_dict().items():
        if not torch.equal(value, reloaded.state_dict()[key]):
            raise AssertionError(f"checkpoint mismatch: {key}")

    reloaded.append_slot("liver", 4, init_from="background")
    reloaded.freeze_for_incremental(
        old_slots=("organ",), new_slots=("liver",), background_policy="frozen"
    )
    frozen_hashes = hash_frozen_parameters(reloaded)
    incremental_optimizer = torch.optim.AdamW(
        [parameter for parameter in reloaded.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    liver_mask = torch.zeros(2, 1, 32, 32, device=device)
    liver_mask[:, :, 20:29, 2:11] = 1
    incremental_batch = {
        "image": batch["image"],
        "visible_masks": {"liver": liver_mask},
        "sample_index": batch["sample_index"],
    }
    incremental_loss, _ = compute_incremental_minimal_objective(
        reloaded, incremental_batch, "liver", epoch=0
    )
    incremental_optimizer.zero_grad()
    incremental_loss.backward()
    for name, parameter in reloaded.named_parameters():
        if not parameter.requires_grad and parameter.grad is not None:
            raise AssertionError(f"frozen gradient exists: {name}")
    incremental_optimizer.step()
    frozen_check = compare_parameter_hashes(
        frozen_hashes, hash_frozen_parameters(reloaded)
    )
    if not frozen_check["match"]:
        raise AssertionError(f"frozen parameters changed: {frozen_check}")
    result = {
        "steps": args.steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": final_loss / initial_loss,
        "organ_dice": final_dice,
        "checkpoint_exact": not load_report["missing_keys"] and not load_report["unexpected_keys"],
        "incremental_loss": float(incremental_loss.detach()),
        "incremental_visible_keys": ["liver"],
        "frozen_parameters_unchanged": frozen_check["match"],
    }
    if not final_loss < initial_loss * 0.5:
        raise AssertionError(f"tiny overfit did not reduce loss enough: {result}")
    with open(output_dir / "tiny_overfit_results.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
