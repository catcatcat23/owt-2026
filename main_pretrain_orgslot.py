"""Stage-explicit OrganSlotBank runner for synthetic gates and real smokes."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from datasets.orgslot_manifest import OrganSlotManifestDataset
from OWT_models_orgslot import (
    OrganSlotMaskedAutoencoderViT,
    mae_vit_base_patch16,
    mae_vit_basefix16_patch16,
)
from engine_pretrain_orgslot import train_one_epoch, validate_visible_heads
from util.checkpoint_orgslot import (
    append_and_initialize_new_slots,
    compare_parameter_hashes,
    hash_frozen_parameters,
    load_original_owt_initialization,
    load_orgslot_base_checkpoint,
    save_orgslot_checkpoint,
)
from util.label_visibility import (
    StrictVisibilityDataset,
    assert_case_splits_disjoint,
    load_visibility_config,
)


class SyntheticRawDataset(Dataset):
    """Debug-only raw dataset; strict visibility is applied by a wrapper."""

    def __init__(self, length=8, image_size=32, seed=0, dimension="2D", frames=4):
        self.length = int(length)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.dimension = dimension
        self.frames = int(frames)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        size = self.image_size
        label_2d = torch.zeros(size, size)
        half = size // 2
        label_2d[2:half - 1, 2:half - 1] = 1
        label_2d[2:half - 1, half + 1:size - 2] = 2
        label_2d[half + 1:size - 2, 2:half - 1] = 3
        label_2d[half + 1:size - 2, half + 1:size - 2] = 4
        if self.dimension == "3D":
            label = label_2d[None].repeat(3, self.frames, 1, 1)
        else:
            label = label_2d[None].repeat(3, 1, 1)
        generator = torch.Generator().manual_seed(self.seed + index)
        noise = 0.01 * torch.rand(label.shape, generator=generator)
        image = label[:1].div(4).repeat(3, *([1] * (label.ndim - 1))) + noise
        return {
            "image": image.clamp(0, 1),
            "label": label,
            "sample_index": index,
            "case_name": f"synthetic_{index:03d}",
            "slice_index": index,
        }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _model_args(args, slot_count):
    return SimpleNamespace(
        LA=True,
        arch_version="v1",
        dataset_type=args.dimension,
        token_factor=args.token_factor,
        organ_token_total=args.token_factor * slot_count,
        fix_frame=args.fix_frame,
        temp_stride=1,
        loss_version=args.loss_version,
        text_encoding="None",
    )


def build_model(args, slot_specs):
    model_args = _model_args(args, len(slot_specs))
    common = {
        "img_size": args.image_size,
        "model_args": model_args,
        "slot_specs": slot_specs,
        "slot_tg_depth": args.slot_tg_depth,
    }
    if args.model_size == "base":
        factory = (
            mae_vit_basefix16_patch16
            if args.dimension == "3D"
            else mae_vit_base_patch16
        )
        return factory(**common)
    return OrganSlotMaskedAutoencoderViT(
        patch_size=16,
        embed_dim=args.embed_dim,
        depth=2,
        num_heads=4,
        decoder_embed_dim=args.embed_dim,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
        **common,
    )


def parse_args():
    parser = argparse.ArgumentParser("OrganSlotBank stage-aware trainer")
    parser.add_argument(
        "--stage",
        choices=("base", "incremental_min", "incremental_suppress"),
        default="base",
    )
    parser.add_argument(
        "--method", choices=("ours", "sequential_ft", "head_only"), default="ours"
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data-path", help="training CSV for real-data mode")
    parser.add_argument("--val-data-path", help="case-disjoint validation CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--visibility-config", default="configs/orgslot/owt_legacy_debug.json"
    )
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--owt-checkpoint")
    parser.add_argument("--new-slot", default="liver")
    parser.add_argument("--dimension", choices=("2D", "3D"), default="2D")
    parser.add_argument("--model-size", choices=("debug", "base"), default="debug")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--fix-frame", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--token-factor", type=int)
    parser.add_argument("--slot-tg-depth", type=int, default=1)
    parser.add_argument("--loss-version", default="L2")
    parser.add_argument("--intensity-norm", choices=("per_sample", "fixed_255"), default="per_sample")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lambda-seg", type=float, default=1.0)
    parser.add_argument("--lambda-lpips", type=float, default=0.0)
    parser.add_argument("--lambda-rec", type=float, default=0.1)
    parser.add_argument("--lambda-sup", type=float, default=1.0)
    parser.add_argument("--old-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--lambda-bg-seg", type=float, default=0.25)
    parser.add_argument("--background-policy", choices=("frozen", "plastic"), default="frozen")
    parser.add_argument("--train-calibration", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.image_size is None:
        args.image_size = 224 if args.model_size == "base" else 32
    if args.token_factor is None:
        args.token_factor = 20 if args.model_size == "base" else 2
    return args


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_raw_dataset(args, csv_path, max_samples):
    if args.synthetic:
        length = max_samples if max_samples is not None else 8
        return SyntheticRawDataset(
            length=length,
            image_size=args.image_size,
            seed=args.seed,
            dimension=args.dimension,
            frames=args.fix_frame,
        )
    return OrganSlotManifestDataset(
        csv_path,
        dataset_type=args.dimension,
        fix_frame=args.fix_frame,
        intensity_norm=args.intensity_norm,
        expected_size=args.image_size,
        max_samples=max_samples,
    )


def _configure_incremental_method(model, args, old_slot_names):
    if args.method == "ours":
        return model.freeze_for_incremental(
            old_slots=old_slot_names,
            new_slots=(args.new_slot,),
            background_policy=args.background_policy,
            train_calibration=args.train_calibration,
        )
    if args.method == "head_only":
        return model.freeze_for_head_only(
            args.new_slot, train_calibration=args.train_calibration
        )
    return model.unfreeze_all()


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_git_provenance(output_dir):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], check=True, capture_output=True, text=True
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary"], check=True, capture_output=True, text=True
    ).stdout
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(
        status if status else "clean\n", encoding="utf-8"
    )
    (output_dir / "git_diff.patch").write_text(diff, encoding="utf-8")


def main():
    args = parse_args()
    if not args.synthetic and (not args.data_path or not args.val_data_path):
        raise ValueError("real-data mode requires separate --data-path and --val-data-path")
    if args.stage != "base" and not args.base_checkpoint:
        raise ValueError("incremental stages require --base-checkpoint")
    if args.stage == "base" and args.method != "ours":
        raise ValueError("--method applies only to incremental stages")
    if args.stage == "incremental_suppress" and args.method != "ours":
        raise ValueError("incremental_suppress requires frozen old slots (--method ours)")
    if args.stage != "incremental_suppress" and args.lambda_sup != 1.0:
        raise ValueError("--lambda-sup is used only by incremental_suppress")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "resolved_config.json", vars(args))
    _write_git_provenance(output_dir)
    shutil.copyfile(args.visibility_config, output_dir / "class_map.json")
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]), encoding="utf-8"
    )
    if not args.synthetic:
        _write_json(
            output_dir / "manifest_checksums.json",
            {
                "train": _sha256(args.data_path),
                "validation": _sha256(args.val_data_path),
            },
        )

    classes, stages = load_visibility_config(args.visibility_config)
    base_stage = stages["base"]
    class_by_name = {spec.name: spec for spec in classes}
    base_specs = [
        {"name": name, "raw_class_id": class_by_name[name].raw_id}
        for name in base_stage.visible_slots
    ]
    model = build_model(args, base_specs)
    load_report = None
    if args.stage == "base":
        stage_spec = base_stage
        if args.owt_checkpoint:
            load_report = load_original_owt_initialization(
                model,
                args.owt_checkpoint,
                slot_order=base_stage.visible_slots,
                report_path=output_dir / "checkpoint_load_report.json",
            )
    else:
        _, load_report = load_orgslot_base_checkpoint(
            model,
            args.base_checkpoint,
            report_path=output_dir / "checkpoint_load_report.json",
        )
        append_and_initialize_new_slots(
            model, [(args.new_slot, class_by_name[args.new_slot].raw_id)]
        )
        stage_spec = stages["incremental_liver"]
        old_slot_names = tuple(
            name for name in base_stage.visible_slots if name != "background"
        )
        _configure_incremental_method(model, args, old_slot_names)

    train_raw = _build_raw_dataset(args, args.data_path, args.max_train_samples)
    val_raw = _build_raw_dataset(args, args.val_data_path, args.max_val_samples)
    if not args.synthetic:
        assert_case_splits_disjoint(
            {"train": train_raw.case_ids, "validation": val_raw.case_ids}
        )
        _write_json(
            output_dir / "case_split_report.json",
            {
                "train_cases": list(train_raw.case_ids),
                "validation_cases": list(val_raw.case_ids),
            },
        )
    train_dataset = StrictVisibilityDataset(train_raw, classes, stage_spec)
    val_dataset = StrictVisibilityDataset(val_raw, classes, stage_spec)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        generator=generator,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        worker_init_fn=seed_worker,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("selected method has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    _write_json(output_dir / "parameter_report.json", model.parameter_report())

    frozen_before = hash_frozen_parameters(model)
    freeze_audit = {
        "initial_frozen_parameter_count": len(frozen_before),
        "epochs": [],
    }
    metrics_path = output_dir / "metrics.jsonl"
    best_metric = None
    old_slot_names = tuple(
        name for name in base_stage.visible_slots if name != "background"
    )
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            stage=args.stage,
            new_slot=args.new_slot,
            lambda_seg=args.lambda_seg,
            lambda_lpips=args.lambda_lpips,
            lambda_rec=args.lambda_rec,
            lambda_sup=args.lambda_sup,
            background_weight=args.lambda_bg_seg,
            old_slot_names=old_slot_names,
            old_confidence_threshold=args.old_confidence_threshold,
            max_steps=args.max_steps,
        )
        frozen_after = hash_frozen_parameters(model)
        comparison = compare_parameter_hashes(frozen_before, frozen_after)
        freeze_audit["epochs"].append({"epoch": epoch, **comparison})
        if not comparison["match"]:
            raise RuntimeError(f"frozen parameter drift at epoch {epoch}: {comparison}")
        validation = validate_visible_heads(model, val_loader, device)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation}
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True))
        metric = validation["mean_visible_dice"]
        if best_metric is None or (np.isfinite(metric) and metric > best_metric):
            best_metric = metric
            save_orgslot_checkpoint(
                output_dir / "best_checkpoint.pth",
                model,
                optimizer,
                epoch,
                extra={"stage": args.stage, "method": args.method, "best_metric": metric},
            )

    _write_json(output_dir / "freeze_audit.json", freeze_audit)
    save_orgslot_checkpoint(
        output_dir / "last_checkpoint.pth",
        model,
        optimizer,
        args.epochs - 1,
        extra={
            "synthetic_smoke": args.synthetic,
            "stage": args.stage,
            "method": args.method,
            "checkpoint_load_report_present": load_report is not None,
        },
    )


if __name__ == "__main__":
    main()
