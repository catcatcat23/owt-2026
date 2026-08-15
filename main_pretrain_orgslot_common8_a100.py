"""DDP trainer for Common8 native-resample crop to 448 experiments."""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from datasets.orgslot_highres import (
    OrganAwareMixDataset,
    OrganSlotHighResTransform,
    TransformDataset,
)
from datasets.orgslot_manifest import OrganSlotManifestDataset
from engine_pretrain_orgslot_common8_a100 import train_one_epoch
from OWT_models_orgslot import FUSION_MODES, mae_vit_base_patch16
from VQ.lpips import LPIPS
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.label_visibility import (
    StrictVisibilityDataset,
    assert_case_splits_disjoint,
    load_visibility_config,
)
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser(
        "OrganSlotBank Common8 1x1x2/448 pre-training", add_help=False
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=798, type=int)
    parser.add_argument("--accum_iter", default=12, type=int)
    parser.add_argument("--input_size", default=448, type=int)
    parser.add_argument("--token_factor", default=20, type=int)
    parser.add_argument("--slot_tg_depth", default=1, type=int)
    parser.add_argument(
        "--fusion_mode",
        choices=FUSION_MODES,
        default="linear_sqrt",
    )
    parser.add_argument("--fusion_reference_count", type=int)
    parser.add_argument(
        "--training_scope",
        choices=("reconstruction", "head_only"),
        default="reconstruction",
    )
    parser.add_argument("--init_checkpoint", default="")

    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=40, type=int)
    parser.add_argument("--max_optimizer_updates", default=118800, type=int)
    parser.add_argument("--warmup_updates", default=5940, type=int)

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_data_path", required=True)
    parser.add_argument("--preprocess_summary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--visibility_config",
        default="configs/orgslot/common8_offline.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--organ_roi_aug", action="store_true")
    parser.add_argument("--disable_train_augmentation", action="store_true")
    parser.add_argument("--organ_roi_probability", default=0.2, type=float)
    parser.add_argument("--global_crop_size", default=448, type=int)
    parser.add_argument("--roi_crop_size", default=384, type=int)
    parser.add_argument("--roi_center_jitter", default=0.1, type=float)
    parser.add_argument("--focus_class_ids", default="4,5,6")
    parser.add_argument("--roi_index", default=None)

    parser.add_argument("--loss_version", default="L2-LPIPS")
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_seg", default=0.0, type=float)
    parser.add_argument("--lambda_bg_seg", default=0.25, type=float)
    parser.add_argument(
        "--lpips_state",
        default="/mnt/DATA-4/anteng/pretrained/owt_lpips_vgg16.pth",
    )
    parser.add_argument(
        "--tgr_mode",
        choices=("legacy_batch", "per_sample"),
        default="legacy_batch",
    )
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--max_steps_per_epoch", default=None, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_val_samples", default=None, type=int)
    parser.add_argument("--no_save", action="store_true")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_provenance(args, output_dir):
    if not misc.is_main_process():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (output_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (output_dir / "git_status.txt").write_text(
        status if status else "clean\n", encoding="utf-8"
    )
    (output_dir / "git_diff.patch").write_text(diff, encoding="utf-8")
    shutil.copyfile(args.visibility_config, output_dir / "class_map.json")
    shutil.copyfile(args.preprocess_summary, output_dir / "preprocess_summary.json")
    checksums = {
        "train": _sha256(args.data_path),
        "validation": _sha256(args.val_data_path),
        "preprocess_summary": _sha256(args.preprocess_summary),
    }
    if args.roi_index is not None:
        checksums["roi_index"] = _sha256(args.roi_index)
        shutil.copyfile(args.roi_index, output_dir / "roi_index.json")
    with open(
        output_dir / "dataset_manifest_checksums.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(checksums, handle, indent=2, sort_keys=True)


def _model_args(args, slot_count):
    return SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=args.token_factor,
        organ_token_total=args.token_factor * slot_count,
        fix_frame=0,
        temp_stride=0,
        loss_version=[name for name in args.loss_version.split("-") if name != "LPIPS"],
        text_encoding="None",
    )


def _checkpoint_value(checkpoint, name, default=None):
    if name in checkpoint:
        return checkpoint[name]
    saved_args = checkpoint.get("args")
    if saved_args is None:
        return default
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def _load_initial_checkpoint(model, path, args):
    checkpoint = torch.load(path, map_location="cpu")
    for name, requested in (
        ("input_size", args.input_size),
        ("token_factor", args.token_factor),
        ("slot_tg_depth", args.slot_tg_depth),
        ("fusion_mode", args.fusion_mode),
        ("fusion_reference_count", args.fusion_reference_count),
    ):
        saved = _checkpoint_value(checkpoint, name)
        if saved is not None and saved != requested:
            raise ValueError(
                "initial checkpoint {} {} != requested {}".format(
                    name, saved, requested
                )
            )
    full_state = checkpoint["model"]
    state = {
        key: value
        for key, value in full_state.items()
        if not key.startswith("perceptual_loss.")
    }
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("initial checkpoint load was not exact: {}".format(result))
    return {
        "path": str(Path(path).resolve()),
        "sha256": _sha256(path),
        "source_epoch": int(checkpoint.get("epoch", -1)),
        "source_tensor_count": len(full_state),
        "loaded_tensor_count": len(state),
        "stripped_lpips_tensor_count": len(full_state) - len(state),
        "exact": True,
    }


def _seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _build_dataset(args, csv_path, classes, stage, training, max_samples):
    raw = OrganSlotManifestDataset(
        csv_path,
        dataset_type="2D",
        intensity_norm="fixed_255",
        expected_size=None,
        max_samples=max_samples,
    )
    transform = OrganSlotHighResTransform(
        output_size=args.input_size,
        training=training and not args.disable_train_augmentation,
        global_crop_size=args.global_crop_size,
        roi_crop_size=args.roi_crop_size,
        roi_center_jitter=args.roi_center_jitter,
    )
    transformed = TransformDataset(raw, transform)
    if training and args.organ_roi_aug:
        if args.roi_index is None:
            raise ValueError("--organ_roi_aug requires --roi_index")
        with open(args.roi_index, "r", encoding="utf-8") as handle:
            roi_metadata = json.load(handle)
        focus_class_ids = tuple(
            int(value) for value in args.focus_class_ids.split(",") if value
        )
        transformed = OrganAwareMixDataset(
            raw,
            transform,
            roi_metadata["class_indices"],
            focus_class_ids=focus_class_ids,
            roi_probability=args.organ_roi_probability,
        )
    return raw, StrictVisibilityDataset(transformed, classes, stage)


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    if args.device != "cuda":
        raise ValueError("the formal A100 entry point requires --device cuda")
    if args.lambda_lpips and "LPIPS" not in args.loss_version.split("-"):
        raise ValueError("lambda_lpips > 0 requires LPIPS in --loss_version")
    if args.training_scope == "head_only":
        if not args.init_checkpoint:
            raise ValueError("head_only requires --init_checkpoint")
        if args.resume:
            raise ValueError("head_only init and --resume are mutually exclusive")
        if args.lambda_lpips != 0:
            raise ValueError("head_only requires --lambda_lpips 0")
        if args.lambda_seg <= 0:
            raise ValueError("head_only requires --lambda_seg > 0")
    if args.disable_train_augmentation and args.organ_roi_aug:
        raise ValueError("disabled augmentation cannot be combined with ROI augmentation")
    for path in (
        args.data_path,
        args.val_data_path,
        args.preprocess_summary,
        args.visibility_config,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if args.organ_roi_aug and (args.roi_index is None or not Path(args.roi_index).is_file()):
        raise FileNotFoundError(args.roi_index)
    if args.init_checkpoint and not Path(args.init_checkpoint).is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if args.lambda_lpips and not Path(args.lpips_state).is_file():
        raise FileNotFoundError(args.lpips_state)
    with open(args.preprocess_summary, "r", encoding="utf-8") as handle:
        preprocess = json.load(handle)
    expected_preprocess = preprocess.get("config", {})
    if expected_preprocess.get("spacing_mm") != [1.0, 1.0, 2.0]:
        raise ValueError("preprocessing spacing must be exactly 1x1x2 mm")
    if expected_preprocess.get("offline_spatial_matrix") != "native_after_resampling":
        raise ValueError("preprocessing must retain the native resampled matrix")
    if args.input_size != 448 or args.global_crop_size != 448:
        raise ValueError("this experiment requires 448 crop and 448 model input")
    if args.organ_roi_aug:
        with open(args.roi_index, "r", encoding="utf-8") as handle:
            roi_metadata = json.load(handle)
        if roi_metadata.get("manifest_sha256") != _sha256(args.data_path):
            raise ValueError("ROI index does not match the training manifest")
        requested_focus = sorted(
            int(value) for value in args.focus_class_ids.split(",") if value
        )
        if sorted(roi_metadata.get("focus_class_ids", ())) != requested_focus:
            raise ValueError("ROI index focus classes do not match arguments")

    classes, stages = load_visibility_config(args.visibility_config)
    stage = stages["base"]
    class_by_name = {item.name: item for item in classes}
    slot_specs = [
        {"name": name, "raw_class_id": class_by_name[name].raw_id}
        for name in stage.visible_slots
    ]
    if len(slot_specs) != 9:
        raise ValueError("Common8 requires eight foreground slots plus background")
    if args.fusion_reference_count is None:
        args.fusion_reference_count = len(slot_specs)

    train_raw, dataset_train = _build_dataset(
        args,
        args.data_path,
        classes,
        stage,
        training=True,
        max_samples=args.max_train_samples,
    )
    val_raw, _ = _build_dataset(
        args,
        args.val_data_path,
        classes,
        stage,
        training=False,
        max_samples=args.max_val_samples,
    )
    assert_case_splits_disjoint(
        {"train": train_raw.case_ids, "validation": val_raw.case_ids}
    )

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=_seed_worker,
    )
    updates_per_epoch = len(data_loader_train) // args.accum_iter
    if updates_per_epoch < 1:
        raise ValueError("not enough batches for one optimizer update")
    required_epochs = math.ceil(args.max_optimizer_updates / updates_per_epoch)
    if args.start_epoch == 0:
        args.epochs = required_epochs
    elif args.epochs < required_epochs:
        raise ValueError("epochs is too small for the requested update budget")

    model_args = _model_args(args, len(slot_specs))
    model = mae_vit_base_patch16(
        img_size=args.input_size,
        norm_pix_loss=False,
        model_args=model_args,
        slot_specs=slot_specs,
        slot_tg_depth=args.slot_tg_depth,
        fusion_mode=args.fusion_mode,
        fusion_reference_count=args.fusion_reference_count,
    )
    initial_checkpoint_report = None
    if args.init_checkpoint:
        initial_checkpoint_report = _load_initial_checkpoint(
            model, args.init_checkpoint, args
        )
    if args.lambda_lpips:
        perceptual_loss = LPIPS(vgg_pretrained=False).eval()
        lpips_state = torch.load(args.lpips_state, map_location="cpu")
        result = perceptual_loss.load_state_dict(lpips_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("LPIPS state did not load exactly: {}".format(result))
        model.perceptual_loss = perceptual_loss
    if args.training_scope == "head_only":
        model.freeze_for_all_heads(train_calibration=True)
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and ".head." not in name
            and not name.endswith("calibration_scale")
            and not name.endswith("calibration_bias")
        ]
        if unexpected:
            raise RuntimeError("head_only exposed unexpected parameters: {}".format(unexpected))
    elif args.lambda_seg == 0:
        for slot_name in model.slot_names:
            slot = model.slot_bank.get_slot(slot_name)
            for parameter in slot.head.parameters():
                parameter.requires_grad = False
            slot.calibration_scale.requires_grad = False
            slot.calibration_bias.requires_grad = False

    device = torch.device(args.device)
    model.to(device)
    model_without_ddp = model
    effective_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    print("base lr: {:.2e}".format(args.lr * 256 / effective_batch_size))
    print("actual lr: {:.2e}".format(args.lr))
    print("effective batch size: {}".format(effective_batch_size))
    print("training samples: {}".format(len(dataset_train)))
    print("steps per epoch: {}".format(len(data_loader_train)))
    print("optimizer updates per full epoch: {}".format(updates_per_epoch))
    print("max optimizer updates: {}".format(args.max_optimizer_updates))
    print("warmup updates: {}".format(args.warmup_updates))
    print("slots: {}".format(model.slot_names))
    print("training scope: {}".format(args.training_scope))
    print("trainable parameters: {}".format(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    ))
    print("organ ROI augmentation: {}".format(args.organ_roi_aug))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=(args.lambda_seg != 0),
        )
        model_without_ddp = model.module
    parameter_groups = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        parameter_groups, lr=args.lr, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    output_dir = Path(args.output_dir)
    _write_provenance(args, output_dir)
    if misc.is_main_process():
        (output_dir / "vis").mkdir(parents=True, exist_ok=True)
        with open(output_dir / "parameter_report.json", "w", encoding="utf-8") as handle:
            json.dump(model_without_ddp.parameter_report(), handle, indent=2)
        with open(output_dir / "slot_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(model_without_ddp.slot_bank.metadata(), handle, indent=2)
        if initial_checkpoint_report is not None:
            with open(
                output_dir / "initial_checkpoint_load.json",
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(initial_checkpoint_report, handle, indent=2, sort_keys=True)

    log_writer = None
    if misc.is_main_process() and args.log_dir is not None:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)

    print("Start training for {} epochs".format(args.epochs))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )
        if not args.no_save and (
            epoch % args.save_freq == 0
            or train_stats["optimizer_updates"] >= args.max_optimizer_updates
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        record = {
            "epoch": epoch,
            **{"train_{}".format(key): value for key, value in train_stats.items()},
        }
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(output_dir / "log.txt", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        if train_stats["optimizer_updates"] >= args.max_optimizer_updates:
            print("Reached optimizer update budget at epoch {}".format(epoch))
            break

    elapsed = time.time() - start_time
    print("Training time {}".format(datetime.timedelta(seconds=int(elapsed))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "OrganSlotBank Common8 1x1x2/448 pre-training",
        parents=[get_args_parser()],
    )
    main(parser.parse_args())
