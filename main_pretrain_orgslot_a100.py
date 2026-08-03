"""Legacy-matched full A100 trainer for OrganSlotBank.

The command-line defaults mirror ``run_OWT_A100.sh`` for the AutoPET 2D
comparison.  Unlike the Gate-A runner, this entry point supports DDP, AMP,
warmup/cosine scheduling, effective-batch LR scaling, resume, and periodic
optimizer/scaler checkpoints.
"""

import argparse
import datetime
import hashlib
import json
import os
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
import torchvision.transforms as transforms

from datasets.dataset3D import dataset_reader, RandomGenerator
from engine_pretrain_orgslot_a100 import train_one_epoch
from OWT_models_orgslot import mae_vit_base_patch16
from VQ.lpips import LPIPS
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.label_visibility import StrictVisibilityDataset, load_visibility_config


def get_args_parser():
    parser = argparse.ArgumentParser(
        "OrganSlotBank legacy-matched pre-training", add_help=False
    )
    parser.add_argument("--batch_size", default=96, type=int)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--accum_iter", default=1, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--token_factor", default=20, type=int)
    parser.add_argument("--slot_tg_depth", default=1, type=int)
    parser.add_argument(
        "--fusion_mode",
        choices=("post_layernorm", "linear_sqrt"),
        default="post_layernorm",
    )

    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--blr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--warmup_epochs", default=60, type=int)

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument(
        "--visibility_config",
        default="configs/orgslot/autopet_all4_offline.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--loss_version", default="L2-LPIPS")
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_seg", default=0.0, type=float)
    parser.add_argument("--lambda_bg_seg", default=0.25, type=float)
    parser.add_argument(
        "--lpips_state",
        default="/mnt/DATA-4/anteng/pretrained/owt_lpips_vgg16.pth",
        help="LPIPS-only state extracted from the matched original OWT checkpoint",
    )
    parser.add_argument(
        "--tgr_mode",
        choices=("legacy_batch", "per_sample"),
        default="legacy_batch",
    )
    parser.add_argument("--save_freq", default=100, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--max_steps_per_epoch", default=None, type=int)
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="skip checkpoints for short pipeline smoke tests",
    )

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
    with open(output_dir / "dataset_manifest_checksums.json", "w", encoding="utf-8") as handle:
        json.dump({"train": _sha256(args.data_path)}, handle, indent=2)


def _model_args(args, slot_count, initialize_lpips=True):
    loss_version = args.loss_version.split("-")
    if not initialize_lpips:
        loss_version = [name for name in loss_version if name != "LPIPS"]
    return SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=args.token_factor,
        organ_token_total=args.token_factor * slot_count,
        fix_frame=0,
        temp_stride=0,
        loss_version=loss_version,
        text_encoding="None",
    )


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
    if args.lambda_lpips and not Path(args.lpips_state).is_file():
        raise FileNotFoundError(
            "local LPIPS state is required before DDP launch: {}".format(
                args.lpips_state
            )
        )
    if args.lambda_seg != 0:
        print(
            "WARNING: lambda_seg != 0 is not a loss-matched comparison to "
            "the original OWT checkpoint"
        )

    classes, stages = load_visibility_config(args.visibility_config)
    stage = stages["base"]
    class_by_name = {item.name: item for item in classes}
    slot_specs = [
        {"name": name, "raw_class_id": class_by_name[name].raw_id}
        for name in stage.visible_slots
    ]
    if len(slot_specs) != 5:
        raise ValueError(
            "legacy AutoPET all-class comparison requires 5 slots including background"
        )

    legacy_dataset = dataset_reader(
        base_dir=args.data_path,
        split="train",
        num_classes=4,
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.input_size, args.input_size], low_res=[128, 128])]
        ),
        model_args=_model_args(args, len(slot_specs), initialize_lpips=False),
    )
    dataset_train = StrictVisibilityDataset(legacy_dataset, classes, stage)
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
    )

    model = mae_vit_base_patch16(
        img_size=args.input_size,
        norm_pix_loss=False,
        model_args=_model_args(args, len(slot_specs), initialize_lpips=False),
        slot_specs=slot_specs,
        slot_tg_depth=args.slot_tg_depth,
        fusion_mode=args.fusion_mode,
    )
    if args.lambda_lpips:
        perceptual_loss = LPIPS(vgg_pretrained=False).eval()
        lpips_state = torch.load(args.lpips_state, map_location="cpu")
        result = perceptual_loss.load_state_dict(lpips_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("LPIPS state did not load exactly: {}".format(result))
        model.perceptual_loss = perceptual_loss
    if args.lambda_seg == 0:
        for slot_name in model.slot_names:
            slot = model.slot_bank.get_slot(slot_name)
            for parameter in slot.head.parameters():
                parameter.requires_grad = False
            slot.calibration_scale.requires_grad = False
            slot.calibration_bias.requires_grad = False
    device = torch.device(args.device)
    model.to(device)
    model_without_ddp = model

    effective_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size()
    )
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    print("base lr: {:.2e}".format(args.lr * 256 / effective_batch_size))
    print("actual lr: {:.2e}".format(args.lr))
    print("effective batch size: {}".format(effective_batch_size))
    print("training samples: {}".format(len(dataset_train)))
    print("steps per epoch: {}".format(len(data_loader_train)))
    print("slots: {}".format(model.slot_names))

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
            epoch % args.save_freq == 0 or epoch + 1 == args.epochs
        ):
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        record = {"epoch": epoch, **{"train_{}".format(k): v for k, v in train_stats.items()}}
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(output_dir / "log.txt", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    elapsed = time.time() - start_time
    print("Training time {}".format(datetime.timedelta(seconds=int(elapsed))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "OrganSlotBank legacy-matched pre-training",
        parents=[get_args_parser()],
    )
    main(parser.parse_args())
