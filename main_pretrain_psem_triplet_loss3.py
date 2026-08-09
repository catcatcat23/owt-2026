"""PSEM-v3 anchor/context triplet pre-training with LossBalance-v3."""

import datetime
import json
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory
from datasets.dataset3D import RandomGenerator, dataset_reader
from engine_pretrain_psem_triplet_loss3 import train_one_epoch
from main_pretrain import get_args_parser as get_base_args_parser
from util.frequency_balanced_loss import build_class_frequency_weights
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = get_base_args_parser()
    parser.description = "PSEM-v3 anchor/context triplet OWT pre-training"
    parser.add_argument(
        "--intensity_norm",
        default="per_sample",
        choices=("per_sample", "fixed_255"),
    )
    parser.add_argument(
        "--mask_schedule",
        default="triplet_context_v1",
        choices=("triplet_context_v1",),
    )
    parser.add_argument("--max_train_samples", default=0, type=int)
    parser.add_argument("--positive_roi_loss_weight", type=float, default=0.25)
    parser.add_argument(
        "--roi_positive_sample_counts", type=int, nargs="+", required=True
    )
    parser.add_argument("--roi_frequency_alpha", type=float, default=0.5)
    parser.add_argument("--roi_max_weight_ratio", type=float, default=4.0)
    parser.add_argument("--lpips_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--delta_loss_weight",
        type=float,
        default=0.1,
        help="Weight for Loss3-style supervision of R(S+c)-R(S).",
    )
    return parser


def main(args):
    misc.init_distributed_mode(args)
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"PSEM-v3 worktree: {os.getcwd()}")
    print(f"PSEM-v3 schedule: {args.mask_schedule}")
    print(f"{args}".replace(", ", ",\n"))

    device = torch.device(args.device)
    cudnn.benchmark = True

    dataset_train = dataset_reader(
        base_dir=args.data_path,
        split="train",
        num_classes=args.num_classes,
        transform=transforms.Compose([
            RandomGenerator(
                output_size=[args.input_size, args.input_size],
                low_res=[128, 128],
                renormalize_after_resize=args.intensity_norm != "fixed_255",
            )
        ]),
        model_args=args,
    )
    print(f"The length of train set is: {len(dataset_train)}")
    if len(args.roi_positive_sample_counts) != args.num_classes:
        raise ValueError("roi_positive_sample_counts must match foreground classes")
    args.roi_class_weights = build_class_frequency_weights(
        args.roi_positive_sample_counts,
        dataset_size=len(dataset_train),
        alpha=args.roi_frequency_alpha,
        max_weight_ratio=args.roi_max_weight_ratio,
    ).tolist()
    print(f"ROI class weights: {args.roi_class_weights}")

    if args.max_train_samples > 0:
        subset_size = min(args.max_train_samples, len(dataset_train))
        dataset_train = Subset(dataset_train, range(subset_size))
        print(f"PSEM-v3 smoke subset: first {subset_size} CSV rows")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=misc.get_world_size(),
        rank=misc.get_rank(),
        shuffle=True,
        drop_last=False,
    )
    print(f"Sampler_train = {sampler_train}")

    if misc.get_rank() == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler_train,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    import OWT_models_psem_lossbalance_v3 as OWT_models_psem

    model = OWT_models_psem.__dict__[args.model](
        img_size=args.input_size,
        norm_pix_loss=args.norm_pix_loss,
        model_args=args,
    )
    if args.checkpoint != "None":
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        message = model.load_state_dict(checkpoint["model"], strict=True)
        print(f"Loaded checkpoint {args.checkpoint}: {message}")

    model.to(device)
    model_without_ddp = model
    print(f"Model = {model_without_ddp}")

    # Learning-rate scaling follows unique source images, not the three query
    # views derived from each image.
    effective_batch_size = (
        args.batch_size * args.accum_iter * misc.get_world_size()
    )
    if args.lr is None:
        args.lr = args.blr * effective_batch_size / 256
    print(f"base lr: {args.lr * 256 / effective_batch_size:.2e}")
    print(f"actual lr: {args.lr:.2e}")
    print(f"effective source-image batch size: {effective_batch_size}")
    print(f"effective query batch size: {3 * effective_batch_size}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(
        model_without_ddp, args.weight_decay
    )
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr, betas=(0.9, 0.95)
    )
    loss_scaler = NativeScaler()
    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    print(f"Start PSEM-v3 triplet training for {args.epochs} epochs")
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
        if args.output_dir and (
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

        log_stats = {
            **{f"train_{key}": value for key, value in train_stats.items()},
            "epoch": epoch,
            "method": "PSEM-v3-triplet-Loss3",
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"),
                mode="a",
                encoding="utf-8",
            ) as log_file:
                log_file.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print(f"Training time {datetime.timedelta(seconds=int(total_time))}")


def finalize_args(args):
    args.num_classes_with_bg = args.num_classes + 1
    args.organ_token_total = args.token_factor * args.num_classes_with_bg
    args.cls_num = 1
    args.loss_version = args.loss_version.split("-")

    args.fix_frame = 0
    args.temp_stride = 0
    if "-3D" in args.training_version:
        args.dataset_type = "3D"
        if "-Fixfr" in args.training_version:
            args.fix_frame = int(
                args.training_version.split("-Fixfr")[1].split("-")[0]
            )
        if "-TS" in args.training_version:
            args.temp_stride = int(
                args.training_version.split("-TS")[1].split("-")[0]
            )
        args.training_version = args.training_version.split("-3D")[0]

    if "-LA" not in args.model:
        raise ValueError("PSEM-v3 requires an OWT -LA model")
    args.LA = True
    args.model = args.model.split("-LA")[0].split("-")[0]

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    parsed_args = get_args_parser().parse_args()
    main(finalize_args(parsed_args))
