"""AutoPET image-only MAE pretraining for WORD OrganSlot transfer."""

import argparse
import contextlib
import datetime
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets.mae_manifest import (
    MAEManifestDataset,
    read_image_records,
    split_records_by_case,
)
from models_mae_transfer import mae_transfer_base_patch16
import util.misc as misc


def get_args_parser():
    parser = argparse.ArgumentParser("AutoPET architecture-matched MAE")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--mask_ratio", default=0.75, type=float)
    parser.add_argument("--norm_pix_loss", action="store_true")
    parser.add_argument("--validation_fraction", default=0.1, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--accum_iter", default=6, type=int)
    parser.add_argument("--max_optimizer_updates", default=100000, type=int)
    parser.add_argument("--warmup_updates", default=5000, type=int)
    parser.add_argument("--blr", default=1e-4, type=float)
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument("--weight_decay", default=0.05, type=float)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--save_update_freq", default=10000, type=int)
    parser.add_argument("--validation_epoch_freq", default=10, type=int)
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max_train_samples", type=int)
    parser.add_argument("--max_val_samples", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def _set_lr(optimizer, update, args):
    if update < args.warmup_updates:
        factor = float(update + 1) / max(1, args.warmup_updates)
        learning_rate = args.lr * factor
    else:
        progress = (update - args.warmup_updates) / max(
            1, args.max_optimizer_updates - args.warmup_updates
        )
        progress = min(max(progress, 0.0), 1.0)
        learning_rate = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _all_ranks_finite(value):
    finite = torch.tensor(
        [int(torch.isfinite(value.detach()).all())],
        dtype=torch.int32,
        device=value.device,
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
    return bool(finite.item())


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, data_loader, device, mask_ratio):
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    sample_count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        with _autocast(device):
            loss, _, _ = model(images, mask_ratio=mask_ratio)
        if not _all_ranks_finite(loss):
            raise FloatingPointError("non-finite validation MAE loss")
        loss_sum += loss.detach().double() * images.shape[0]
        sample_count += images.shape[0]
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(loss_sum)
        torch.distributed.all_reduce(sample_count)
    model.train()
    return float((loss_sum / sample_count.clamp_min(1.0)).item())


def _save_checkpoint(path, model, optimizer, epoch, updates, args):
    if not misc.is_main_process():
        return
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "optimizer_updates": int(updates),
        "input_size": int(args.input_size),
        "mask_ratio": float(args.mask_ratio),
        "encoder_depth": 6,
        "decoder_depth": 8,
        "decoder_dim": 768,
        "args": args,
    }
    torch.save(payload, path)


def main(args):
    misc.init_distributed_mode(args)
    rank = misc.get_rank()
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    if args.input_size != 224:
        raise ValueError("the available AutoPET manifest is native 224; use input_size 224")
    if not Path(args.data_path).is_file():
        raise FileNotFoundError(args.data_path)
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(args.resume)
    if not 0.0 < args.mask_ratio < 1.0:
        raise ValueError("mask_ratio must be in (0,1)")
    if args.max_optimizer_updates <= 0 or args.warmup_updates < 0:
        raise ValueError("invalid optimizer update budget")
    if args.warmup_updates >= args.max_optimizer_updates:
        raise ValueError("warmup must be shorter than the total update budget")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.device == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required for this experiment")

    records = read_image_records(args.data_path)
    train_records, validation_records = split_records_by_case(
        records, args.validation_fraction, args.seed
    )
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        validation_records = validation_records[: args.max_val_samples]
    train_dataset = MAEManifestDataset(
        train_records, args.input_size, training=True
    )
    validation_dataset = MAEManifestDataset(
        validation_records, args.input_size, training=False
    )
    if set(train_dataset.case_ids) & set(validation_dataset.case_ids):
        raise RuntimeError("AutoPET patient leakage between MAE train and validation")

    world_size = misc.get_world_size()
    train_sampler = DistributedSampler(
        train_dataset, world_size, rank, shuffle=True, seed=args.seed
    )
    validation_sampler = DistributedSampler(
        validation_dataset, world_size, rank, shuffle=False
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    updates_per_epoch = len(train_loader) // args.accum_iter
    if updates_per_epoch < 1:
        raise ValueError("not enough batches for one optimizer update")

    device = torch.device(args.device)
    model = mae_transfer_base_patch16(
        img_size=args.input_size, norm_pix_loss=args.norm_pix_loss
    ).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    effective_batch = args.batch_size * args.accum_iter * world_size
    args.lr = args.blr * effective_batch / 256.0
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    start_epoch = 0
    optimizer_updates = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_updates = int(checkpoint["optimizer_updates"])

    output_dir = Path(args.output_dir)
    if misc.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "split.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "seed": args.seed,
                    "validation_fraction": args.validation_fraction,
                    "train_cases": list(train_dataset.case_ids),
                    "validation_cases": list(validation_dataset.case_ids),
                    "train_slices": len(train_dataset),
                    "validation_slices": len(validation_dataset),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)

    print("AutoPET MAE train/validation slices: {}/{}".format(
        len(train_dataset), len(validation_dataset)
    ))
    print("effective batch: {}, actual lr: {:.3e}".format(effective_batch, args.lr))
    print("updates per epoch: {}, target updates: {}".format(
        updates_per_epoch, args.max_optimizer_updates
    ))

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    epoch = start_epoch
    while optimizer_updates < args.max_optimizer_updates:
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_updates = 0
        model.train()
        usable_steps = updates_per_epoch * args.accum_iter
        for step, batch in enumerate(train_loader):
            if step >= usable_steps or optimizer_updates >= args.max_optimizer_updates:
                break
            images = batch["image"].to(device, non_blocking=True)
            with _autocast(device):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)
            if not _all_ranks_finite(loss):
                raise FloatingPointError(
                    "non-finite MAE loss at epoch {} step {} update {} samples {}".format(
                        epoch,
                        step,
                        optimizer_updates,
                        batch["sample_index"].tolist(),
                    )
                )
            (loss / args.accum_iter).backward()
            if (step + 1) % args.accum_iter != 0:
                continue
            learning_rate = _set_lr(optimizer, optimizer_updates, args)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model_without_ddp.parameters(),
                args.clip_grad,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_updates += 1
            epoch_updates += 1
            epoch_loss += float(loss.detach())
            if misc.is_main_process() and optimizer_updates % 100 == 0:
                print(
                    "epoch={} update={}/{} loss={:.6f} lr={:.3e} grad_norm={:.4f}".format(
                        epoch,
                        optimizer_updates,
                        args.max_optimizer_updates,
                        float(loss.detach()),
                        learning_rate,
                        float(grad_norm),
                    ),
                    flush=True,
                )
            if optimizer_updates % args.save_update_freq == 0:
                _save_checkpoint(
                    output_dir / "checkpoint-update{}.pth".format(optimizer_updates),
                    model_without_ddp,
                    optimizer,
                    epoch,
                    optimizer_updates,
                    args,
                )

        record = {
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "train_loss": epoch_loss / max(1, epoch_updates),
        }
        if (
            (epoch + 1) % args.validation_epoch_freq == 0
            or optimizer_updates >= args.max_optimizer_updates
        ):
            record["validation_loss"] = evaluate(
                model, validation_loader, device, args.mask_ratio
            )
        if misc.is_main_process():
            with open(output_dir / "log.txt", "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        epoch += 1

    _save_checkpoint(
        output_dir / "checkpoint-final.pth",
        model_without_ddp,
        optimizer,
        epoch - 1,
        optimizer_updates,
        args,
    )
    elapsed = datetime.timedelta(seconds=int(time.time() - start_time))
    print("MAE pretraining complete in {}".format(elapsed))


if __name__ == "__main__":
    main(get_args_parser().parse_args())
