"""Formal A100 training loop for the legacy-matched OrganSlot comparison.

This loop intentionally mirrors ``engine_pretrain.py``: iteration-level
warmup/cosine scheduling, CUDA AMP, gradient accumulation, DDP metric
reduction, and L2+LPIPS.  The only intended model-side change is replacing the
joint OWT token path with OrganSlotBank.
"""

import random

import torch

from engine_pretrain_orgslot import perceptual_reconstruction_loss
from losses_orgslot import base_reconstruction_loss, base_segmentation_loss
import util.lr_sched as lr_sched
import util.misc as misc
from util.slot_tgr import (
    build_base_reconstruction_target,
    sample_base_slot_keep_mask,
)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _legacy_batch_keep_mask(batch_size, slot_count, device):
    """Reproduce legacy OWT's one random class subset per local DDP batch."""
    class_order = list(range(slot_count))
    random.shuffle(class_order)
    mask_ratio = random.random()
    dropped_count = int(slot_count * mask_ratio)
    keep = torch.ones(batch_size, slot_count, dtype=torch.bool, device=device)
    if dropped_count:
        keep[:, class_order[:dropped_count]] = False
    return keep


def _slot_keep_mask(args, batch, epoch, data_iter_step, slot_count, device):
    del data_iter_step
    if args.tgr_mode == "legacy_batch":
        return _legacy_batch_keep_mask(
            batch["image"].shape[0], slot_count, device
        )
    return sample_base_slot_keep_mask(
        batch["sample_index"].to(device),
        epoch,
        slot_count,
        seed=args.seed,
    )


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    device,
    epoch,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    model_without_ddp = _unwrap_model(model)
    slot_names = model_without_ddp.slot_names
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        if args.max_steps_per_epoch is not None:
            if data_iter_step >= args.max_steps_per_epoch:
                break
        if data_iter_step % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args,
            )

        image = batch["image"].to(device, non_blocking=True)
        visible_masks = {
            name: mask.to(device, non_blocking=True)
            for name, mask in batch["visible_masks"].items()
        }
        batch["image"] = image
        keep = _slot_keep_mask(
            args,
            batch,
            epoch,
            data_iter_step,
            len(slot_names),
            device,
        )
        target = build_base_reconstruction_target(
            image,
            visible_masks,
            slot_names,
            keep,
        )

        with torch.cuda.amp.autocast():
            output = model(image, slot_keep_mask=keep)
            reconstruction_loss = base_reconstruction_loss(
                output["reconstruction"], target
            )
            perceptual_loss = perceptual_reconstruction_loss(
                model_without_ddp,
                output["reconstruction"],
                target,
            )
            segmentation_loss = reconstruction_loss.detach() * 0.0
            if args.lambda_seg != 0:
                segmentation_loss, _ = base_segmentation_loss(
                    output["calibrated_logits"],
                    visible_masks,
                    slot_names,
                    keep,
                    background_weight=args.lambda_bg_seg,
                )
            total_loss = (
                reconstruction_loss
                + args.lambda_lpips * perceptual_loss
                + args.lambda_seg * segmentation_loss
            )

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "non-finite loss at epoch {} step {}: {}".format(
                    epoch, data_iter_step, float(total_loss.detach())
                )
            )

        scaled_loss = total_loss / args.accum_iter
        update_grad = (data_iter_step + 1) % args.accum_iter == 0
        loss_scaler(
            scaled_loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=update_grad,
        )
        if update_grad:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        values = {
            "loss": float(total_loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "p_loss": float(perceptual_loss.detach()),
            "segmentation_loss": float(segmentation_loss.detach()),
            "retained_slots": float(keep.sum(dim=1).float().mean()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        metric_logger.update(**values)

        reduced_values = None
        if update_grad:
            reduced_values = {
                name: misc.all_reduce_mean(values[name])
                for name in (
                    "loss",
                    "reconstruction_loss",
                    "p_loss",
                    "segmentation_loss",
                    "retained_slots",
                )
            }
        if log_writer is not None and update_grad:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            for name in (
                "loss",
                "reconstruction_loss",
                "p_loss",
                "segmentation_loss",
                "retained_slots",
            ):
                log_writer.add_scalar(
                    "train_{}".format(name),
                    reduced_values[name],
                    epoch_1000x,
                )
            log_writer.add_scalar("lr", values["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {name: meter.global_avg for name, meter in metric_logger.meters.items()}
