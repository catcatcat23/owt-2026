"""Update-budget loop for fair legacy OWT Common8 control/ROI runs."""

import math
import random

import torch

from engine_pretrain_orgslot import perceptual_reconstruction_loss
import util.misc as misc
from util.slot_tgr import build_base_reconstruction_target


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _legacy_batch_keep_mask(batch_size, class_count, device):
    """Reproduce the original OWT batch-shared random class deletion."""
    class_order = list(range(class_count))
    random.shuffle(class_order)
    dropped_count = int(class_count * random.random())
    keep = torch.ones(batch_size, class_count, dtype=torch.bool, device=device)
    if dropped_count:
        keep[:, class_order[:dropped_count]] = False
    return keep


def _force_batch_focus_classes(batch, keep, raw_class_ids, device):
    """Retain the union of ROI focus classes while preserving batch-shared TGR.

    Legacy OWT can only pack one shared token subset for the whole batch.  A
    per-row focus override would therefore require PSEM-style padding and would
    no longer be the original architecture.  Retaining the batch focus union
    keeps the token sequence shared and guarantees that no zoomed organ is
    accidentally selected for deletion.
    """
    focus = batch.get("focus_class_id")
    roi = batch.get("roi_applied")
    if focus is None and roi is None:
        return keep, torch.zeros(keep.shape[0], dtype=torch.bool, device=device)
    if focus is None or roi is None:
        raise ValueError("focus_class_id and roi_applied must be provided together")
    focus = focus.to(device=device, dtype=torch.long)
    roi = roi.to(device=device, dtype=torch.bool)
    if focus.shape != (keep.shape[0],) or roi.shape != (keep.shape[0],):
        raise ValueError("ROI metadata must have shape [B]")
    if torch.any(roi & (focus < 0)) or torch.any(~roi & (focus >= 0)):
        raise ValueError("inconsistent ROI/focus metadata")

    raw_to_class = {int(raw_id): index for index, raw_id in enumerate(raw_class_ids)}
    for raw_id in torch.unique(focus[roi]).tolist():
        if int(raw_id) not in raw_to_class:
            raise ValueError("focus class {} has no OWT token group".format(raw_id))
        keep[:, raw_to_class[int(raw_id)]] = True
    if not torch.equal(keep, keep[:1].expand_as(keep)):
        raise RuntimeError("legacy OWT keep mask must remain batch-shared")
    return keep, roi


def _adjust_learning_rate(optimizer, completed_updates, args):
    if completed_updates < args.warmup_updates:
        lr = args.lr * completed_updates / max(1, args.warmup_updates)
    else:
        progress = (
            (completed_updates - args.warmup_updates)
            / max(1, args.max_optimizer_updates - args.warmup_updates)
        )
        progress = min(max(progress, 0.0), 1.0)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


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
    slot_names = tuple(args.slot_names)
    focus_class_ids = tuple(
        int(value) for value in args.focus_class_ids.split(",") if value
    )
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    optimizer.zero_grad()
    updates_per_epoch = len(data_loader) // args.accum_iter
    if updates_per_epoch < 1:
        raise ValueError("data loader is shorter than one accumulation cycle")
    usable_microsteps = updates_per_epoch * args.accum_iter
    completed_updates = epoch * updates_per_epoch

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        if completed_updates >= args.max_optimizer_updates:
            break
        if data_iter_step >= usable_microsteps:
            break
        if args.max_steps_per_epoch is not None and data_iter_step >= args.max_steps_per_epoch:
            break
        if data_iter_step % args.accum_iter == 0:
            _adjust_learning_rate(optimizer, completed_updates, args)

        image = batch["image"].to(device, non_blocking=True)
        visible_masks = {
            name: mask.to(device, non_blocking=True)
            for name, mask in batch["visible_masks"].items()
        }
        keep = _legacy_batch_keep_mask(
            image.shape[0], len(slot_names), device
        )
        keep, roi = _force_batch_focus_classes(
            batch, keep, args.slot_raw_ids, device
        )
        target = build_base_reconstruction_target(
            image, visible_masks, slot_names, keep
        )
        dropped_class_ids = torch.nonzero(~keep[0], as_tuple=False).flatten().tolist()

        with torch.cuda.amp.autocast():
            middle = {
                "image_target": target,
                "random_selected_class": dropped_class_ids,
            }
            reconstruction_loss, reconstruction, _ = model(
                image, mask_ratio=1.0, middle=middle
            )
            perceptual_loss = perceptual_reconstruction_loss(
                model_without_ddp, reconstruction, target
            )
            total_loss = reconstruction_loss + args.lambda_lpips * perceptual_loss

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
            completed_updates += 1

        torch.cuda.synchronize()
        values = {
            "loss": float(total_loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "p_loss": float(perceptual_loss.detach()),
            "retained_classes": float(keep.sum(dim=1).float().mean()),
            "roi_fraction": float(roi.float().mean()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        if "focus_class_id" in batch:
            focus = batch["focus_class_id"].to(device=device, dtype=torch.long)
            for class_id in focus_class_ids:
                values["roi_class_{}".format(class_id)] = float(
                    (roi & focus.eq(class_id)).float().mean()
                )
        metric_logger.update(**values)

        if update_grad:
            reduced = {
                name: misc.all_reduce_mean(value)
                for name, value in values.items()
                if name != "lr"
            }
            if log_writer is not None:
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                for name, value in reduced.items():
                    log_writer.add_scalar("train_{}".format(name), value, epoch_1000x)
                log_writer.add_scalar("lr", values["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {name: meter.global_avg for name, meter in metric_logger.meters.items()}
    stats["optimizer_updates"] = int(completed_updates)
    stats["updates_per_epoch"] = int(updates_per_epoch)
    return stats
