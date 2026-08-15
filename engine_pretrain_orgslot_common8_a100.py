"""Common8 A100 loop with focus-slot retention for organ-aware ROI samples."""

import random
import math

import torch

from engine_pretrain_orgslot import perceptual_reconstruction_loss
from losses_orgslot import base_reconstruction_loss, base_segmentation_loss
import util.misc as misc
from util.slot_tgr import build_base_reconstruction_target, sample_base_slot_keep_mask


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _legacy_batch_keep_mask(batch_size, slot_count, device):
    class_order = list(range(slot_count))
    random.shuffle(class_order)
    dropped_count = int(slot_count * random.random())
    keep = torch.ones(batch_size, slot_count, dtype=torch.bool, device=device)
    if dropped_count:
        keep[:, class_order[:dropped_count]] = False
    return keep


def _slot_keep_mask(args, batch, epoch, slot_count, device):
    if args.tgr_mode == "legacy_batch":
        return _legacy_batch_keep_mask(batch["image"].shape[0], slot_count, device)
    if args.tgr_mode == "fixed_per_sample":
        epoch = 0
    return sample_base_slot_keep_mask(
        batch["sample_index"].to(device), epoch, slot_count, seed=args.seed
    )


def _segmentation_supervision_mask(mode, keep):
    if mode == "retained":
        return keep
    if mode == "all":
        return torch.ones_like(keep)
    raise ValueError("unknown segmentation supervision mode: {}".format(mode))


def _force_focus_slots(model_without_ddp, batch, keep, device):
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

    raw_to_slot = {
        int(model_without_ddp.slot_bank.get_slot(name).raw_class_id): index
        for index, name in enumerate(model_without_ddp.slot_names)
    }
    for raw_id in torch.unique(focus[roi]).tolist():
        if int(raw_id) not in raw_to_slot:
            raise ValueError("focus class {} has no OrganSlot".format(raw_id))
        rows = roi & focus.eq(int(raw_id))
        keep[rows, raw_to_slot[int(raw_id)]] = True
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
    slot_names = model_without_ddp.slot_names
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
        batch["image"] = image
        if args.training_scope == "head_only":
            keep = torch.ones(
                image.shape[0],
                len(slot_names),
                dtype=torch.bool,
                device=device,
            )
            roi = batch.get("roi_applied")
            if roi is None:
                roi = torch.zeros(image.shape[0], dtype=torch.bool, device=device)
            else:
                roi = roi.to(device=device, dtype=torch.bool)
                if roi.shape != (image.shape[0],):
                    raise ValueError("roi_applied must have shape [B]")
            target = None
        else:
            keep = _slot_keep_mask(args, batch, epoch, len(slot_names), device)
            keep, roi = _force_focus_slots(model_without_ddp, batch, keep, device)
            target = build_base_reconstruction_target(
                image, visible_masks, slot_names, keep
            )

        with torch.cuda.amp.autocast():
            if args.training_scope == "head_only":
                output = model(
                    image,
                    slot_keep_mask=keep,
                    decode_reconstruction=False,
                )
                segmentation_keep = _segmentation_supervision_mask(
                    args.seg_supervision, keep
                )
                segmentation_loss, per_slot_segmentation = base_segmentation_loss(
                    output["calibrated_logits"],
                    visible_masks,
                    slot_names,
                    segmentation_keep,
                    background_weight=args.lambda_bg_seg,
                    loss_type=args.seg_loss_type,
                    focal_alpha=args.focal_alpha,
                    focal_gamma=args.focal_gamma,
                )
                reconstruction_loss = segmentation_loss.detach() * 0.0
                perceptual_loss = segmentation_loss.detach() * 0.0
                total_loss = args.lambda_seg * segmentation_loss
            else:
                output = model(image, slot_keep_mask=keep)
                reconstruction_loss = base_reconstruction_loss(
                    output["reconstruction"], target
                )
                perceptual_loss = perceptual_reconstruction_loss(
                    model_without_ddp, output["reconstruction"], target
                )
                segmentation_loss = reconstruction_loss.detach() * 0.0
                segmentation_keep = _segmentation_supervision_mask(
                    args.seg_supervision, keep
                )
                per_slot_segmentation = {}
                if args.lambda_seg != 0:
                    segmentation_loss, per_slot_segmentation = base_segmentation_loss(
                        output["calibrated_logits"],
                        visible_masks,
                        slot_names,
                        segmentation_keep,
                        background_weight=args.lambda_bg_seg,
                        loss_type=args.seg_loss_type,
                        focal_alpha=args.focal_alpha,
                        focal_gamma=args.focal_gamma,
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
            completed_updates += 1

        torch.cuda.synchronize()
        values = {
            "loss": float(total_loss.detach()),
            "reconstruction_loss": float(reconstruction_loss.detach()),
            "p_loss": float(perceptual_loss.detach()),
            "segmentation_loss": float(segmentation_loss.detach()),
            "weighted_segmentation_loss": float(
                (args.lambda_seg * segmentation_loss).detach()
            ),
            "retained_slots": float(keep.sum(dim=1).float().mean()),
            "roi_fraction": float(roi.float().mean()),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for slot_index, slot_name in enumerate(slot_names):
            if slot_name not in per_slot_segmentation:
                continue
            values["seg_{}_loss".format(slot_name)] = float(
                per_slot_segmentation[slot_name].detach()
            )
            values["seg_{}_supervised_samples".format(slot_name)] = float(
                segmentation_keep[:, slot_index].sum()
            )
            probabilities = torch.sigmoid(
                output["calibrated_logits"][slot_name].detach()
            )
            target_mask = visible_masks[slot_name].bool()
            values["seg_{}_predicted_fraction".format(slot_name)] = float(
                probabilities.ge(0.5).float().mean()
            )
            values["seg_{}_target_fraction".format(slot_name)] = float(
                target_mask.float().mean()
            )
            if torch.any(target_mask):
                positive_probability = probabilities[target_mask].mean()
            else:
                positive_probability = probabilities.sum() * 0.0
            values["seg_{}_positive_probability".format(slot_name)] = float(
                positive_probability
            )
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
