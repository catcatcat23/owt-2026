import math
import os
import sys
from typing import Iterable

import numpy as np
import torch
from PIL import Image

import util.lr_sched as lr_sched
import util.misc as misc
from util.per_sample_mask_schedule import (
    MODE_DIRECT_NEGATIVE,
    MODE_DIRECT_POSITIVE,
    MODE_LEAVE_ONE_OUT,
    QUERY_MODE_NAMES,
    masked_reconstruction_target,
    present_class_mask,
    query_matched_schedule,
)


def _save_first_batch(image, image_target, label, pred, args):
    os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)
    if args.dataset_type == "2D":
        arrays = {
            "0_pred.png": pred[-1],
            "0_image_target.png": image_target[-1],
            "0_image.png": image[-1],
        }
        for filename, tensor in arrays.items():
            array = tensor.detach().cpu().numpy()
            array = np.transpose((array * 255).astype(np.uint8), (1, 2, 0))
            Image.fromarray(array).save(os.path.join(args.output_dir, "vis", filename))
        mask = label[-1, 0].detach().cpu().numpy().astype(np.uint8)
        Image.fromarray(mask).save(os.path.join(args.output_dir, "vis", "0_mask.png"))
        return

    for slice_index in range(pred.shape[2]):
        arrays = {
            f"0_pred_{slice_index}.png": pred[-1, :, slice_index],
            f"0_image_target_{slice_index}.png": image_target[-1, :, slice_index],
            f"0_image_{slice_index}.png": image[-1, :, slice_index],
        }
        for filename, tensor in arrays.items():
            array = tensor.detach().cpu().numpy()
            array = np.transpose((array * 255).astype(np.uint8), (1, 2, 0))
            Image.fromarray(array).save(os.path.join(args.output_dir, "vis", filename))


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = f"Epoch: [{epoch}]"
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    text_features = None
    if args.text_encoding != "None":
        text_features = torch.load(args.text_encoding, map_location="cpu").to(device)
        text_features = text_features.repeat_interleave(args.token_factor, dim=0)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, 20, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        sample_indices = batch["sample_index"].to(device, non_blocking=True)

        present_mask = present_class_mask(label, args.num_classes_with_bg)
        keep_mask, drop_mask, slots, mode_ids, query_classes = (
            query_matched_schedule(present_mask, sample_indices, epoch)
        )
        image_target = masked_reconstruction_target(image, label, keep_mask)

        present_count = present_mask.sum(dim=1)
        kept_count = keep_mask.sum(dim=1)
        dropped_count = drop_mask.sum(dim=1)
        max_token_length = int(kept_count.max().item()) * args.token_factor
        valid_token_count = kept_count.sum().item() * args.token_factor
        padded_token_count = image.shape[0] * max_token_length
        padding_fraction = 1.0 - valid_token_count / max(padded_token_count, 1)
        mask_ratio = (
            dropped_count.float() / args.num_classes_with_bg
        ).mean().item()

        middle = {
            "image_target": image_target,
            "class_keep_mask": keep_mask,
            "label": label,
        }
        if text_features is not None:
            middle["text_features"] = text_features.unsqueeze(0).expand(
                image.shape[0], -1, -1
            )

        with torch.cuda.amp.autocast():
            loss, pred, middle_output = model(
                image, mask_ratio=mask_ratio, middle=middle
            )

        reconstruction_loss_value = loss.item()
        per_sample_mse = (
            (pred.detach() - image_target).square().flatten(1).mean(dim=1)
        )
        prediction_energy = pred.detach().square().flatten(1).mean(dim=1)
        negative_query = mode_ids == MODE_DIRECT_NEGATIVE
        negative_hallucination_sum = (
            prediction_energy[negative_query].sum().item()
        )
        negative_query_count = negative_query.sum().item()
        perceptual_loss_value = 0.0
        if "LPIPS" in args.loss_version:
            perceptual_loss_value = middle_output["p_loss"].item()
            loss = loss + args.lpips_loss_weight * middle_output["p_loss"]

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        if data_iter_step == 0 and misc.is_main_process():
            _save_first_batch(image, image_target, label, pred, args)
            print(
                "PSEM first batch:",
                f"present={present_count.tolist()[:8]}",
                f"kept={kept_count.tolist()[:8]}",
                f"slots={slots.tolist()[:8]}",
                "modes=" + str([
                    QUERY_MODE_NAMES[int(value)]
                    for value in mode_ids.tolist()[:8]
                ]),
                f"query_classes={query_classes.tolist()[:8]}",
            )

        metric_logger.update(loss=reconstruction_loss_value)
        metric_logger.update(
            global_recon_loss=middle_output["global_recon_loss"].item(),
            positive_roi_loss=middle_output["positive_roi_loss"].item(),
            removed_monitor_loss=middle_output["removed_monitor_loss"].item(),
        )
        if "LPIPS" in args.loss_version:
            metric_logger.update(p_loss=perceptual_loss_value)
        metric_logger.update(
            present_classes=present_count.float().mean().item(),
            kept_classes=kept_count.float().mean().item(),
            dropped_classes=dropped_count.float().mean().item(),
            padding_fraction=padding_fraction,
            lr=optimizer.param_groups[0]["lr"],
        )

        for mode_id, mode_name in QUERY_MODE_NAMES.items():
            selected = mode_ids == mode_id
            metric_logger.update(**{
                f"mode_{mode_name}_fraction": selected.float().mean().item(),
                f"{mode_name}_mse_sum": per_sample_mse[selected].sum().item(),
                f"{mode_name}_sample_count": selected.sum().item(),
            })

        queried_present = torch.gather(
            present_mask, 1, query_classes.clamp_min(0).unsqueeze(1)
        ).squeeze(1)
        leave_one_out = mode_ids == MODE_LEAVE_ONE_OUT
        metric_logger.update(
            leave_one_out_present_fraction=(
                leave_one_out & queried_present
            ).float().mean().item(),
            leave_one_out_absent_fraction=(
                leave_one_out & ~queried_present
            ).float().mean().item(),
        )
        metric_logger.update(
            negative_hallucination_sum=negative_hallucination_sum,
            negative_query_count=negative_query_count,
        )

        for class_id in range(1, args.num_classes_with_bg):
            metric_logger.update(**{
                f"direct_positive_c{class_id}_fraction": (
                    (mode_ids == MODE_DIRECT_POSITIVE)
                    & (query_classes == class_id)
                ).float().mean().item(),
                f"direct_negative_c{class_id}_fraction": (
                    (mode_ids == MODE_DIRECT_NEGATIVE)
                    & (query_classes == class_id)
                ).float().mean().item(),
            })

        reduced_loss = misc.all_reduce_mean(reconstruction_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("train_loss", reduced_loss, epoch_1000x)
            log_writer.add_scalar(
                "psem/kept_classes",
                kept_count.float().mean().item(),
                epoch_1000x,
            )
            log_writer.add_scalar(
                "psem/padding_fraction", padding_fraction, epoch_1000x
            )
            if negative_query_count > 0:
                log_writer.add_scalar(
                    "psem/negative_hallucination_energy",
                    negative_hallucination_sum / negative_query_count,
                    epoch_1000x,
                )
            log_writer.add_scalar(
                "lr", optimizer.param_groups[0]["lr"], epoch_1000x
            )

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {
        key: meter.global_avg for key, meter in metric_logger.meters.items()
    }
    for mode_name in QUERY_MODE_NAMES.values():
        mse_sum = stats.pop(f"{mode_name}_mse_sum")
        sample_count = stats.pop(f"{mode_name}_sample_count")
        stats[f"{mode_name}_mse"] = mse_sum / max(sample_count, 1e-12)
    hallucination_sum = stats.pop("negative_hallucination_sum")
    negative_count = stats.pop("negative_query_count")
    stats["negative_hallucination_energy"] = (
        hallucination_sum / max(negative_count, 1e-12)
    )
    stats["negative_query_count"] = negative_count
    return stats
