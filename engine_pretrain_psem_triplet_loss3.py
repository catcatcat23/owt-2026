import contextlib
import os
import sys
from typing import Iterable

import torch
import torch.distributed as dist

import util.lr_sched as lr_sched
import util.misc as misc
from util.per_sample_mask_schedule import (
    masked_reconstruction_target,
    present_class_mask,
)
from util.triplet_query_loss import triplet_delta_loss
from util.triplet_query_schedule import triplet_query_schedule


def _nonfinite_component_names(loss_components):
    """Return scalar/tensor loss names containing NaN or Inf."""
    return [
        name
        for name, value in loss_components.items()
        if not bool(torch.isfinite(torch.as_tensor(value).detach()).all())
    ]


def _check_losses_finite(
    loss_components,
    device,
    epoch,
    data_iter_step,
    sample_indices,
):
    """Fail every DDP rank with useful diagnostics if any rank is non-finite."""
    nonfinite = _nonfinite_component_names(loss_components)
    all_finite = torch.tensor(
        0.0 if nonfinite else 1.0,
        device=device,
        dtype=torch.float32,
    )
    if misc.is_dist_avail_and_initialized():
        dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if bool(all_finite.item()):
        return

    rank = misc.get_rank()
    values = {
        name: float(torch.as_tensor(value).detach().float().cpu())
        for name, value in loss_components.items()
    }
    diagnostic = (
        "non-finite loss detected: "
        f"rank={rank} epoch={epoch} step={data_iter_step} "
        f"sample_indices={sample_indices.detach().cpu().tolist()} "
        f"nonfinite_components={nonfinite} values={values}"
    )
    sys.stderr.write(diagnostic + "\n")
    sys.stderr.flush()
    raise FloatingPointError(diagnostic)


def _autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.cuda.amp.autocast(dtype=dtype)


def _check_training_tensors_finite(
    tensors, device, epoch, data_iter_step, sample_indices
):
    nonfinite = [
        name for name, value in tensors.items()
        if not bool(torch.isfinite(value.detach()).all())
    ]
    all_finite = torch.tensor(
        0.0 if nonfinite else 1.0, device=device, dtype=torch.float32
    )
    if misc.is_dist_avail_and_initialized():
        dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if bool(all_finite.item()):
        return
    diagnostic = (
        "non-finite training tensor detected: "
        f"rank={misc.get_rank()} epoch={epoch} step={data_iter_step} "
        f"sample_indices={sample_indices.detach().cpu().tolist()} "
        f"nonfinite_tensors={nonfinite}"
    )
    sys.stderr.write(diagnostic + "\n")
    sys.stderr.flush()
    raise FloatingPointError(diagnostic)


def _check_model_parameters_finite(
    model, device, epoch, data_iter_step, sample_indices
):
    model_without_ddp = model.module if hasattr(model, "module") else model
    nonfinite = [
        name for name, parameter in model_without_ddp.named_parameters()
        if not bool(torch.isfinite(parameter.detach()).all())
    ]
    all_finite = torch.tensor(
        0.0 if nonfinite else 1.0, device=device, dtype=torch.float32
    )
    if misc.is_dist_avail_and_initialized():
        dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if bool(all_finite.item()):
        return
    diagnostic = (
        "non-finite model parameter detected: "
        f"rank={misc.get_rank()} epoch={epoch} step={data_iter_step} "
        f"sample_indices={sample_indices.detach().cpu().tolist()} "
        f"parameters={nonfinite[:20]}"
    )
    sys.stderr.write(diagnostic + "\n")
    sys.stderr.flush()
    raise FloatingPointError(diagnostic)


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
    """Train one PSEM-v3 epoch with aligned anchor/context triplets."""
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
        source_batch_size = image.shape[0]
        _check_training_tensors_finite(
            {"image": image, "label": label},
            device,
            epoch,
            data_iter_step,
            sample_indices,
        )

        schedule = triplet_query_schedule(
            sample_indices,
            epoch,
            args.num_classes_with_bg,
        )
        direct_mask = schedule["direct_mask"]
        context_mask = schedule["context_mask"]
        plus_mask = schedule["plus_mask"]
        anchor_classes = schedule["anchor_classes"]

        target_direct = masked_reconstruction_target(image, label, direct_mask)
        target_context = masked_reconstruction_target(image, label, context_mask)
        target_plus = masked_reconstruction_target(image, label, plus_mask)

        triplet_image = torch.cat((image, image, image), dim=0)
        triplet_label = torch.cat((label, label, label), dim=0)
        triplet_target = torch.cat(
            (target_direct, target_context, target_plus), dim=0
        )
        triplet_keep_mask = torch.cat(
            (direct_mask, context_mask, plus_mask), dim=0
        )
        _check_training_tensors_finite(
            {
                "target_direct": target_direct,
                "target_context": target_context,
                "target_plus": target_plus,
            },
            device,
            epoch,
            data_iter_step,
            sample_indices,
        )

        kept_count = triplet_keep_mask.sum(dim=1)
        dropped_count = args.num_classes_with_bg - kept_count
        max_token_length = int(kept_count.max().item()) * args.token_factor
        valid_token_count = kept_count.sum().item() * args.token_factor
        padded_token_count = triplet_image.shape[0] * max_token_length
        padding_fraction = 1.0 - valid_token_count / max(padded_token_count, 1)
        mask_ratio = (
            dropped_count.float() / args.num_classes_with_bg
        ).mean().item()

        middle = {
            "image_target": triplet_target,
            "class_keep_mask": triplet_keep_mask,
            "label": triplet_label,
        }
        if text_features is not None:
            middle["text_features"] = text_features.unsqueeze(0).expand(
                triplet_image.shape[0], -1, -1
            )

        with _autocast_context(device, args.amp_dtype):
            branch_loss, triplet_pred, middle_output = model(
                triplet_image,
                mask_ratio=mask_ratio,
                middle=middle,
            )
            pred_direct, pred_context, pred_plus = triplet_pred.chunk(3, dim=0)
            delta_stats = triplet_delta_loss(
                pred_context=pred_context,
                pred_plus=pred_plus,
                target_direct=target_direct,
                label=label,
                direct_mask=direct_mask,
                class_weights=middle_output["roi_class_weights"],
                positive_roi_loss_weight=args.positive_roi_loss_weight,
            )
            loss = branch_loss + args.delta_loss_weight * delta_stats["loss"]
            if "LPIPS" in args.loss_version:
                loss = loss + args.lpips_loss_weight * middle_output["p_loss"]

        loss_components = {
            "total_loss": loss,
            "branch_loss": branch_loss,
            "global_recon_loss": middle_output["global_recon_loss"],
            "positive_roi_loss": middle_output["positive_roi_loss"],
            "removed_monitor_loss": middle_output["removed_monitor_loss"],
            "delta_loss": delta_stats["loss"],
            "delta_global_loss": delta_stats["global_loss"],
            "delta_positive_roi_loss": delta_stats["positive_roi_loss"],
        }
        if "LPIPS" in args.loss_version:
            loss_components["p_loss"] = middle_output["p_loss"]
        _check_losses_finite(
            loss_components,
            device,
            epoch,
            data_iter_step,
            sample_indices,
        )

        loss_value = loss.item()
        loss = loss / accum_iter
        update_grad = (data_iter_step + 1) % accum_iter == 0
        try:
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=model.parameters(),
                update_grad=update_grad,
            )
        except RuntimeError as error:
            diagnostic = (
                "non-finite gradient detected during unscale/clip: "
                f"rank={misc.get_rank()} epoch={epoch} step={data_iter_step} "
                f"sample_indices={sample_indices.detach().cpu().tolist()} "
                f"error={error}"
            )
            sys.stderr.write(diagnostic + "\n")
            sys.stderr.flush()
            raise FloatingPointError(diagnostic) from error
        if update_grad:
            if grad_norm is None or not bool(torch.isfinite(grad_norm.detach())):
                raise FloatingPointError(
                    "non-finite gradient norm at epoch {} step {}".format(
                        epoch, data_iter_step
                    )
                )
            if (
                args.finite_check_interval > 0
                and (data_iter_step // accum_iter) % args.finite_check_interval == 0
            ):
                _check_model_parameters_finite(
                    model, device, epoch, data_iter_step, sample_indices
                )
            optimizer.zero_grad()

        torch.cuda.synchronize()

        present_mask = present_class_mask(label, args.num_classes_with_bg)
        anchor_present = torch.gather(
            present_mask, 1, anchor_classes.unsqueeze(1)
        ).squeeze(1)
        direct_mse = (
            (pred_direct.detach() - target_direct).square().flatten(1).mean(dim=1)
        )
        context_mse = (
            (pred_context.detach() - target_context).square().flatten(1).mean(dim=1)
        )
        plus_mse = (
            (pred_plus.detach() - target_plus).square().flatten(1).mean(dim=1)
        )
        direct_energy = pred_direct.detach().square().flatten(1).mean(dim=1)
        delta_energy = delta_stats["prediction"].detach().square().flatten(1).mean(dim=1)
        absent_count = (~anchor_present).sum().item()

        if data_iter_step == 0 and misc.is_main_process():
            print(
                "PSEM-v3 triplet first batch:",
                f"anchors={anchor_classes.tolist()[:8]}",
                f"anchor_present={anchor_present.tolist()[:8]}",
                f"full_context={schedule['full_context'].tolist()[:8]}",
                f"direct_kept={direct_mask.sum(dim=1).tolist()[:8]}",
                f"context_kept={context_mask.sum(dim=1).tolist()[:8]}",
                f"plus_kept={plus_mask.sum(dim=1).tolist()[:8]}",
            )

        metric_logger.update(
            loss=loss_value,
            branch_loss=branch_loss.item(),
            global_recon_loss=middle_output["global_recon_loss"].item(),
            positive_roi_loss=middle_output["positive_roi_loss"].item(),
            removed_monitor_loss=middle_output["removed_monitor_loss"].item(),
            delta_loss=delta_stats["loss"].item(),
            delta_global_loss=delta_stats["global_loss"].item(),
            delta_positive_roi_loss=delta_stats["positive_roi_loss"].item(),
            direct_mse=direct_mse.mean().item(),
            context_mse=context_mse.mean().item(),
            plus_mse=plus_mse.mean().item(),
            anchor_present_fraction=anchor_present.float().mean().item(),
            full_context_fraction=schedule["full_context"].float().mean().item(),
            direct_kept_classes=direct_mask.sum(dim=1).float().mean().item(),
            context_kept_classes=context_mask.sum(dim=1).float().mean().item(),
            plus_kept_classes=plus_mask.sum(dim=1).float().mean().item(),
            padding_fraction=padding_fraction,
            negative_direct_energy_sum=direct_energy[~anchor_present].sum().item(),
            negative_delta_energy_sum=delta_energy[~anchor_present].sum().item(),
            negative_anchor_count=absent_count,
            lr=optimizer.param_groups[0]["lr"],
        )
        if update_grad:
            metric_logger.update(grad_norm=grad_norm.item())
        if "LPIPS" in args.loss_version:
            metric_logger.update(p_loss=middle_output["p_loss"].item())

        for class_id in range(1, args.num_classes_with_bg):
            selected = anchor_classes == class_id
            metric_logger.update(**{
                f"anchor_c{class_id}_fraction": selected.float().mean().item(),
                f"anchor_c{class_id}_present_fraction": (
                    selected & anchor_present
                ).float().mean().item(),
            })

        reduced_loss = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (data_iter_step / len(data_loader) + epoch) * 1000
            )
            log_writer.add_scalar("train_loss", reduced_loss, epoch_1000x)
            log_writer.add_scalar(
                "triplet/delta_loss", delta_stats["loss"].item(), epoch_1000x
            )
            log_writer.add_scalar(
                "triplet/anchor_present_fraction",
                anchor_present.float().mean().item(),
                epoch_1000x,
            )
            log_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    stats = {key: meter.global_avg for key, meter in metric_logger.meters.items()}
    negative_count = stats.pop("negative_anchor_count")
    stats["negative_direct_energy"] = (
        stats.pop("negative_direct_energy_sum") / max(negative_count, 1e-12)
    )
    stats["negative_delta_energy"] = (
        stats.pop("negative_delta_energy_sum") / max(negative_count, 1e-12)
    )
    stats["negative_anchor_count"] = negative_count
    return stats
