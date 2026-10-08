"""Controlled H-SAM-inspired coarse supervision; original losses unchanged."""
import torch
from torch.nn import functional as F
from losses_orgslot import base_segmentation_loss


def stage_weight(epoch):
    return 0.6 ** (0.990 ** epoch)


def combined_coarse_losses(logits, targets, names, keep, args, diagnostics):
    """Tail coarse: nearest-GT loss; optional pre-read priors: full-GT loss."""
    from copy import copy
    stage_args = copy(args)
    stage_args.hsam_supervision = "downsample_gt"
    coarse, lost = coarse_segmentation_loss(
        {name: value[0] for name, value in logits.items()},
        targets, names, keep, stage_args)
    aux = coarse.new_zeros(())
    if args.hsam_supervision == "downsample_gt_soft_prior_aux":
        stage_args.hsam_supervision = "soft_prior_aux"
        aux, _ = coarse_segmentation_loss(
            {name: value[1:] for name, value in logits.items()},
            targets, names, keep, stage_args, diagnostics)
    return coarse, lost, aux


def coarse_segmentation_loss(logits, targets, names, keep, args, diagnostics=None):
    # TGR can produce a background-only rank while peers retain foreground.
    # Establish the metric schema before the legal empty-prediction return.
    if args.hsam_supervision == "soft_prior_aux" and diagnostics is not None:
        diagnostics.update({"soft_prior_p{}_loss".format(scale): 0.0 for scale in (16, 8, 4)})
    # TGR may retain only background on a rank with no ROI focus. Background
    # has no head when its loss weight is zero; an empty dictionary is legal.
    supervised = [name for index, name in enumerate(names)
                  if (name != "background" or args.lambda_bg_seg != 0)
                  and bool(keep[:, index].any())]
    missing = [name for name in supervised if name not in logits]
    if missing:
        raise ValueError("Missing coarse predictions for supervised slots: {}".format(missing))
    if not logits:
        # Reconstruction retains its normal autograd graph. Do not skip the
        # training iteration/backward on this rank (other ranks may have heads).
        return targets[names[0]].new_zeros((), dtype=torch.float32), 0
    if args.hsam_supervision == "soft_prior_aux":
        from copy import copy
        stage_args = copy(args)
        stage_args.hsam_supervision = "upsample_logits"
        losses = []
        for i, scale in enumerate((16, 8, 4)):
            loss, _ = coarse_segmentation_loss(
                {name: value[i] for name, value in logits.items()},
                targets, names, keep, stage_args)
            losses.append(loss)
            if diagnostics is not None:
                diagnostics["soft_prior_p{}_loss".format(scale)] = float(loss.detach())
        return torch.stack(losses).mean(), 0
    if args.hsam_supervision == "m2f_hard":
        from copy import copy
        stage_args = copy(args)
        stage_args.hsam_supervision = "upsample_logits"
        losses = [coarse_segmentation_loss(
            {name: value[:, i:i+1] for name, value in logits.items()},
            targets, names, keep, stage_args)[0] for i in range(4)]
        return torch.stack(losses).sum(), 0
    reference = next(iter(logits.values()))
    predictions, labels = {}, {}
    lost = 0
    for index, name in enumerate(names):
        prediction = logits.get(name, torch.zeros_like(reference))
        target = targets[name].to(prediction.device).float()
        if target.ndim == 5:
            if prediction.ndim != 5 or prediction.shape[2] != target.shape[2]:
                raise ValueError("3D coarse supervision must preserve slice count")
            if args.segmentation_unit != "slice":
                raise ValueError("3D coarse supervision requires slice-wise loss")
        if args.hsam_supervision == "downsample_gt":
            resized = F.interpolate(target, size=prediction.shape[2:], mode="nearest")
            if target.ndim == 5:
                positive = target.flatten(3).any(-1).squeeze(1)
                remaining = resized.flatten(3).any(-1).squeeze(1)
                lost += int((positive & ~remaining & keep[:, index, None]).sum().item())
            else:
                lost += int((target.flatten(1).any(1) & ~resized.flatten(1).any(1)
                             & keep[:, index]).sum().item())
            target = resized
        elif args.hsam_supervision == "upsample_logits":
            prediction = F.interpolate(prediction.float(), size=target.shape[2:],
                                       mode="trilinear" if target.ndim == 5 else "bilinear", align_corners=False)
        else:
            raise ValueError("invalid coarse supervision mode")
        predictions[name], labels[name] = prediction, target
    fields = ("focal_alpha", "focal_gamma", "tversky_alpha_fp", "tversky_beta_fn",
              "tversky_eps", "balanced_focal_weight", "hard_negative_ratio",
              "negative_slice_weight", "segmentation_unit", "background_reduction")
    kwargs = {name: getattr(args, name) for name in fields}
    loss, _ = base_segmentation_loss(predictions, labels, names, keep,
        background_weight=args.lambda_bg_seg, loss_type=args.seg_loss_type, **kwargs)
    return loss, lost
