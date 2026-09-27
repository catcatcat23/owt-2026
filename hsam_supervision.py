"""Controlled H-SAM-inspired coarse supervision; original losses unchanged."""
import torch
from torch.nn import functional as F
from losses_orgslot import base_segmentation_loss


def stage_weight(epoch):
    return 0.6 ** (0.990 ** epoch)


def coarse_segmentation_loss(logits, targets, names, keep, args):
    if not logits:
        raise ValueError("H-SAM requires active coarse predictions")
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
        if args.hsam_supervision == "downsample_gt":
            resized = F.interpolate(target, size=prediction.shape[-2:], mode="nearest")
            lost += int((target.flatten(1).any(1) & ~resized.flatten(1).any(1)
                         & keep[:, index]).sum().item())
            target = resized
        elif args.hsam_supervision == "upsample_logits":
            prediction = F.interpolate(prediction.float(), size=target.shape[-2:],
                                       mode="bilinear", align_corners=False)
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
