"""4+4 mechanics, deliberately separate from the offline/Common8 trainer."""
import torch
from torch import nn

from losses_orgslot import base_segmentation_loss

OLD = ("spleen", "right_kidney", "left_kidney", "gallbladder")
NEW = ("esophagus", "pancreas", "liver", "stomach")
BASE = ("background",) + OLD


def configure_stage2(model, background_policy):
    if background_policy not in ("frozen", "composition", "separation"):
        raise ValueError("invalid background policy")
    if tuple(model.slot_names) != BASE + NEW:
        raise ValueError("stage2 requires ordered background + old4 + new4")
    model.requires_grad_(False)
    for name in NEW:
        slot = model.slot_bank.get_slot(name)
        slot.requires_grad_(True)
        slot.calibration_scale.requires_grad_(False)
        slot.calibration_bias.requires_grad_(False)
    if background_policy != "frozen":
        slot = model.slot_bank.get_slot("background")
        for name in ("collector", "tg_encoder", "token_norm", "aher"):
            getattr(slot, name).requires_grad_(True)
    # In particular, the unused background head/calibration remain frozen.


def teacher_regions(new_masks, old_probabilities, low=0.1, high=0.9):
    """Disjoint pseudo partition; only new GT is accepted at this boundary."""
    if set(new_masks) != set(NEW) or set(old_probabilities) != set(OLD):
        raise ValueError("stage2 accepts only new4 GT and old4 teacher probabilities")
    if not 0 <= low < high <= 1:
        raise ValueError("expected 0 <= low < high <= 1")
    new_union = torch.stack([new_masks[n].bool() for n in NEW]).any(0)
    probs = torch.stack([old_probabilities[n].detach().float() for n in OLD])
    if not torch.isfinite(probs).all() or torch.any((probs < 0) | (probs > 1)):
        raise ValueError("invalid teacher probabilities")
    confident = probs >= high
    unique = confident.sum(0).eq(1) & ~new_union
    regions = {n: new_masks[n].bool() for n in NEW}
    regions.update({n: confident[i] & unique for i, n in enumerate(OLD)})
    regions["background"] = (probs < low).all(0) & ~new_union
    valid = torch.stack(list(regions.values())).any(0)
    return regions, valid


def region_mse(prediction, target, region):
    error = (prediction.float() - target.float()).square()
    weights = region.expand_as(error).float()
    return (error * weights).sum() / weights.sum().clamp_min(1)


def composition_target(image, regions, keep, names):
    retained = torch.zeros_like(regions["background"])
    for i, name in enumerate(names):
        retained |= regions[name] & keep[:, i].bool().view(-1, 1, 1, 1)
    return image * retained.to(image.dtype)


class Incremental44Objective(nn.Module):
    """One DDP forward includes both recon decodes; teacher is external/frozen."""
    def __init__(self, student, args):
        super().__init__()
        self.student = student
        self.args = args

    def train(self, mode=True):
        super().train(False)
        # Keep old/shared paths in eval mode as well as freezing parameters.
        if mode:
            for name in NEW:
                self.student.slot_bank.get_slot(name).train(True)
            if self.args.background_policy != "frozen":
                self.student.slot_bank.get_slot("background").train(True)
        return self

    def forward(self, image, masks, old_probs, keep):
        args, net = self.args, self.student
        regions, valid = teacher_regions(masks, old_probs, args.teacher_low, args.teacher_high)
        # Compute all canvases once, then use the actual TGR mask for composition.
        all_keep = torch.ones_like(keep)
        head_keep = torch.zeros_like(keep)
        head_keep[:, 5:] = True
        out = net(image, all_keep, head_compute_mask=head_keep,
                  return_diagnostics=True, decode_reconstruction=False)
        canvases = out["slot_canvases"]
        reconstruction = net.forward_decoder(net.fuse_canvases(canvases, keep, net.slot_names))
        target = composition_target(image, regions, keep, net.slot_names)
        recon = region_mse(reconstruction, target, valid)
        seg_keep = keep[:, 5:] if args.seg_supervision == "retained" else all_keep[:, 5:]
        seg, per_slot = base_segmentation_loss(
            {n: out["slot_logits"][n] for n in NEW}, masks, NEW, seg_keep,
            background_weight=0, loss_type="small_organ",
            focal_alpha=args.focal_alpha, focal_gamma=args.focal_gamma,
            tversky_alpha_fp=args.tversky_alpha_fp, tversky_beta_fn=args.tversky_beta_fn,
            tversky_eps=args.tversky_eps, balanced_focal_weight=args.balanced_focal_weight,
            hard_negative_ratio=args.hard_negative_ratio,
            negative_slice_weight=args.negative_slice_weight,
            background_reduction=args.background_reduction)
        separation = recon * 0
        if args.background_policy == "separation":
            bg = net.forward_decoder(canvases["background"])
            foreground = valid & ~regions["background"]
            separation = region_mse(bg, torch.zeros_like(image), foreground)
            separation = separation + args.background_preserve_weight * region_mse(
                bg, image, regions["background"])
        loss = recon + args.lambda_seg * seg + args.lambda_background * separation
        stats = {"loss": loss.detach(), "reconstruction": recon.detach(),
                 "segmentation": seg.detach(), "background_separation": separation.detach(),
                 "valid_fraction": valid.float().mean(),
                 "background_fraction": regions["background"].float().mean(),
                 "ignored_fraction": (~valid).float().mean()}
        for i, name in enumerate(NEW):
            stats["seg_" + name] = per_slot[name].detach()
            stats["supervised_" + name] = seg_keep[:, i].float().sum()
            stats["gt_fraction_" + name] = masks[name].float().mean()
        for name in OLD:
            stats["teacher_fraction_" + name] = regions[name].float().mean()
        return loss, stats
