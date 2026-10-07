"""4+4 mechanics, deliberately separate from the offline/Common8 trainer."""
import copy
import torch
from torch import nn

from losses_orgslot import base_segmentation_loss

OLD = ("spleen", "right_kidney", "left_kidney", "gallbladder")
NEW = ("esophagus", "pancreas", "liver", "stomach")
BASE = ("background",) + OLD

READOUT_MODULES = ("query_norm", "query_proj", "query_cross_attention")


def validate_incremental_architecture(args):
    """Reject accidental mixing of the E and SAM-tail protocols."""
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    signature = (get("slot_head_type"), get("query_refinement"), get("hsam_supervision", "none"))
    if signature not in (("arm_e_multiscale_query", "cross_attn", "none"),
                         ("arm_f_sam_tail", "none", "soft_prior_aux")):
        raise ValueError("Unsupported incremental architecture: " + str(signature))
    return signature[0]


class SlotTokenInteraction(nn.Module):
    """No pixel decoder, tail or extra prior network; copies trained Stage1 blocks."""
    def __init__(self, decoder):
        super().__init__()
        self.input_proj = copy.deepcopy(decoder.input_proj)
        self.blocks = copy.deepcopy(decoder.blocks)


def install_slot_interactions(model):
    from ArmFDecoder import ArmFDecoder
    decoder = model.pixel_query_decoder
    if not isinstance(decoder, ArmFDecoder) or decoder.readout != "sam_tail" or \
            len(decoder.grid_size) != 2 or getattr(decoder, "hsam_supervision", "none") != "soft_prior_aux":
        raise ValueError("Slot interactions require 2D SAM-tail soft_prior_aux")
    if any(hasattr(model.slot_bank.get_slot(n), "interaction") for n in model.slot_names):
        raise ValueError("Slot interactions already installed")
    for name in model.slot_names:
        model.slot_bank.get_slot(name).interaction = SlotTokenInteraction(decoder)


class StageQueryReadout(nn.Module):
    """Stage-private query-only copy; shares the unchanged P4 and mask formula."""
    def __init__(self, decoder):
        super().__init__()
        from OrganSlotEmbed import MultiScalePixelQueryDecoder2D
        if not isinstance(decoder, MultiScalePixelQueryDecoder2D) or decoder.query_refinement != "cross_attn":
            raise ValueError("Stage readout requires 2D Arm E cross-attention")
        self.channels = decoder.channels
        for name in READOUT_MODULES:
            setattr(self, name, copy.deepcopy(getattr(decoder, name)))

    def forward_mask(self, pixel_features, tokens, slot_identity, output_size):
        from OrganSlotEmbed import MultiScalePixelQueryDecoder2D
        query = self.query_proj(self.query_norm(tokens).mean(dim=1)) + slot_identity.unsqueeze(0)
        query, _ = self.query_cross_attention(query, pixel_features)
        return MultiScalePixelQueryDecoder2D.forward_mask_from_query(self, pixel_features, query, output_size)


def install_stage_readout(model):
    if hasattr(model, "new_stage_readout"):
        raise ValueError("Stage readout already installed")
    model.new_stage_readout = StageQueryReadout(model.pixel_query_decoder)
    model.new_stage_readout_slots = NEW


def validate_readout_adapter(args):
    get = args.get if isinstance(args, dict) else lambda k, d=None: getattr(args, k, d)
    mode = get("stage2_readout_adapter", "none")
    if mode not in ("none", "residual16"):
        raise ValueError("Unknown Stage2 readout adapter")
    if mode != "none" and (
            get("incremental_stage") != "stage2" or
            get("stage2_shared_segmentation") != "slot_private" or
            get("slot_head_type") != "arm_f_sam_tail" or
            get("hsam_supervision", "none") != "soft_prior_aux"):
        raise ValueError("Readout adapter requires Stage2 SAM soft_prior_aux + slot_private")
    return mode


def install_stage_mask_adapter(model):
    """One stage-shared residual mask-weight adapter; old organs bypass it."""
    from ArmFDecoder import ArmFDecoder
    decoder = model.pixel_query_decoder
    if (not isinstance(decoder, ArmFDecoder) or decoder.readout != "sam_tail" or
            len(decoder.grid_size) != 2 or
            getattr(decoder, "hsam_supervision", "none") != "soft_prior_aux" or
            tuple(model.slot_names) != BASE + NEW):
        raise ValueError("Readout adapter requires expanded 2D SAM soft_prior_aux")
    if hasattr(model, "new_stage_mask_adapter"):
        raise ValueError("Readout adapter already installed")
    with torch.random.fork_rng(devices=[]):
        adapter = nn.Sequential(nn.Linear(decoder.channels, 16), nn.GELU(),
                                nn.Linear(16, decoder.channels))
        nn.init.zeros_(adapter[-1].weight)
        nn.init.zeros_(adapter[-1].bias)
    model.new_stage_mask_adapter = adapter
    model.new_stage_mask_adapter_slots = NEW


def configure_stage2(model, background_policy, shared_segmentation="frozen",
                     readout_adapter="none"):
    if shared_segmentation not in ("frozen", "train", "query_shared", "query_split", "slot_private"):
        raise ValueError("invalid shared segmentation policy")
    if background_policy not in ("frozen", "composition", "separation"):
        raise ValueError("invalid background policy")
    if tuple(model.slot_names) != BASE + NEW:
        raise ValueError("stage2 requires ordered background + old4 + new4")
    validate_readout_adapter({"stage2_readout_adapter": readout_adapter,
        "incremental_stage": "stage2", "stage2_shared_segmentation": shared_segmentation,
        "slot_head_type": model.slot_bank.get_slot(NEW[0]).head_type,
        "hsam_supervision": getattr(model.pixel_query_decoder, "hsam_supervision", "none")})
    if shared_segmentation == "slot_private":
        if not all(hasattr(model.slot_bank.get_slot(n), "interaction") for n in model.slot_names):
            install_slot_interactions(model)
    elif any(hasattr(model.slot_bank.get_slot(n), "interaction") for n in model.slot_names):
        raise ValueError("Private interactions require slot_private policy")
    if shared_segmentation == "query_split" and not hasattr(model, "new_stage_readout"):
        install_stage_readout(model)
    if shared_segmentation != "query_split" and hasattr(model, "new_stage_readout"):
        raise ValueError("Cannot reuse split-readout model for a different policy")
    if readout_adapter == "residual16" and not hasattr(model, "new_stage_mask_adapter"):
        install_stage_mask_adapter(model)
    if readout_adapter == "none" and hasattr(model, "new_stage_mask_adapter"):
        raise ValueError("Cannot disable an installed readout adapter")
    model.requires_grad_(False)
    if readout_adapter == "residual16":
        model.new_stage_mask_adapter.requires_grad_(True)
    if shared_segmentation == "train":
        if model.pixel_query_decoder is None:
            raise ValueError("shared segmentation training requires pixel_query_decoder")
        model.pixel_query_decoder.requires_grad_(True)
    elif shared_segmentation == "query_shared":
        for name in READOUT_MODULES:
            getattr(model.pixel_query_decoder, name).requires_grad_(True)
    elif shared_segmentation == "query_split":
        model.new_stage_readout.requires_grad_(True)
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
            if hasattr(self.student, "new_stage_mask_adapter"):
                self.student.new_stage_mask_adapter.train(True)
            if getattr(self.args, "stage2_shared_segmentation", "frozen") == "train":
                self.student.pixel_query_decoder.train(True)
            elif getattr(self.args, "stage2_shared_segmentation", "frozen") == "query_shared":
                for name in READOUT_MODULES:
                    getattr(self.student.pixel_query_decoder, name).train(True)
            elif getattr(self.args, "stage2_shared_segmentation", "frozen") == "query_split":
                self.student.new_stage_readout.train(True)
            for name in NEW:
                self.student.slot_bank.get_slot(name).train(True)
            if self.args.background_policy != "frozen":
                background = self.student.slot_bank.get_slot("background")
                background.train(True)
                if hasattr(background, "interaction"):
                    background.interaction.eval()
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
        aux = seg * 0
        aux_stats = {}
        if getattr(args, "hsam_supervision", "none") == "soft_prior_aux":
            from hsam_supervision import coarse_segmentation_loss
            aux, _ = coarse_segmentation_loss(
                {n: out["coarse_logits"][n] for n in NEW}, masks, NEW,
                seg_keep, args, diagnostics=aux_stats)
        loss = recon + args.lambda_seg * (seg + 0.25 * aux) + args.lambda_background * separation
        stats = {"loss": loss.detach(), "reconstruction": recon.detach(),
                 "segmentation": seg.detach(), "background_separation": separation.detach(),
                 "valid_fraction": valid.float().mean(),
                 "background_fraction": regions["background"].float().mean(),
                 "ignored_fraction": (~valid).float().mean()}
        if getattr(args, "hsam_supervision", "none") == "soft_prior_aux":
            stats.update({"coarse_segmentation": aux.detach(),
                          "weighted_coarse_segmentation": args.lambda_seg * 0.25 * aux.detach(),
                          **aux_stats})
        for i, name in enumerate(NEW):
            stats["seg_" + name] = per_slot[name].detach()
            stats["supervised_" + name] = seg_keep[:, i].float().sum()
            stats["gt_fraction_" + name] = masks[name].float().mean()
        for name in OLD:
            stats["teacher_fraction_" + name] = regions[name].float().mean()
        return loss, stats
