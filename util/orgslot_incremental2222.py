"""Four stages with persistent spatial and interaction branches per added pair."""
import copy
import math
from pathlib import Path
import json

import torch
from torch import nn
from util.orgslot_incremental44 import SlotTokenInteraction

GROUPS = (("spleen", "right_kidney"), ("left_kidney", "gallbladder"),
          ("esophagus", "pancreas"), ("liver", "stomach"))
NAMES = sum(GROUPS, ())


def seen(stage):
    if stage not in range(1, 5):
        raise ValueError("Stage must be 1..4")
    return ("background",) + sum(GROUPS[:stage], ())


def install_group(model, stage):
    if stage not in (2, 3, 4):
        raise ValueError("Only incremental stages have private pixels")
    if model.pixel_query_decoder.readout != "sam_tail":
        raise ValueError("2222 requires SAM-tail")
    if not hasattr(model, "stage_pixel_decoders"):
        model.stage_pixel_decoders = nn.ModuleDict()
        model.slot_pixel_groups = {}
    key = "stage" + str(stage)
    if key in model.stage_pixel_decoders:
        raise ValueError("Group already installed")
    pixels = copy.deepcopy(model.pixel_query_decoder.pixels)
    for name in ("query_norm", "query_proj"):
        if hasattr(pixels, name):
            delattr(pixels, name)
    model.stage_pixel_decoders[key] = pixels
    # Same trained base interaction for all later groups; no previous-group overwrite.
    interaction = SlotTokenInteraction(model.pixel_query_decoder)
    for name in GROUPS[stage - 1]:
        model.slot_bank.get_slot(name).interaction = interaction
        model.slot_pixel_groups[name] = key


def restore_groups(model, stage):
    if tuple(model.slot_names) != seen(stage):
        raise ValueError("Checkpoint slot order mismatch")
    for index in range(2, stage + 1):
        install_group(model, index)


def configure_group(model, stage, background_policy="separation"):
    if background_policy != "separation":
        raise ValueError("2222 pins background separation")
    model.requires_grad_(False)
    model.current_pixel_group = "stage" + str(stage)
    model.stage_pixel_decoders[model.current_pixel_group].requires_grad_(True)
    model.incremental_old_names = sum(GROUPS[:stage - 1], ())
    model.incremental_new_names = GROUPS[stage - 1]
    for name in model.incremental_new_names:
        slot = model.slot_bank.get_slot(name)
        slot.requires_grad_(True)
        slot.calibration_scale.requires_grad_(False)
        slot.calibration_bias.requires_grad_(False)
    background = model.slot_bank.get_slot("background")
    for name in ("collector", "tg_encoder", "token_norm", "aher"):
        getattr(background, name).requires_grad_(True)


def validate_complete(checkpoint, stage):
    saved = checkpoint["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    if (saved.get("incremental_protocol") != "2222" or
            saved.get("stage_index") != stage or saved.get("max_optimizer_updates") != 118800):
        raise ValueError("Require preceding 2222 stage with 118800 updates")
    steps = [float(v['step']) for v in checkpoint.get('optimizer', {}).get('state', {}).values() if 'step' in v]
    if not steps or not all(math.isfinite(v) for v in steps) or max(steps) != 118800:
        raise ValueError("Previous stage has not finished")
    return saved


def final_checkpoint(run, stage):
    run = Path(run)
    records = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
    if not records or records[-1].get('optimizer_updates') != 118800:
        raise ValueError("Run has not completed its budget")
    last = records[-1]
    if any(isinstance(v, (float, int)) and not math.isfinite(v) for v in last.values()):
        raise ValueError("Non-finite final log")
    path = run / ('checkpoint-%s.pth' % last['epoch'])
    checkpoint = torch.load(path, map_location='cpu')
    validate_complete(checkpoint, stage)
    if checkpoint['epoch'] != last['epoch']:
        raise ValueError("Checkpoint/log mismatch")
    for name, value in checkpoint['model'].items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError("Non-finite tensor: " + name)
    if stage > 1:
        audit = json.loads((run / 'freeze_audit.json').read_text())
        if audit.get('epoch') != last['epoch'] or not audit.get('match'):
            raise ValueError("Missing final frozen parameter audit")
    return path.resolve()
