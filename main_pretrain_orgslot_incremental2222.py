"""SAM soft-prior+aux, four 2-class stages with persistent group-private pixels."""
import argparse
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.label_visibility import load_visibility_config, assert_case_splits_disjoint
from util.mae_transfer import load_mae_transfer_checkpoint
from util.checkpoint_orgslot import hash_frozen_parameters, compare_parameter_hashes
from util.orgslot_incremental44 import Incremental44Objective
from main_pretrain_orgslot_common8_a100 import (
    get_args_parser, _build_dataset, _model_args, _seed_worker, _sha256, _write_provenance,
)
from engine_pretrain_orgslot_common8_a100 import (
    train_one_epoch, _autocast_context, _check_finite, _adjust_learning_rate,
    _slot_keep_mask, _force_focus_slots,
)
from OWT_models_orgslot import mae_vit_base_patch16
from VQ.lpips import LPIPS
import timm.optim.optim_factory as optim_factory


from main_pretrain_orgslot_incremental44 import parser as base_parser, train_stage2
from util.orgslot_incremental2222 import GROUPS, NAMES, seen, restore_groups, install_group, configure_group, validate_complete


def parser():
    p = base_parser()
    for action in p._actions:
        if action.dest == "incremental_stage":
            action.required = False
    p.add_argument("--stage_index", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--previous_checkpoint", default="")
    p.set_defaults(visibility_config="configs/orgslot/common8_incremental2222.json",
                   slot_head_type="arm_f_sam_tail", query_refinement="none",
                   hsam_supervision="soft_prior_aux", incremental_protocol="2222")
    return p


def main(args):
    args.incremental_stage = "stage" + str(args.stage_index)
    if (args.slot_head_type != "arm_f_sam_tail" or args.query_refinement != "none" or
            args.hsam_supervision != "soft_prior_aux" or args.dimension != "2D"):
        raise ValueError("2222 pins 2D SAM soft_prior_aux")
    if (args.stage2_tail != "shared" or args.stage2_pixel_decoder != "shared" or
            args.stage2_readout_adapter != "none" or args.stage2_shared_segmentation != "frozen"):
        raise ValueError("4+4 policy flags cannot be used in 2222")
    if args.max_optimizer_updates != 118800 or args.background_policy != "separation":
        raise ValueError("2222 pins 118800 updates per stage and background separation")
    if args.stage1_checkpoint:
        raise ValueError("Use previous_checkpoint from this 2222 chain")
    if args.init_checkpoint or args.mae_init_scope != "encoder":
        raise ValueError("Only MAE encoder initialization is permitted")
    if args.seg_loss_type != "small_organ" or args.lambda_bg_seg != 0 or args.training_scope != "reconstruction":
        raise ValueError("Keep the specified small-organ joint-training objective; no background head supervision")
    if args.max_steps_per_epoch or args.max_train_samples or args.max_val_samples:
        raise ValueError("Use unit tests for tiny/debug data; formal manifests must not be truncated")
    if args.lr is None or args.lr <= 0 or args.clip_grad <= 0:
        raise ValueError("Require positive explicit LR and gradient clipping")
    if args.lambda_collector_attention or args.positive_roi_loss_weight:
        raise ValueError("Additional offline ablations are not part of incremental44")
    if args.resume and args.start_epoch:
        raise ValueError("Resume derives start_epoch from checkpoint")
    if not args.resume and args.start_epoch:
        raise ValueError("start_epoch without resume is invalid")
    if not 0 <= args.teacher_low < args.teacher_high <= 1:
        raise ValueError("Invalid teacher thresholds")
    if args.background_lr_scale <= 0 or min(args.lambda_background, args.background_preserve_weight) < 0:
        raise ValueError("Invalid background coefficients")
    misc.init_distributed_mode(args)
    device = torch.device(args.device)
    torch.manual_seed(args.seed + misc.get_rank())
    random.seed(args.seed + misc.get_rank())
    np.random.seed(args.seed + misc.get_rank())
    args.fusion_reference_count = args.fusion_reference_count or 9
    args.focus_class_ids = ",".join(str(NAMES.index(n) + 1) for n in GROUPS[args.stage_index - 1] if n in ("gallbladder", "esophagus", "pancreas"))
    if not args.focus_class_ids:
        args.organ_roi_aug = False
    if args.batch_size * args.accum_iter * misc.get_world_size() != 192:
        raise ValueError("Require effective batch 192")
    args.train_manifest_sha256 = _sha256(args.data_path)
    args.validation_manifest_sha256 = _sha256(args.val_data_path)
    args.test_manifest_sha256 = _sha256(args.test_data_path)
    preprocess = json.loads(Path(args.preprocess_summary).read_text())["config"]
    if not np.allclose(preprocess["spacing_mm"], args.expected_spacing, rtol=0, atol=1e-6) or preprocess.get("offline_spatial_matrix") != "native_after_resampling":
        raise ValueError("Require native 0.7/0.7/2 resampling")
    if args.input_size != 448 or args.global_crop_size != 448 or args.roi_crop_size != 384:
        raise ValueError("Require 448 input/global crop and 384 ROI")
    if args.organ_roi_aug:
        roi = json.loads(Path(args.roi_index).read_text())
        if roi["manifest_sha256"] != args.train_manifest_sha256:
            raise ValueError("ROI index/manifest mismatch")
    classes, stages = load_visibility_config(args.visibility_config)
    for index in range(1, 5):
        expected = seen(1) if index == 1 else GROUPS[index - 1]
        if stages["stage" + str(index)].visible_slots != expected:
            raise ValueError("Unexpected 2222 class order")
    raw, dataset = _build_dataset(args, args.data_path, classes, stages[args.incremental_stage], True, None)
    val_raw, _ = _build_dataset(args, args.val_data_path, classes, stages[args.incremental_stage], False, None)
    test_raw, _ = _build_dataset(args, args.test_data_path, classes, stages[args.incremental_stage], False, None)
    assert_case_splits_disjoint({"train": raw.case_ids, "validation": val_raw.case_ids, "test": test_raw.case_ids})
    split_path = Path(__file__).resolve().parent / "configs/orgslot/word_official96_seed42.json"
    expected_splits = json.loads(split_path.read_text())["splits"]
    for split_name, case_ids in (("Training", raw.case_ids), ("Validation", val_raw.case_ids), ("Test", test_raw.case_ids)):
        expected_ids = {item["case_id"] for item in expected_splits[split_name]}
        if set(case_ids) != expected_ids:
            raise ValueError("Require the pinned official-pool split: " + split_name)
    args.split_plan_sha256 = _sha256(split_path)
    sampler = DistributedSampler(dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.pin_mem, drop_last=True, worker_init_fn=_seed_worker)
    per_epoch = len(loader) // args.accum_iter
    if not per_epoch or args.max_optimizer_updates <= 0:
        raise ValueError("Insufficient batches/update budget")
    build_stage = max(1, args.stage_index - 1)
    specs = [{"name": n, "raw_class_id": i} for i, n in enumerate(seen(build_stage))]
    net = mae_vit_base_patch16(img_size=args.input_size, norm_pix_loss=False,
        model_args=_model_args(args, len(specs)), slot_specs=specs, slot_tg_depth=args.slot_tg_depth,
        fusion_mode=args.fusion_mode, fusion_reference_count=args.fusion_reference_count,
        slot_head_type=args.slot_head_type, slot_head_channels=args.slot_head_channels,
        pixel_pe=args.pixel_pe, query_refinement=args.query_refinement)
    teacher = None
    mae_report = None
    if args.incremental_stage == "stage1":
        if args.previous_checkpoint or (not args.resume and not args.mae_init_checkpoint):
            raise ValueError("Stage1 requires fresh MAE encoder initialization (or stage1 resume)")
        if not args.resume:
            mae_report = load_mae_transfer_checkpoint(net, args.mae_init_checkpoint, "encoder")
            args.mae_checkpoint_sha256 = mae_report["sha256"]
        if args.lambda_lpips:
            net.perceptual_loss = LPIPS(vgg_pretrained=False).eval()
            net.perceptual_loss.load_state_dict(torch.load(args.lpips_state, map_location="cpu"), strict=True)
            net.perceptual_loss.requires_grad_(False)
        args.roi_class_weights = [0.0, 1.0, 1.0]
        model = net
    else:
        if not args.previous_checkpoint or args.mae_init_checkpoint:
            raise ValueError("Later stages require the preceding 2222 checkpoint only")
        args.stage1_checkpoint_sha256 = _sha256(args.previous_checkpoint)
        checkpoint = torch.load(args.previous_checkpoint, map_location="cpu")
        saved = validate_complete(checkpoint, args.stage_index - 1)
        for key in ("train_manifest_sha256", "validation_manifest_sha256", "test_manifest_sha256",
                    "input_size", "token_factor", "slot_tg_depth", "slot_head_type",
                    "slot_head_channels", "query_refinement", "pixel_pe", "fusion_mode",
                    "fusion_reference_count", "seed", "hsam_supervision"):
            if saved.get(key) != getattr(args, key):
                raise ValueError("Previous stage protocol mismatch: " + key)
        restore_groups(net, args.stage_index - 1)
        net.load_state_dict({k: v for k, v in checkpoint["model"].items() if not k.startswith("perceptual_loss.")}, strict=True)
        teacher = copy.deepcopy(net).requires_grad_(False).eval().to(device)
        for name in GROUPS[args.stage_index - 1]:
            net.append_slot(name, NAMES.index(name) + 1, init_from="background")
        install_group(net, args.stage_index)
        configure_group(net, args.stage_index, args.background_policy)
        model = Incremental44Objective(net, args)
    model.to(device)
    groups = optim_factory.add_weight_decay(net, args.weight_decay)
    # Preserve decay/no-decay groups, splitting out a lower-LR background group.
    bg_ids = {id(p) for p in net.slot_bank.get_slot("background").parameters()}
    param_groups = []
    for group in groups:
        for background in (False, True):
            params = [p for p in group["params"] if (id(p) in bg_ids) == background]
            if params:
                scale = args.background_lr_scale if background and teacher is not None else 1.0
                param_groups.append({**group, "params": params, "lr_scale": scale})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    scaler = NativeScaler(enabled=args.amp_dtype == "fp16")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        saved = checkpoint["args"] if isinstance(checkpoint["args"], dict) else vars(checkpoint["args"])
        if saved.get("incremental_protocol") != "2222" or saved.get("stage_index") != args.stage_index:
            raise ValueError("Resume stage mismatch")
        if saved.get("stage2_tail", "shared") != args.stage2_tail:
            raise ValueError("Resume protocol mismatch: stage2_tail")
        if saved.get("stage2_pixel_decoder", "shared") != args.stage2_pixel_decoder:
            raise ValueError("Resume protocol mismatch: stage2_pixel_decoder")
        if saved.get("stage2_readout_adapter", "none") != args.stage2_readout_adapter:
            raise ValueError("Resume protocol mismatch: stage2_readout_adapter")
        if saved.get("stage2_shared_segmentation", "frozen") != args.stage2_shared_segmentation:
            raise ValueError("Resume protocol mismatch: stage2_shared_segmentation")
        if saved.get("hsam_supervision", "none") != args.hsam_supervision:
            raise ValueError("Resume protocol mismatch: hsam_supervision")
        for key in ("incremental_stage", "slot_head_type", "background_policy", "train_manifest_sha256", "validation_manifest_sha256", "test_manifest_sha256", "stage1_checkpoint_sha256", "batch_size", "accum_iter", "max_optimizer_updates", "warmup_updates", "lr", "weight_decay", "lambda_seg", "lambda_background", "background_preserve_weight", "background_lr_scale", "teacher_low", "teacher_high", "seg_supervision", "fusion_mode", "fusion_reference_count", "query_refinement", "pixel_pe", "seed", "organ_roi_probability", "organ_roi_aug", "focus_class_ids"):
            if saved.get(key) != getattr(args, key, None):
                raise ValueError("Resume protocol mismatch: " + key)
        misc.load_model(args, net, optimizer, scaler)
        if "mae_checkpoint_sha256" in saved:
            args.mae_checkpoint_sha256 = saved["mae_checkpoint_sha256"]
    output = Path(args.output_dir)
    if not args.resume and output.exists() and any(output.glob("checkpoint-*.pth")):
        raise ValueError("Refusing to overwrite an existing training run")
    _write_provenance(args, output)
    if misc.is_main_process():
        source_root = Path(__file__).resolve().parent
        (output / "incremental_sources.json").write_text(json.dumps({p: _sha256(source_root / p) for p in
            ("main_pretrain_orgslot_incremental2222.py", "util/orgslot_incremental2222.py", "util/orgslot_incremental44.py", "main_pretrain_orgslot_incremental44.py")}, indent=2))
        (output / "slot_metadata.json").write_text(json.dumps(net.slot_bank.metadata(), indent=2))
        (output / "trainable_parameters.json").write_text(json.dumps([n for n, p in net.named_parameters() if p.requires_grad], indent=2))
        if mae_report is not None:
            (output / "mae_initial_checkpoint_load.json").write_text(json.dumps(mae_report, indent=2))
    frozen_hashes = hash_frozen_parameters(net) if teacher is not None else None
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    print("Incremental2222", args.incremental_stage, "slots", net.slot_names,
          "effective_batch", args.batch_size * args.accum_iter * misc.get_world_size(), flush=True)
    for epoch in range(args.start_epoch, math.ceil(args.max_optimizer_updates / per_epoch)):
        sampler.set_epoch(epoch)
        if teacher is None:
            stats = train_one_epoch(model, loader, optimizer, device, epoch, scaler, args=args)
        else:
            stats = train_stage2(model, teacher, loader, optimizer, device, epoch, scaler, args)
        final = stats["optimizer_updates"] >= args.max_optimizer_updates
        if not args.no_save and (epoch % args.save_freq == 0 or final):
            if frozen_hashes is not None:
                audit = compare_parameter_hashes(frozen_hashes, hash_frozen_parameters(net))
                if not audit["match"]:
                    raise RuntimeError("Frozen parameters changed: " + str(audit))
                if misc.is_main_process():
                    (output / "freeze_audit.json").write_text(json.dumps({"epoch": epoch, **audit}, indent=2))
            misc.save_model(args, epoch, model, net, optimizer, scaler)
        if misc.is_main_process():
            with (output / "log.txt").open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, **stats}) + "\n")
        if final:
            break


if __name__ == "__main__":
    main(parser().parse_args())
