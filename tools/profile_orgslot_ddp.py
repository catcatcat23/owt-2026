"""Temporary stage timer for diagnosing OrganSlotBank DDP performance."""

import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def stamp(rank, name, started):
    torch.cuda.synchronize()
    print(
        f"rank={rank} stage={name} elapsed={time.perf_counter() - started:.3f}s",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-slots", type=int, default=9)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    import OWT_models_orgslot

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    started = time.perf_counter()
    slot_names = (
        "background",
        "spleen",
        "right_kidney",
        "left_kidney",
        "gallbladder",
        "esophagus",
        "pancreas",
        "liver",
        "stomach",
    )[: args.num_slots]
    slot_specs = [
        {"name": name, "raw_class_id": index}
        for index, name in enumerate(slot_names)
    ]
    model_args = SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=20,
        organ_token_total=20 * args.num_slots,
        fix_frame=0,
        temp_stride=0,
        loss_version=["L2"],
        text_encoding="None",
    )
    model = OWT_models_orgslot.mae_vit_base_patch16(
        img_size=args.input_size,
        norm_pix_loss=False,
        model_args=model_args,
        slot_specs=slot_specs,
        slot_tg_depth=1,
        fusion_mode="linear_sqrt",
    ).to(device)

    for slot_name in model.slot_names:
        slot = model.slot_bank.get_slot(slot_name)
        for parameter in slot.head.parameters():
            parameter.requires_grad_(False)
        slot.calibration_scale.requires_grad_(False)
        slot.calibration_bias.requires_grad_(False)

    model = DDP(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=args.bucket_cap_mb,
        broadcast_buffers=False,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-4,
    )
    stamp(rank, "setup", started)

    images = torch.randn(
        args.batch_size,
        3,
        args.input_size,
        args.input_size,
        device=device,
    )
    retained = torch.ones(
        args.batch_size,
        args.num_slots,
        dtype=torch.bool,
        device=device,
    )
    target = torch.randn_like(images)

    dist.barrier()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(images, slot_keep_mask=retained)
        loss = torch.nn.functional.mse_loss(outputs["reconstruction"], target)
    stamp(rank, "forward", started)

    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler()
    scaler.scale(loss).backward()
    stamp(rank, "scaled_backward_and_allreduce", started)

    started = time.perf_counter()
    scaler.unscale_(optimizer)
    stamp(rank, "unscale", started)

    started = time.perf_counter()
    parameters = [
        parameter for parameter in model.parameters() if parameter.grad is not None
    ]
    gradient_norm = torch.norm(
        torch.stack(
            [torch.norm(parameter.grad.detach(), 2.0) for parameter in parameters]
        ),
        2.0,
    )
    stamp(rank, "gradient_norm", started)

    started = time.perf_counter()
    scaler.step(optimizer)
    scaler.update()
    stamp(rank, "scaled_optimizer", started)
    if rank == 0:
        print(f"gradient_norm={float(gradient_norm):.6f}", flush=True)

    dist.barrier()
    if rank == 0:
        print("profile_complete", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
