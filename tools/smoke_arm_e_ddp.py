"""Two-rank Arm E DDP check with deliberately different retained organs."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from OWT_models_orgslot import OrganSlotMaskedAutoencoderViT


def _model_args():
    return SimpleNamespace(
        LA=True,
        arch_version="v1",
        dataset_type="2D",
        token_factor=2,
        organ_token_total=6,
        fix_frame=4,
        temp_stride=1,
        loss_version="L2",
        text_encoding="None",
    )


def _tiny_arm_e():
    specs = [
        {"name": "background", "raw_class_id": 0},
        {"name": "kidney", "raw_class_id": 1},
        {"name": "spleen", "raw_class_id": 2},
    ]
    return OrganSlotMaskedAutoencoderViT(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=nn.LayerNorm,
        model_args=_model_args(),
        slot_specs=specs,
        slot_tg_depth=1,
        fusion_mode="post_layernorm",
        slot_head_type="arm_e_multiscale_query",
        slot_head_channels=16,
    )


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("Arm E DDP smoke requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = _tiny_arm_e().to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    keep = torch.tensor(
        [[1, 1, 0] if rank == 0 else [1, 0, 1]],
        dtype=torch.bool,
        device=device,
    )
    head_compute = keep.clone()
    head_compute[:, 0] = False
    for step in range(3):
        generator = torch.Generator(device=device).manual_seed(100 + rank + step)
        images = torch.rand(
            1, 3, 32, 32, generator=generator, device=device
        )
        output = model(
            images,
            slot_keep_mask=keep,
            head_compute_mask=head_compute,
            decode_reconstruction=True,
        )
        loss = output["reconstruction"].square().mean()
        loss = loss + sum(
            logits.square().mean()
            for logits in output["calibrated_logits"].values()
        )
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite Arm E DDP smoke loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        packed = torch.tensor(
            [float(loss.detach()), float(keep.sum())],
            device=device,
        )
        dist.all_reduce(packed)
        if rank == 0:
            print(
                "step={} reduced_loss={:.6f} retained_sum={:.0f}".format(
                    step, packed[0].item(), packed[1].item()
                ),
                flush=True,
            )
    dist.barrier()
    if rank == 0:
        print("ARM_E_DDP_DIFFERENT_ORGANS_OK", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
