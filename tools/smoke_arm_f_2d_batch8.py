"""Full-resolution, all-organ Arm F 2D batch8 DDP memory/gradient preflight.

Synthetic inputs; covers model + reconstruction MSE + segmentation, not LPIPS
or real-data loading. Formal training remains the end-to-end validation.
"""
import json
import os
import time

import torch
import torch.distributed as dist

from losses_orgslot import base_segmentation_loss
from tools.eval_common8_orgslot_reconstruction_threshold import build_model
from tools.eval_common8_orgslot_reconstruction_threshold import parse_class_configuration


def main():
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    specs, _ = parse_class_configuration('configs/orgslot/common8_offline.json')
    model, report = build_model('orgslot', os.environ['PREFLIGHT_CHECKPOINT'], specs, 448, 'linear_sqrt')
    model = model.to(device).train()
    names = model.slot_names
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-5)
    keep = torch.ones(8, len(names), device=device, dtype=torch.bool)
    active = keep.clone()
    active[:, names.index('background')] = False
    images = torch.rand(8, 3, 448, 448, device=device)
    masks = {name: torch.zeros(8, 1, 448, 448, device=device) for name in names}
    for name in names:
        masks[name][:4, :, 180:240, 180:240] = 1
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step in range(12):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(images, slot_keep_mask=keep, head_compute_mask=active)
            loss, _ = base_segmentation_loss(output['calibrated_logits'], masks, names, keep,
                background_weight=0, loss_type='small_organ', segmentation_unit='slice',
                background_reduction=os.environ.get('BACKGROUND_REDUCTION', 'mean'))
            loss = (output['reconstruction'] - images).square().mean() + 0.01 * loss
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite loss')
        (loss / 6).backward()
        if (step + 1) % 6 == 0:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite gradient')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(json.dumps({'rank': dist.get_rank(), 'micro_batch': 8, 'accum_iter': 6,
        'effective_batch': 8 * 6 * dist.get_world_size(), 'steps': 12,
        'seconds': time.monotonic() - started, 'peak_gib': torch.cuda.max_memory_allocated() / 2**30,
        'checkpoint_epoch': report['epoch'], 'passed': True}), flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
