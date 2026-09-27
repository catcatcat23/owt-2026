"""Paired pre/post-reverse linear probes; frozen original Arm F, one organ/run."""
import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset


class FeatureCapture:
    """Capture raw P4 before reverse and residual+FFN P4 before classifier."""
    def __init__(self, decoder):
        if decoder.readout != "attention" or len(decoder.grid_size) != 2:
            raise ValueError("Only original 2D arm_f_attention checkpoints supported")
        self.values = {}
        self.handles = [
            decoder.pixel_norm.register_forward_pre_hook(self.hook("pre")),
            decoder.classifier.register_forward_pre_hook(self.hook("post")),
        ]

    def hook(self, name):
        def capture(module, inputs):
            if name in self.values:
                raise RuntimeError("Multiple organ heads executed; select exactly one")
            self.values[name] = inputs[0].detach()
        return capture

    def clear(self):
        self.values.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


def probe_loss(logits, target):
    """Equal positive/negative slice-group weight; identical for both probes."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    prob = logits.sigmoid()
    dice = 1 - (2 * (prob * target).sum((1, 2, 3)) + 1e-6) / (
        prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1e-6)
    positive = target.sum((1, 2, 3)) > 0
    terms = []
    if positive.any():
        terms.append((bce + dice)[positive].mean())
    if (~positive).any():
        terms.append(bce[~positive].mean())
    return torch.stack(terms).mean()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-csv", required=True, help="Original TRAIN manifest only")
    parser.add_argument("--class-map", default="configs/orgslot/common8_offline.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-id", type=int, default=6, choices=range(1, 9))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from tools.eval_common8_orgslot_reconstruction_threshold import (
        build_dataset, build_model, parse_class_configuration, sha256)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    specs, _ = parse_class_configuration(Path(args.class_map))
    model, report = build_model("orgslot", Path(args.checkpoint), specs, 448, "linear_sqrt")
    model.requires_grad_(False).eval().to(args.device)
    capture = FeatureCapture(model.pixel_query_decoder)
    slot_index = next(i for i, s in enumerate(specs) if s['raw_class_id'] == args.class_id)
    slot_name = specs[slot_index]['name']
    dataset = build_dataset(Path(args.train_csv), 448, 448)
    import csv
    with open(args.train_csv) as stream:
        records = list(csv.DictReader(stream))
    ids = [Path(r['mask_pth']).parent.name for r in records]
    cases = sorted(set(ids))
    if len(cases) != 96:
        raise ValueError("Expected 96 original training cases; refuse test manifest")
    random.Random(args.seed).shuffle(cases)
    held = set(cases[:20])
    groups = {"probe_train": [i for i, c in enumerate(ids) if c not in held],
              "probe_holdout": [i for i, c in enumerate(ids) if c in held]}
    provenance = dict(vars(args), checkpoint_sha256=sha256(Path(args.checkpoint)),
        manifest_sha256=sha256(Path(args.train_csv)), load_report=report,
        holdout_cases=sorted(held), train_cases=sorted(set(cases)-held),
        warning="Backbone saw probe-holdout during original training; NOT independent validation.",
        protocol="center448, no ROI/augmentation, FP32, fixed0.5 raw, BCE+positive Dice probe loss; no checkpoint selection")
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {k: DataLoader(Subset(dataset, v), batch_size=args.batch_size,
        shuffle=k=='probe_train', num_workers=args.workers, generator=generator)
        for k,v in groups.items()}
    heads = nn.ModuleDict({'pre': nn.Linear(model.pixel_query_decoder.channels, 1)}).to(args.device)
    heads['post'] = copy.deepcopy(heads['pre'])
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0)
    updates = 0
    for epoch in range(args.epochs):
        for split, loader in loaders.items():
            counts = {}; loss_sum = 0.; batches = 0
            for batch in loader:
                images = batch['image'].to(args.device)
                target = (batch['full_label'] == args.class_id).float().to(args.device)
                keep = torch.zeros(images.shape[0], len(specs), dtype=torch.bool, device=args.device)
                keep[:, slot_index] = True
                capture.clear()
                with torch.no_grad():
                    original = model(images, slot_keep_mask=keep, head_compute_mask=keep,
                                     decode_reconstruction=False)
                if set(capture.values) != {'pre', 'post'}:
                    raise RuntimeError('Missing pre/post capture')
                logits = {}
                with torch.set_grad_enabled(split == 'probe_train'):
                    for name, feature in capture.values.items():
                        side = int(feature.shape[1] ** .5)
                        if side * side != feature.shape[1]:
                            raise ValueError('Expected square 2D P4')
                        value = heads[name](feature.float()).transpose(1, 2).reshape(-1,1,side,side)
                        logits[name] = F.interpolate(value, size=(448,448), mode='bilinear', align_corners=False)
                    loss = sum(probe_loss(v, target) for v in logits.values())
                    if not torch.isfinite(loss):
                        raise RuntimeError('Nonfinite probe loss')
                    if split == 'probe_train':
                        optimizer.zero_grad(); loss.backward()
                        if any(p.grad is not None for p in model.parameters()):
                            raise RuntimeError('Frozen backbone received gradients')
                        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step(); updates += 1
                loss_sum += loss.item(); batches += 1
                logits['original'] = original['calibrated_logits'][slot_name]
                for name, value in logits.items():
                    pred = value.detach() > 0
                    for i, case in enumerate(batch['case_id']):
                        t = target[i].bool(); p = pred[i]
                        entry = counts.setdefault((name,case), [0,0,0])
                        for j, n in enumerate([(p&t).sum(),p.sum(),t.sum()]):
                            entry[j] += int(n)
            rows = []
            for (name,case),(tp,p,t) in sorted(counts.items()):
                rows.append(dict(model=name,case=case,tp=tp,pred=p,gt=t,
                    dice=2*tp/(p+t) if p+t else 1., precision=tp/p if p else None,
                    recall=tp/t if t else None, volume_ratio=p/t if t else None))
            result = dict(epoch=epoch,split=split,updates=updates,loss=loss_sum/max(batches,1),
                peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else 0,
                metrics=rows, train_metrics_online=split=='probe_train')
            (out/f'{split}_epoch{epoch:03d}.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
        torch.save(dict(heads=heads.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates), out/'probe_last.pth')
    capture.close()


if __name__ == '__main__':
    main()
