"""Validate latest resume or completed stage checkpoint; no silent fresh restart."""
import argparse
import json
import math
import sys
from pathlib import Path
import torch
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.orgslot_incremental2222 import final_checkpoint


def resume_checkpoint(run, stage):
    paths = list(Path(run).glob('checkpoint-*.pth'))
    if not paths:
        if (Path(run)/'log.txt').exists():
            raise ValueError('Logs exist without recovery checkpoint; refuse fresh restart')
        return ''
    path = max(paths, key=lambda p: int(p.stem.split('-')[-1]))
    ckpt = torch.load(path, map_location='cpu')
    args = ckpt['args'] if isinstance(ckpt['args'], dict) else vars(ckpt['args'])
    if args.get('incremental_protocol') != '2222' or args.get('stage_index') != stage:
        raise ValueError('Resume protocol mismatch')
    steps = [float(v['step']) for v in ckpt.get('optimizer', {}).get('state', {}).values() if 'step' in v]
    if not steps or not all(math.isfinite(v) for v in steps) or not 0 < max(steps) <= 118800:
        raise ValueError('Invalid optimizer budget')
    for name, value in ckpt['model'].items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError('Non-finite tensor: '+name)
    return str(path.resolve())


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('stage', type=int, choices=range(1,5))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    print(resume_checkpoint(a.run, a.stage) if a.resume else final_checkpoint(a.run, a.stage))
