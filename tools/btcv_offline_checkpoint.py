"""Validate the most advanced checkpoint of one BTCV Offline run chain."""
import argparse
from pathlib import Path
import torch

p = argparse.ArgumentParser(__doc__)
p.add_argument('run', type=Path)
p.add_argument('--resume', action='store_true')
p.add_argument('--final', action='store_true')
a = p.parse_args()
paths = list(a.run.glob('train_*/checkpoint-*.pth'))
if not paths:
    if a.final or list(a.run.glob('train_*/log.txt')):
        raise RuntimeError('Missing recovery checkpoint; no silent restart')
    print('')
else:
    path = max(paths, key=lambda v: int(v.stem.split('-')[-1]))
    c = torch.load(path, map_location='cpu')
    cfg = c['args'] if isinstance(c['args'], dict) else vars(c['args'])
    assert not cfg.get('full_field', False), 'Rejected superseded full-field checkpoint'
    assert cfg.get('hsam_supervision') == 'soft_prior_aux'
    assert cfg.get('slot_head_type') == 'arm_f_sam_tail'
    assert cfg.get('max_optimizer_updates') == 118800
    assert 'BTCV_Training' in cfg['data_path']
    steps = [float(s['step']) for s in c['optimizer']['state'].values() if 'step' in s]
    assert steps and 0 < max(steps) <= 118800
    if a.final:
        assert max(steps) == 118800, 'Training budget not complete'
    for name, tensor in c['model'].items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise RuntimeError('Non-finite checkpoint tensor: ' + name)
    print(path.resolve())
