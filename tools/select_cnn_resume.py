"""Select the newest finite CNN training checkpoint after its writer has stopped."""
import sys
from pathlib import Path

import torch


def select(directory, backbone, supervision):
    paths = [p for p in Path(directory).glob('checkpoint-*.pth')
             if p.stem.split('-')[-1].isdigit()]
    for path in sorted(paths, key=lambda p: int(p.stem.split('-')[-1]), reverse=True):
        try:
            state = torch.load(path, map_location='cpu')
            args = state['args']
            args = args if isinstance(args, dict) else vars(args)
            expected = dict(backbone=backbone, hsam_supervision=supervision,
                            cnn_skip_source='encoder', batch_size=16, accum_iter=3,
                            max_optimizer_updates=118800)
            if any(args.get(k) != v for k, v in expected.items()):
                raise ValueError('Checkpoint protocol mismatch: ' + str(path))
            assert state['epoch'] == int(path.stem.split('-')[-1])
            assert state['optimizer']['state']
            tensors = list(state['model'].values())
            tensors += [v for entry in state['optimizer']['state'].values()
                        for v in entry.values() if torch.is_tensor(v)]
            assert all(not t.is_floating_point() or torch.isfinite(t).all() for t in tensors)
            steps = [float(v['step']) for v in state['optimizer']['state'].values() if 'step' in v]
            assert steps and 0 < max(steps) <= 118800
            if state['epoch'] == 802:
                assert max(steps) == 118800
            return path.resolve()
        except ValueError:
            raise
        except Exception as exc:
            print(f'Skipping incomplete checkpoint {path}: {exc}', file=sys.stderr)
    raise RuntimeError('No valid CNN resume checkpoint in ' + directory)


if __name__ == '__main__':
    print(select(*sys.argv[1:]))
