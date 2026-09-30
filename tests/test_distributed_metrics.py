import os
import tempfile
import unittest
from types import SimpleNamespace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from util.distributed_metrics import reduce_metrics
from hsam_supervision import coarse_segmentation_loss


def worker(rank, path):
    backend = os.environ.get('METRICS_TEST_BACKEND', 'gloo')
    device = 'cuda' if backend == 'nccl' else 'cpu'
    if backend == 'nccl':
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method='file://' + path, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        values = {'a': 1., 'b': 3.} if rank == 0 else {'b': 5., 'a': 3.}
        assert reduce_metrics(values, device) == {'a': 2., 'b': 4.}
        args = SimpleNamespace(hsam_supervision='soft_prior_aux', lambda_bg_seg=0.,
            seg_loss_type='small_organ', focal_alpha=.75, focal_gamma=2.,
            tversky_alpha_fp=.3, tversky_beta_fn=.7, tversky_eps=1e-6,
            balanced_focal_weight=.5, hard_negative_ratio=.02,
            negative_slice_weight=.1, segmentation_unit='slab', background_reduction='topk')
        names = ['background', 'organ']
        targets = {n: torch.zeros(1, 1, 8, 8) for n in names}
        targets['organ'][:, :, 2:4, 2:4] = 1
        logits = {} if rank == 0 else {'organ': [torch.zeros(1, 1, s, s, requires_grad=True) for s in (2, 4, 8)]}
        keep = torch.tensor([[True, rank == 1]])
        diagnostics = {}
        loss, _ = coarse_segmentation_loss(logits, targets, names, keep, args, diagnostics)
        assert len(diagnostics) == 3
        if rank == 0:
            assert loss.item() == 0 and all(v == 0 for v in diagnostics.values())
        else:
            loss.backward()
            assert all(torch.isfinite(t.grad).all() for t in logits['organ'])
        averaged = reduce_metrics(diagnostics, device)
        assert all(v > 0 for v in averaged.values())
        bad = {'a': 1.} if rank == 0 else {'b': 1., 'a': 2.}
        try:
            reduce_metrics(bad, device)
        except RuntimeError as error:
            assert 'schema' in str(error)
        else:
            raise AssertionError('Different schemas did not fail together')
        dist.barrier()
    finally:
        dist.destroy_process_group()


class MetricsTests(unittest.TestCase):
    def test_two_ranks_order_and_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(worker, args=(os.path.join(directory, 'rendezvous'),), nprocs=2, join=True)
