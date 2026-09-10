import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from util.distributed_metrics import reduce_metrics


def worker(rank, path):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        values = {'a': 1., 'b': 3.} if rank == 0 else {'b': 5., 'a': 3.}
        assert reduce_metrics(values, 'cpu') == {'a': 2., 'b': 4.}
        bad = {'a': 1.} if rank == 0 else {'b': 1., 'a': 2.}
        try:
            reduce_metrics(bad, 'cpu')
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
