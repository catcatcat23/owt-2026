import pickle
import tempfile
import unittest
from pathlib import Path
from util.checkpoint_io import atomic_checkpoint_save


class CheckpointIOTest(unittest.TestCase):
    def test_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            for epoch in [0, 10, 20]:
                atomic_checkpoint_save(epoch, p / f'checkpoint-{epoch}.pth', pickle.dump, keep=2)
            self.assertEqual(sorted(x.name for x in p.iterdir()), ['checkpoint-10.pth', 'checkpoint-20.pth'])

    def test_failed_save_keeps_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'checkpoint-0.pth'
            atomic_checkpoint_save(1, p, pickle.dump)
            def fail(state, stream):
                stream.write(b'partial')
                raise OSError('quota')
            with self.assertRaises(OSError):
                atomic_checkpoint_save(2, p, fail, keep=1)
            self.assertEqual(pickle.loads(p.read_bytes()), 1)
            self.assertEqual(len(list(p.parent.iterdir())), 1)


if __name__ == '__main__':
    unittest.main()
