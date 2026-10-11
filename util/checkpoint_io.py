"""Atomic checkpoint writes with optional bounded retention."""
import os
import tempfile
from pathlib import Path


def atomic_checkpoint_save(state, path, save, keep=0):
    path = Path(path)
    keep = int(keep)
    if keep < 0:
        raise ValueError('checkpoint retention must be nonnegative')
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    if keep:
        candidates = [p for p in path.parent.glob('checkpoint-*.pth')
                      if p.stem[len('checkpoint-'):].isdigit() and not p.is_symlink()]
        candidates.sort(key=lambda p: int(p.stem[len('checkpoint-'):]))
        for old in candidates[:-keep]:
            old.unlink()
