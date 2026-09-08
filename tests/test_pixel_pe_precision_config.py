"""Guard the submission precision, not merely the model's AMP support."""
import pathlib
import unittest


class PrecisionConfigTests(unittest.TestCase):
    def test_pe_job_explicitly_selects_bf16(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        script = (root / "slurm/orgslot/train/pixel_pe_2d.sbatch").read_text()
        self.assertIn("export AMP_DTYPE=bf16 CLIP_GRAD=1.0 FINITE_CHECK_INTERVAL=1", script)
        self.assertIn("torch.cuda.is_bf16_supported()", script)
        self.assertIn("export MAX_UPDATES=2 WARMUP_UPDATES=0 SAVE_FREQ=1", script)
        self.assertIn("SAVE_FREQ=25", script)

    def test_fail_fast_is_not_disabled(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        self.assertIn("error_if_nonfinite=True", (root / "util/misc.py").read_text())
