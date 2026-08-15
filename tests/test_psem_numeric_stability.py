import contextlib
import io
import unittest

import torch

from VQ.lpips import normalize_tensor
from engine_pretrain_psem_triplet_loss3 import (
    _check_losses_finite,
    _nonfinite_component_names,
)


class PsemNumericStabilityTests(unittest.TestCase):
    def test_lpips_zero_feature_normalization_is_finite(self):
        features = torch.zeros(2, 4, 3, 3, dtype=torch.float16)
        normalized = normalize_tensor(features)
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized, torch.zeros_like(normalized)))

    def test_nonfinite_component_names_identifies_nan_and_inf(self):
        components = {
            "finite": torch.tensor(1.0),
            "nan": torch.tensor(float("nan")),
            "inf": torch.tensor(float("inf")),
        }
        self.assertEqual(
            _nonfinite_component_names(components),
            ["nan", "inf"],
        )

    def test_finite_loss_check_passes(self):
        _check_losses_finite(
            {"total_loss": torch.tensor(1.0)},
            torch.device("cpu"),
            epoch=4,
            data_iter_step=7,
            sample_indices=torch.tensor([11, 13]),
        )

    def test_nonfinite_loss_check_reports_context(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaisesRegex(
                FloatingPointError,
                "nonfinite_components=\\['p_loss'\\]",
            ):
                _check_losses_finite(
                    {"p_loss": torch.tensor(float("nan"))},
                    torch.device("cpu"),
                    epoch=428,
                    data_iter_step=1471,
                    sample_indices=torch.tensor([5, 9]),
                )
        message = stderr.getvalue()
        self.assertIn("rank=0 epoch=428 step=1471", message)
        self.assertIn("sample_indices=[5, 9]", message)


if __name__ == "__main__":
    unittest.main()
