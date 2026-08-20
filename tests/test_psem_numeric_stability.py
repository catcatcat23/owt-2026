import contextlib
import io
import unittest

import torch

from VQ.lpips import normalize_tensor
from engine_pretrain_psem_triplet_loss3 import (
    _autocast_context,
    _check_losses_finite,
    _check_model_parameters_finite,
    _check_training_tensors_finite,
    _nonfinite_component_names,
)
from util.misc import NativeScalerWithGradNormCount


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

    def test_cpu_autocast_policy_is_noop(self):
        value = torch.ones(2, dtype=torch.float32)
        with _autocast_context(torch.device("cpu"), "bf16"):
            result = value * 2
        self.assertEqual(result.dtype, torch.float32)

    def test_nonfinite_input_check_reports_sample_indices(self):
        with self.assertRaisesRegex(
            FloatingPointError, "nonfinite_tensors=\['image'\]"
        ):
            _check_training_tensors_finite(
                {"image": torch.tensor([float("inf")])},
                torch.device("cpu"),
                epoch=500,
                data_iter_step=1296,
                sample_indices=torch.tensor([50370]),
            )

    def test_nonfinite_parameter_check_names_parameter(self):
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "weight"):
            _check_model_parameters_finite(
                model,
                torch.device("cpu"),
                epoch=500,
                data_iter_step=1296,
                sample_indices=torch.tensor([50370]),
            )

    def test_disabled_scaler_clips_finite_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = NativeScalerWithGradNormCount(enabled=False)
        norm = scaler(
            parameter.square().sum(),
            optimizer,
            clip_grad=1.0,
            parameters=[parameter],
            update_grad=True,
        )
        self.assertTrue(torch.isfinite(norm))
        self.assertLess(float(parameter), 2.0)
        self.assertEqual(scaler.state_dict(), {})

    def test_gradient_clip_rejects_nonfinite_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scaler = NativeScalerWithGradNormCount(enabled=False)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            scaler(
                parameter * torch.tensor(float("inf")),
                optimizer,
                clip_grad=1.0,
                parameters=[parameter],
                update_grad=True,
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
