"""Dense-negative loss checks, including the Arm F 3D model path."""
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from losses_orgslot import base_segmentation_loss, small_organ_segmentation_loss, tversky_loss
from test_orgslot_model import tiny_model


class DenseBackgroundTests(unittest.TestCase):
    def test_empty_formula_and_all_pixel_gradients(self):
        x = torch.linspace(-4, 4, 100).reshape(1, 1, 10, 10).requires_grad_()
        y = torch.zeros_like(x)
        with patch('torch.topk', side_effect=AssertionError('top-k must not run')):
            loss, _ = small_organ_segmentation_loss(x, y)
        torch.testing.assert_close(loss, 0.1 * F.binary_cross_entropy_with_logits(x, y))
        loss.backward()
        torch.testing.assert_close(x.grad, 0.1 * x.detach().sigmoid() / x.numel())
        self.assertTrue((x.grad > 0).all())

    def test_positive_formula_and_dense_background_gradients(self):
        x = torch.linspace(-4, 4, 100).reshape(1, 1, 10, 10).requires_grad_()
        y = torch.zeros_like(x)
        y[:, :, 4:6, 4:6] = 1
        p = x.sigmoid()
        ce = F.binary_cross_entropy_with_logits(x, y, reduction='none')
        pos = y.bool()
        pf = (0.75 * (1-p[pos]).square() * ce[pos]).mean()
        nf = (0.25 * p[~pos].square() * ce[~pos]).mean()
        with patch('torch.topk', side_effect=AssertionError('top-k must not run')):
            loss, details = small_organ_segmentation_loss(x, y)
        torch.testing.assert_close(loss, tversky_loss(x, y) + 0.5 * (pf + nf))
        grad, = torch.autograd.grad(details['hard_negative_focal_loss'], x)
        self.assertTrue((grad[~pos] > 0).all())
        self.assertTrue((grad[pos] == 0).all())

    def test_arm_f_3d_mixed_and_empty_queries_backward(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = tiny_model('3D', slot_head_type='arm_f_attention', slot_head_channels=16)
        images = torch.randn(1, 3, 4, 32, 32)
        keep = torch.ones(1, 3, dtype=torch.bool)
        output = model(images, keep, decode_reconstruction=False)
        logits = output['calibrated_logits']
        masks = {name: torch.zeros_like(value) for name, value in logits.items()}
        masks['kidney'][:, :, 1, 12:20, 12:20] = 1
        diagnostics = {}
        loss, _ = base_segmentation_loss(logits, masks, model.slot_names, keep,
            background_weight=0, loss_type='small_organ', segmentation_unit='slice',
            diagnostics=diagnostics)
        self.assertEqual(diagnostics['kidney']['positive_samples'].item(), 1)
        self.assertEqual(diagnostics['kidney']['negative_samples'].item(), 3)
        self.assertEqual(diagnostics['spleen']['negative_samples'].item(), 4)
        (0.01 * loss).backward()
        gradients = [p.grad for p in model.pixel_query_decoder.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)


if __name__ == '__main__':
    unittest.main()
