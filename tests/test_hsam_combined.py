import unittest
from types import SimpleNamespace
import torch
from test_orgslot_model import tiny_model
from hsam_supervision import combined_coarse_losses, coarse_segmentation_loss, stage_weight


class CombinedTests(unittest.TestCase):
    def test_combination_gradients_and_regression(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = tiny_model(slot_head_type='arm_f_sam_tail', slot_head_channels=16).eval()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        image = torch.randn(1, 3, 32, 32)
        names = model.slot_names
        keep = torch.ones(1, len(names), dtype=torch.bool)
        targets = {n: torch.zeros(1, 1, 32, 32) for n in names}
        targets[names[1]][:, :, 8:16, 8:16] = 1
        args = SimpleNamespace(lambda_bg_seg=0., seg_loss_type='small_organ',
            focal_alpha=.75, focal_gamma=2., tversky_alpha_fp=.3, tversky_beta_fn=.7,
            tversky_eps=1e-6, balanced_focal_weight=.5, hard_negative_ratio=.02,
            negative_slice_weight=.1, segmentation_unit='slice', background_reduction='topk')
        model.pixel_query_decoder.configure_mask_supervision('downsample_gt')
        baseline = model(image)
        outputs = []
        for mode in ('downsample_gt_soft_prior', 'downsample_gt_soft_prior_aux'):
            model.zero_grad()
            args.hsam_supervision = mode
            model.pixel_query_decoder.configure_mask_supervision(mode)
            output = model(image)
            outputs.append(output)
            coarse = output['coarse_logits'][names[1]]
            self.assertEqual(len(coarse), 4)
            self.assertEqual([v.shape[-1] for v in coarse], [8, 2, 4, 8])
            for v in coarse:
                v.retain_grad()
            # Exported coarse tensors are expanded views; gate-gradient behavior
            # is separately covered by test_hsam_supervision.test_gate_and_gradient.
            final = sum(v.square().mean() for v in output['calibrated_logits'].values())
            metrics = {}
            c, lost, aux = combined_coarse_losses(output['coarse_logits'], targets, names, keep, args, metrics)
            self.assertEqual(lost, 0)
            self.assertEqual(len(metrics), 3 if mode.endswith('_aux') else 0)
            w = stage_weight(100)
            loss = .01 * (w * final + (1-w) * c + .25 * aux)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            for v in coarse[1:]:
                if mode.endswith('_aux'):
                    self.assertGreater(v.grad.abs().sum().item(), 0.)
                else:
                    self.assertIsNone(v.grad)  # Detached routing, no extra prior supervision.
            empty_keep = torch.zeros_like(keep)
            empty_keep[:, 0] = True
            empty_metrics = {}
            c, lost, aux = combined_coarse_losses({}, targets, names, empty_keep, args, empty_metrics)
            self.assertEqual(c.item() + aux.item(), 0.)
            self.assertEqual(set(metrics), set(empty_metrics))  # DDP schemas stay identical.
        for name in names:
            torch.testing.assert_close(outputs[0]['calibrated_logits'][name], outputs[1]['calibrated_logits'][name], rtol=0, atol=0)
        model.load_state_dict(state, strict=True)  # No new learned parameters.
        model.pixel_query_decoder.configure_mask_supervision('downsample_gt')
        again = model(image)
        for name in names:
            torch.testing.assert_close(baseline['calibrated_logits'][name], again['calibrated_logits'][name], rtol=0, atol=0)
