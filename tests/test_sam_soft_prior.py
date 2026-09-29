import unittest
from types import SimpleNamespace
import torch
from test_orgslot_model import tiny_model
from hsam_supervision import coarse_segmentation_loss


class SoftPriorTests(unittest.TestCase):
    def test_modes_and_aux_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        model = tiny_model(slot_head_type="arm_f_sam_tail", slot_head_channels=16).eval()
        image = torch.randn(1, 3, 32, 32)
        model.pixel_query_decoder.configure_mask_supervision("soft_prior")
        a = model(image)
        model.pixel_query_decoder.configure_mask_supervision("soft_prior_aux")
        b = model(image)
        for name in a['calibrated_logits']:
            torch.testing.assert_close(a['calibrated_logits'][name], b['calibrated_logits'][name], rtol=0, atol=0)
        args = SimpleNamespace(hsam_supervision='soft_prior_aux', lambda_bg_seg=0., seg_loss_type='small_organ', focal_alpha=.75, focal_gamma=2., tversky_alpha_fp=.3, tversky_beta_fn=.7, tversky_eps=1e-6, balanced_focal_weight=.5, hard_negative_ratio=.02, negative_slice_weight=.1, segmentation_unit='slice', background_reduction='topk')
        names = model.slot_names
        targets = {n: torch.zeros(1, 1, 32, 32) for n in names}
        targets[names[1]][:, :, 10:15, 10:15] = 1
        keep = torch.ones(1, len(names), dtype=torch.bool)
        diagnostics = {}
        loss, _ = coarse_segmentation_loss(b['coarse_logits'], targets, names, keep, args, diagnostics)
        self.assertEqual(len(diagnostics), 3)
        prior = b['coarse_logits'][names[1]][0]
        prior.retain_grad()
        loss.backward()
        self.assertGreater(prior.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        keep[:] = False
        keep[:, 0] = True
        out = model(image, slot_keep_mask=keep, head_compute_mask=torch.zeros_like(keep))
        loss, _ = coarse_segmentation_loss(out['coarse_logits'], targets, names, keep, args)
        (out['reconstruction'].square().mean() + loss).backward()
        self.assertEqual(loss.item(), 0.)
