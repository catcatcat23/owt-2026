import unittest
from types import SimpleNamespace
import torch
from ArmFDecoder import probability_guided_attention
from hsam_supervision import coarse_segmentation_loss, stage_weight
from test_orgslot_model import tiny_model


class HSAMTests(unittest.TestCase):
    def test_gate_and_gradient(self):
        torch.set_num_threads(2)
        module = torch.nn.MultiheadAttention(16, 4, batch_first=True)
        q, k = torch.randn(2, 3, 16), torch.randn(2, 7, 16)
        prior = torch.ones(2, 7, 1, requires_grad=True)
        got = probability_guided_attention(module, q, k, k, prior)
        expected = module(q, k, k, need_weights=False)[0]
        torch.testing.assert_close(got, expected, atol=1e-6, rtol=1e-5)
        got.square().sum().backward()
        self.assertGreater(prior.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(prior.grad).all())

    def test_model_both_losses(self):
        torch.set_num_threads(2)
        model = tiny_model(slot_head_type="arm_f_sam_tail", slot_head_channels=16)
        args = SimpleNamespace(lambda_bg_seg=.25, seg_loss_type="small_organ",
            focal_alpha=.75, focal_gamma=2., tversky_alpha_fp=.3,
            tversky_beta_fn=.7, tversky_eps=1e-6, balanced_focal_weight=.5,
            hard_negative_ratio=.02, negative_slice_weight=.1,
            segmentation_unit="slab", background_reduction="topk")
        for mode in ("downsample_gt", "upsample_logits", "m2f_hard"):
            model.zero_grad()
            model.pixel_query_decoder.configure_mask_supervision(mode)
            args.hsam_supervision = mode
            output = model(torch.randn(1, 3, 32, 32), decode_heads=True,
                           decode_reconstruction=False)
            names = list(output["calibrated_logits"])
            targets = {name: torch.zeros(1, 1, 32, 32) for name in names}
            targets[names[1]][:, :, 1, 1] = 1
            keep = torch.ones(1, len(names), dtype=torch.bool)
            loss, lost = coarse_segmentation_loss(output["coarse_logits"], targets,
                                                  names, keep, args)
            self.assertEqual(lost, int(mode == "downsample_gt"))
            loss = loss + sum(x.square().mean() for x in output["calibrated_logits"].values())
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertAlmostEqual(stage_weight(0), .6)
        self.assertGreater(stage_weight(500), stage_weight(0))

    def test_hard_mask_fallback(self):
        from ArmFDecoder import ArmFDecoder
        logits = torch.full((2, 1, 4, 4), -10., requires_grad=True)
        mask = ArmFDecoder.hard_routing(logits, (2, 2), 20, 4)
        self.assertEqual(tuple(mask.shape), (8, 20, 4))
        self.assertFalse(mask.any())
        self.assertFalse(mask.requires_grad)
