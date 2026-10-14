import unittest
from types import SimpleNamespace

import torch

from CNNBackbone import CNN_BACKBONES, CNNTokenEncoder2D
from test_orgslot_model import tiny_model
from hsam_supervision import coarse_segmentation_loss


class CNNBackboneTests(unittest.TestCase):
    def test_encoder_skips_are_used_once_and_receive_head_gradients(self):
        for kind in CNN_BACKBONES:
            with self.subTest(backbone=kind):
                model = tiny_model(backbone=kind, slot_head_type="arm_f_sam_tail",
                                   slot_head_channels=16)
                self.assertIsNone(model.pixel_query_decoder.pixels.spatial_stem)
                captured = {}
                calls = {2: 0, 3: 0}
                def capture(index):
                    def hook(module, inputs, output):
                        calls[index] += 1
                        output.retain_grad()
                        captured[index] = output
                    return hook
                handles = [model.patch_embed.stages[i].register_forward_hook(capture(i))
                           for i in (2, 3)]
                out = model(torch.randn(1, 3, 32, 32), decode_reconstruction=False)
                sum(v.square().mean() for v in out['slot_logits'].values()).backward()
                for handle in handles:
                    handle.remove()
                self.assertEqual(calls, {2: 1, 3: 1})
                self.assertEqual(tuple(captured[2].shape), (1, 128, 8, 8))
                self.assertEqual(tuple(captured[3].shape), (1, 256, 4, 4))
                for feature in captured.values():
                    self.assertTrue(torch.isfinite(feature.grad).all())
                    self.assertGreater(feature.grad.abs().sum().item(), 0)
                for projection in (model.pixel_query_decoder.pixels.p4_proj,
                                   model.pixel_query_decoder.pixels.p8_proj):
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_sam_tail_supervision_and_roundtrip(self):
        for kind in CNN_BACKBONES:
            for mode in ("soft_prior_aux", "downsample_gt"):
                with self.subTest(backbone=kind, supervision=mode):
                    kwargs = dict(backbone=kind, slot_head_type="arm_f_sam_tail",
                                  slot_head_channels=16)
                    model = tiny_model(**kwargs)
                    model.pixel_query_decoder.configure_mask_supervision(mode)
                    images = torch.randn(1, 3, 32, 32)
                    output = model(images)
                    names = model.slot_names
                    targets = {n: torch.zeros(1, 1, 32, 32) for n in names}
                    targets[names[1]][:, :, 8:16, 8:16] = 1
                    args = SimpleNamespace(hsam_supervision=mode, lambda_bg_seg=0.,
                        seg_loss_type="small_organ", focal_alpha=.75, focal_gamma=2.,
                        tversky_alpha_fp=.3, tversky_beta_fn=.7, tversky_eps=1e-6,
                        balanced_focal_weight=.5, hard_negative_ratio=.02,
                        negative_slice_weight=.1, segmentation_unit="slice",
                        background_reduction="topk")
                    coarse, _ = coarse_segmentation_loss(output["coarse_logits"],
                        targets, names, torch.ones(1, len(names), dtype=torch.bool), args)
                    loss = coarse + output["reconstruction"].square().mean()
                    loss += sum(v.square().mean() for v in output["calibrated_logits"].values())
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    for name, p in model.patch_embed.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(model.patch_embed.proj.weight.grad.abs().sum().item(), 0)
                    clone = tiny_model(**kwargs)
                    clone.pixel_query_decoder.configure_mask_supervision(mode)
                    clone.load_state_dict(model.state_dict(), strict=True)
                    model.eval()
                    clone.eval()
                    with torch.no_grad():
                        a = model(images, decode_reconstruction=False)["calibrated_logits"]
                        b = clone(images, decode_reconstruction=False)["calibrated_logits"]
                    for name in names:
                        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)

    def test_encoder_grids_and_gradients(self):
        for kind in CNN_BACKBONES:
            with self.subTest(kind=kind):
                encoder = CNNTokenEncoder2D(kind, 32, 3, 32)
                z = encoder(torch.randn(1, 3, 32, 32))
                self.assertEqual(tuple(z.shape), (1, 4, 32))
                z.square().mean().backward()
                for name, p in encoder.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_joint_forward_backward_and_strict_roundtrip(self):
        for kind in CNN_BACKBONES:
            with self.subTest(kind=kind):
                kwargs = dict(slot_head_type="arm_e_multiscale_query",
                              query_refinement="cross_attn", slot_head_channels=16,
                              backbone=kind)
                model = tiny_model(**kwargs)
                image = torch.randn(1, 3, 32, 32)
                output = model(image)
                self.assertEqual(output["reconstruction"].shape, image.shape)
                loss = output["reconstruction"].square().mean()
                for logits in output["calibrated_logits"].values():
                    self.assertEqual(tuple(logits.shape), (1, 1, 32, 32))
                    loss = loss + logits.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.patch_embed.proj.weight.grad)
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                clone = tiny_model(**kwargs)
                clone.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                clone.eval()
                with torch.no_grad():
                    a = model(image, decode_reconstruction=False)["calibrated_logits"]
                    b = clone(image, decode_reconstruction=False)["calibrated_logits"]
                for organ in a:
                    torch.testing.assert_close(a[organ], b[organ], rtol=0, atol=0)

    def test_downstream_same_seed_unchanged(self):
        kwargs = dict(slot_head_type="arm_e_multiscale_query",
                      query_refinement="cross_attn", slot_head_channels=16)
        torch.manual_seed(42)
        baseline = tiny_model(**kwargs).state_dict()
        for kind in CNN_BACKBONES:
            torch.manual_seed(42)
            candidate = tiny_model(backbone=kind, cnn_skip_source="stem", **kwargs).state_dict()
            for name, value in baseline.items():
                if not name.startswith(("patch_embed.", "blocks1.", "cls_token")):
                    torch.testing.assert_close(value, candidate[name], rtol=0, atol=0)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            CNNTokenEncoder2D("nnunet_resenc_s16", 33, 3, 32)
        with self.assertRaises(ValueError):
            tiny_model(backbone="resnet18_s16_gn")
        for kind in CNN_BACKBONES:
            with self.assertRaises(ValueError):
                tiny_model(backbone=kind)
            with self.assertRaises(ValueError):
                tiny_model(dataset_type="3D", backbone=kind)


if __name__ == "__main__":
    unittest.main()
