import tempfile
import unittest
from pathlib import Path

import torch

from CNNBackbone import CNN_BACKBONES
from models_cnn_mim import CNNMaskedImageModel, load_cnn_encoder_checkpoint
from test_orgslot_model import tiny_model


class CNNMIMTests(unittest.TestCase):
    def test_pretraining_transfer_and_rejections(self):
        for kind in CNN_BACKBONES:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                pre = CNNMaskedImageModel(kind, 32)
                images = torch.rand(2, 3, 32, 32)
                seen = []
                hook = pre.encoder.register_forward_pre_hook(lambda module, args: seen.append(args[0].detach()))
                loss, prediction, mask = pre(images)
                hook.remove()
                self.assertEqual(prediction.shape, images.shape)
                self.assertTrue(torch.equal(mask.sum(1), torch.tensor([3., 3.])))
                pixel_mask = mask.reshape(2, 1, 2, 2).repeat_interleave(16, 2).repeat_interleave(16, 3)
                self.assertEqual(float((seen[0] * pixel_mask).abs().sum()), 0.)
                loss.backward()
                for name, parameter in pre.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                target = tiny_model(backbone=kind, slot_head_type="arm_f_sam_tail", slot_head_channels=16)
                before = {k: v.clone() for k, v in target.state_dict().items()
                          if not k.startswith(("patch_embed.stem.", "patch_embed.stages."))}
                payload = dict(model=pre.state_dict(), backbone=kind,
                               pretraining_complete=True, optimizer_updates=1)
                path = Path(directory) / "final.pth"
                torch.save(payload, path)
                report = load_cnn_encoder_checkpoint(target, path)
                self.assertFalse(report["decoder_loaded"])
                for key, value in before.items():
                    torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
                for key, value in pre.encoder.state_dict().items():
                    torch.testing.assert_close(value, target.patch_embed.state_dict()[key], rtol=0, atol=0)
                payload["pretraining_complete"] = False
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    load_cnn_encoder_checkpoint(target, path)
                payload["pretraining_complete"] = True
                payload["backbone"] = "vit"
                torch.save(payload, path)
                with self.assertRaises(ValueError):
                    load_cnn_encoder_checkpoint(target, path)
