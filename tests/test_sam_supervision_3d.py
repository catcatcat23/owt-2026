import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from test_orgslot_model import tiny_model
from hsam_supervision import coarse_segmentation_loss
from tools import eval_common8_orgslot_heads_3d as evaluator


def loss_args(mode):
    return SimpleNamespace(hsam_supervision=mode, lambda_bg_seg=0.,
        seg_loss_type='small_organ', focal_alpha=.75, focal_gamma=2.,
        tversky_alpha_fp=.3, tversky_beta_fn=.7, tversky_eps=1e-6,
        balanced_focal_weight=.5, hard_negative_ratio=.02,
        negative_slice_weight=.1, segmentation_unit='slice', background_reduction='topk')


class SAMSupervision3DTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_full_model_backward_and_checkpoint(self):
        for mode in ('soft_prior_aux', 'downsample_gt'):
            with self.subTest(mode=mode):
                model = tiny_model('3D', slot_head_type='arm_f_sam_tail', slot_head_channels=16)
                model.pixel_query_decoder.configure_mask_supervision(mode)
                image = torch.randn(1, 3, 4, 32, 32)
                out = model(image)
                names = model.slot_names
                targets = {n: torch.zeros(1, 1, 4, 32, 32) for n in names}
                # One positive slice and three empty slices.
                targets[names[1]][:, :, 1, 8:16, 8:16] = 1
                keep = torch.ones(1, len(names), dtype=torch.bool)
                aux, _ = coarse_segmentation_loss(out['coarse_logits'], targets, names, keep, loss_args(mode))
                self.assertEqual(out['calibrated_logits'][names[1]].shape, (1, 1, 4, 32, 32))
                values = out['coarse_logits'][names[1]]
                for value in values if isinstance(values, list) else [values]:
                    self.assertEqual(value.shape[2], 4)
                loss = aux + sum(v.square().mean() for v in out['calibrated_logits'].values())
                loss = loss + out['reconstruction'].square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(model.pixel_query_decoder.sam_tail.mask_mlp[-1].weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.pixel_query_decoder.input_proj.weight.grad.abs().sum().item(), 0)
                model.eval()
                checkpoint = {'model': model.state_dict(), 'epoch': 0, 'args': {
                    'dimension': '3D', 'fix_frame': 4, 'temp_stride': 1, 'input_size': 32,
                    'fusion_mode': 'post_layernorm', 'token_factor': 2, 'slot_tg_depth': 1,
                    'slot_head_type': 'arm_f_sam_tail', 'slot_head_channels': 16,
                    'hsam_supervision': mode}}
                def build(**kwargs):
                    from test_orgslot_model import OrganSlotMaskedAutoencoderViT
                    kwargs.update(patch_size=16, in_chans=3, embed_dim=32, depth=2,
                        num_heads=4, decoder_embed_dim=32, decoder_depth=1,
                        decoder_num_heads=4, mlp_ratio=2, norm_layer=torch.nn.LayerNorm)
                    return OrganSlotMaskedAutoencoderViT(**kwargs)
                specs = [{'name': n, 'raw_class_id': i} for i, n in enumerate(names)]
                with tempfile.TemporaryDirectory() as tmp, patch.object(evaluator, 'build_orgslot', side_effect=build):
                    path = Path(tmp) / 'checkpoint.pth'
                    torch.save(checkpoint, path)
                    loaded, report = evaluator.build_3d_model(path, specs, 32, 'post_layernorm', 4, 1)
                    self.assertTrue(report['exact'])
                    self.assertEqual(report['hsam_supervision'], mode)
                    loaded.eval()
                    with torch.no_grad():
                        a, b = model(image), loaded(image)
                    for name in names:
                        torch.testing.assert_close(a['calibrated_logits'][name], b['calibrated_logits'][name], rtol=0, atol=0)

    def test_lost_positive_slices_and_empty_rank(self):
        args = loss_args('downsample_gt')
        target = torch.zeros(1, 1, 4, 32, 32)
        target[:, :, 1, 1, 1] = 1
        prediction = torch.randn(1, 1, 4, 8, 8, requires_grad=True)
        loss, lost = coarse_segmentation_loss({'organ': prediction}, {'organ': target},
            ['organ'], torch.ones(1, 1, dtype=torch.bool), args)
        self.assertEqual(lost, 1)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        loss, lost = coarse_segmentation_loss({}, {'background': target}, ['background'],
            torch.ones(1, 1, dtype=torch.bool), loss_args('soft_prior_aux'), {})
        self.assertEqual(loss.item(), 0)
        self.assertEqual(lost, 0)
        with self.assertRaisesRegex(ValueError, 'slice count'):
            coarse_segmentation_loss({'organ': prediction[:, :, :1]}, {'organ': target},
                ['organ'], torch.ones(1, 1, dtype=torch.bool), args)

    def test_bf16_decoder_finite_backward(self):
        from ArmFDecoder import ArmFDecoder
        for mode in ('soft_prior_aux', 'downsample_gt'):
            decoder = ArmFDecoder(3, 32, (4, 2, 2), channels=16, readout='sam_tail')
            decoder.configure_mask_supervision(mode)
            maps = [torch.randn(1, 16, 4, s, s).bfloat16().requires_grad_() for s in (2, 4, 8)]
            tokens = torch.randn(1, 2, 32, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                final, coarse = decoder.forward_mask(maps, tokens, torch.zeros(16), (4, 32, 32))
            loss, _ = coarse_segmentation_loss({'organ': coarse},
                {'organ': torch.zeros_like(final)}, ['organ'],
                torch.ones(1, 1, dtype=torch.bool), loss_args(mode))
            (final.square().mean() + loss).backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            self.assertTrue(all(torch.isfinite(m.grad).all() for m in maps))
