import copy
import unittest

import torch

from test_orgslot_model import tiny_model
from ArmFDecoder import ArmFDecoder, position_encoding


class ArmFTests(unittest.TestCase):
    def test_model_backward_and_checkpoint(self):
        torch.set_num_threads(2)
        for dimension in ('2D', '3D'):
            for head in ('arm_f_attention', 'arm_f_linear'):
                with self.subTest(dimension=dimension, head=head):
                    model = tiny_model(dimension, slot_head_type=head, slot_head_channels=16)
                    shape = (2, 3, 32, 32) if dimension == '2D' else (2, 3, 4, 32, 32)
                    images = torch.randn(*shape)
                    keep = torch.ones(2, 3, dtype=torch.bool)
                    active = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
                    output = model(images, keep, head_compute_mask=active)
                    self.assertEqual(output['reconstruction'].shape, images.shape)
                    loss = output['reconstruction'].square().mean()
                    for i, name in enumerate(model.slot_names):
                        logits = output['slot_logits'][name]
                        self.assertEqual(logits.shape, (2, 1) + shape[2:])
                        self.assertEqual(logits[~active[:, i]].count_nonzero().item(), 0)
                        loss = loss + logits[active[:, i]].square().mean()
                    loss.backward()
                    for name, p in model.pixel_query_decoder.named_parameters():
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                    for slot in model.slot_bank.slots.values():
                        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in slot.collector.parameters()))
                    clone = tiny_model(dimension, slot_head_type=head, slot_head_channels=16)
                    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
                    model.eval()
                    clone.eval()
                    with torch.no_grad():
                        expected = model(images, keep, decode_reconstruction=False)['slot_logits']
                        actual = clone(images, keep, decode_reconstruction=False)['slot_logits']
                    for name in expected:
                        torch.testing.assert_close(expected[name], actual[name])

    def test_token_condition_and_bfloat16(self):
        for grid in ((2, 2), (4, 2, 2)):
            decoder = ArmFDecoder(3, 32, grid, channels=16, token_count=20)
            spatial = (32, 32) if len(grid) == 2 else (4, 32, 32)
            images = torch.randn(1, 3, *spatial)
            z = torch.randn(1, int(torch.tensor(grid).prod()), 32)
            tokens = torch.randn(1, 20, 32, requires_grad=True)
            identity = torch.zeros(16)
            # This cluster's CPU GroupNorm does not support autocast BF16.
            # Exercise BF16 memory at the new attention boundary separately.
            maps = [p.to(torch.bfloat16) for p in decoder.forward_pixels(images, z)]
            with torch.autocast('cpu', dtype=torch.bfloat16):
                first = decoder.forward_mask(maps, tokens, identity, spatial)
                changed = decoder.forward_mask(maps, torch.randn_like(tokens), identity, spatial)
            self.assertGreater((first - changed).abs().max().item(), 1e-5)
            first.square().mean().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            self.assertGreater(tokens.grad.abs().sum().item(), 0)
        pe = position_encoding((4, 8, 8), 16, torch.device('cpu')).reshape(4, 8, 8, 16)
        self.assertFalse(torch.equal(pe[0], pe[1]))


if __name__ == '__main__':
    unittest.main()
