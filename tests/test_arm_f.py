import copy
import unittest
import math
import subprocess
import types

import torch

from test_orgslot_model import tiny_model
from ArmFDecoder import ArmFDecoder, position_encoding


class ArmFTests(unittest.TestCase):
    def test_legacy_decoder_unchanged(self):
        source = subprocess.check_output([
            'git', 'show', 'fd9cf91b5daeec415cd09d239f484d9b8500e712:ArmFDecoder.py'
        ], text=True)
        legacy = types.ModuleType('legacy_arm_f')
        exec(compile(source, 'legacy_arm_f', 'exec'), legacy.__dict__)
        torch.set_num_threads(2)
        for readout in ('attention', 'linear'):
            torch.manual_seed(7)
            old = legacy.ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            torch.manual_seed(7)
            new = ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            for name, value in old.state_dict().items():
                torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
            new.load_state_dict(old.state_dict(), strict=True)
            maps = [torch.randn(1, 16, size, size) for size in (2, 4, 8)]
            tokens, identity = torch.randn(1, 20, 32), torch.randn(16)
            torch.testing.assert_close(old.forward_mask(maps, tokens, identity, (32, 32)),
                                       new.forward_mask(maps, tokens, identity, (32, 32)), rtol=0, atol=0)

    def test_dot_contract(self):
        query = ArmFDecoder(3, 32, (2, 2), channels=16, readout='query_dot')
        reverse = ArmFDecoder(3, 32, (2, 2), channels=16, readout='reverse_dot')
        self.assertFalse(hasattr(query, 'reverse'))
        self.assertFalse(hasattr(query, 'classifier'))
        self.assertFalse(hasattr(reverse, 'classifier'))
        pixels, tokens = torch.randn(2, 64, 16), torch.randn(2, 20, 16)
        expected = torch.nn.functional.normalize(pixels, dim=-1) @ torch.nn.functional.normalize(
            tokens.mean(1), dim=-1).unsqueeze(-1) * math.sqrt(16)
        torch.testing.assert_close(query.dot_readout(pixels, tokens), expected)

    def test_model_backward_and_checkpoint(self):
        torch.set_num_threads(2)
        for dimension in ('2D', '3D'):
            for head in ('arm_f_attention', 'arm_f_linear', 'arm_f_query_dot', 'arm_f_reverse_dot'):
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
