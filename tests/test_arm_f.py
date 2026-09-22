import copy
import unittest
import math
import subprocess
import types

import torch

from test_orgslot_model import tiny_model
from ArmFDecoder import ArmFDecoder, ArmEStyleDecoder3D, position_encoding


class ArmFTests(unittest.TestCase):
    def test_pre_sam_heads_exact_compatibility(self):
        source = subprocess.check_output(
            ['git', 'show', '1016af0:ArmFDecoder.py'], text=True)
        legacy = types.ModuleType('pre_sam_tail')
        exec(compile(source, 'pre_sam_tail', 'exec'), legacy.__dict__)
        torch.set_num_threads(2)
        for readout in ('attention', 'linear', 'query_dot', 'reverse_dot',
                        'reverse_dot_p2', 'reverse_dot_p2_softmask'):
            torch.manual_seed(23)
            old = legacy.ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            torch.manual_seed(23)
            new = ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            for key in old.state_dict():
                torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key],
                                           rtol=0, atol=0)
            new.load_state_dict(old.state_dict(), strict=True)
            images, z = torch.randn(1, 3, 32, 32), torch.randn(1, 4, 32)
            tokens, identity = torch.randn(1, 20, 32), torch.randn(16)
            torch.testing.assert_close(
                old.forward_mask(old.forward_pixels(images, z), tokens, identity, (32, 32)),
                new.forward_mask(new.forward_pixels(images, z), tokens, identity, (32, 32)),
                rtol=0, atol=0)

    def test_sam_tail_order_condition_and_bf16(self):
        torch.set_num_threads(2)
        for grid in ((2, 2), (4, 2, 2)):
            decoder = ArmFDecoder(3, 32, grid, channels=16, readout='sam_tail')
            events, handles = [], []
            for i, block in enumerate(decoder.blocks):
                handles.append(block.register_forward_hook(
                    lambda m, a, o, name='scale' + str(i): events.append(name)))
            for name in ('self_attn', 'token_to_pixel', 'token_ffn',
                         'pixel_to_token', 'final_token_to_pixel', 'mask_mlp'):
                handles.append(getattr(decoder.sam_tail, name).register_forward_hook(
                    lambda m, a, o, name=name: events.append(name)))
            prefix = () if len(grid) == 2 else (4,)
            maps = [torch.randn(2, 16, *prefix, n, n).to(torch.bfloat16).requires_grad_()
                    for n in (2, 4, 8)]
            tokens = torch.randn(2, 20, 32, requires_grad=True)
            identity = torch.randn(16, requires_grad=True)
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = decoder.forward_mask(maps, tokens, identity, (*prefix, 32, 32))
            for handle in handles:
                handle.remove()
            self.assertEqual(events, ['scale0', 'scale1', 'scale2', 'self_attn',
                'token_to_pixel', 'token_ffn', 'pixel_to_token',
                'final_token_to_pixel', 'mask_mlp'])
            self.assertEqual(result.shape, (2, 1, *prefix, 32, 32))
            result.square().mean().backward()
            for value in [tokens, identity] + maps:
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(value.grad.abs().sum().item(), 0)
            for name, param in decoder.sam_tail.named_parameters():
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
                self.assertGreater(param.grad.abs().sum().item(), 0, name)
            changed = decoder.forward_mask(maps, torch.zeros_like(tokens), identity,
                                           (*prefix, 32, 32))
            self.assertGreater((result - changed).abs().max().item(), 1e-5)

    def test_p2_softmask_control_and_routing(self):
        torch.set_num_threads(2)
        torch.manual_seed(9)
        baseline = ArmFDecoder(3, 32, (2, 2), channels=16, readout='reverse_dot_p2')
        torch.manual_seed(9)
        masked = ArmFDecoder(3, 32, (2, 2), channels=16, readout='reverse_dot_p2_softmask')
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, masked.state_dict()[name], rtol=0, atol=0)
        masked.load_state_dict(baseline.state_dict(), strict=True)
        images, z = torch.randn(2, 3, 32, 32), torch.randn(2, 4, 32)
        maps = baseline.forward_pixels(images, z)
        tokens, identity = torch.randn(2, 20, 32), torch.randn(16)
        expected = baseline.forward_mask(maps, tokens, identity, (32, 32))
        masked.soft_mask_alpha = 0
        actual = masked.forward_mask(maps, tokens, identity, (32, 32))
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        masked.soft_mask_alpha = 1
        actual = masked.forward_mask(maps, tokens, identity, (32, 32))
        self.assertGreater((actual - expected).abs().max().item(), 1e-6)
        bf16_maps = [p.detach().to(torch.bfloat16).requires_grad_() for p in maps]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = masked.forward_mask(bf16_maps, tokens, identity, (32, 32))
        output.square().mean().backward()
        for feature in bf16_maps:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
        # Synthetic matched/unmatched pixels: bias favors matched locations,
        # but even a nearly empty prior never produces -inf/NaN.
        query = torch.ones(2, 20, 16, requires_grad=True)
        pixels = torch.stack((torch.ones(2, 16), -torch.ones(2, 16)), dim=1)
        bias = masked.soft_attention_bias(pixels, query, 4)
        self.assertEqual(bias.shape, (8, 20, 2))
        self.assertFalse(bias.requires_grad)
        self.assertTrue(torch.isfinite(bias).all())
        self.assertGreaterEqual(bias.min().item(), math.log(0.2))
        weights = bias.softmax(-1)
        self.assertTrue((weights[..., 0] > weights[..., 1]).all())
        self.assertTrue((weights[..., 1] > 0).all())
        with self.assertRaises(ValueError):
            ArmFDecoder(3, 32, (4, 2, 2), channels=16, readout='reverse_dot_p2_softmask')

    def test_3d_baseline_matches_slice_wise_e_readout(self):
        decoder = ArmEStyleDecoder3D(3, 32, (4, 2, 2), channels=16)
        self.assertFalse(hasattr(decoder, 'blocks'))
        self.assertFalse(hasattr(decoder, 'reverse'))
        p4 = torch.randn(2, 16, 4, 8, 8)
        tokens, identity = torch.randn(2, 20, 32), torch.randn(16)
        actual = decoder.forward_mask([p4], tokens, identity, (4, 32, 32))
        expected = torch.stack([decoder.pixels.forward_mask(
            p4[:, :, t], tokens, identity, (32, 32)) for t in range(4)], dim=2)
        torch.testing.assert_close(actual, expected)

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

    def test_p2_resolution_skip_and_legacy_dot(self):
        torch.set_num_threads(2)
        source = subprocess.check_output(['git', 'show', '4f6488d:ArmFDecoder.py'], text=True)
        legacy = types.ModuleType('legacy_dot')
        exec(compile(source, 'legacy_dot', 'exec'), legacy.__dict__)
        for readout in ('query_dot', 'reverse_dot'):
            torch.manual_seed(5)
            old = legacy.ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            torch.manual_seed(5)
            new = ArmFDecoder(3, 32, (2, 2), channels=16, readout=readout)
            new.load_state_dict(old.state_dict(), strict=True)
            images, z = torch.randn(1, 3, 32, 32), torch.randn(1, 4, 32)
            tokens, identity = torch.randn(1, 20, 32), torch.randn(16)
            torch.testing.assert_close(
                old.forward_mask(old.forward_pixels(images, z), tokens, identity, (32, 32)),
                new.forward_mask(new.forward_pixels(images, z), tokens, identity, (32, 32)),
                rtol=0, atol=0)
        decoder = ArmFDecoder(3, 32, (2, 2), channels=16, readout='reverse_dot_p2')
        maps = decoder.forward_pixels(images, z)
        self.assertEqual([p.shape[-1] for p in maps], [2, 4, 8, 16])
        observed = []
        hook = decoder.p2_fusion.register_forward_hook(lambda m, a, out: observed.append(out.shape))
        maps = [p.detach().to(torch.bfloat16).requires_grad_() for p in maps]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = decoder.forward_mask(maps, tokens, identity, (32, 32))
        hook.remove()
        self.assertEqual(observed[0], (1, 16, 16, 16))
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(maps[3].grad).all())
        self.assertGreater(maps[3].grad.abs().sum().item(), 0)
        changed = decoder.forward_mask(maps[:3] + [torch.zeros_like(maps[3])], tokens, identity, (32, 32))
        self.assertGreater((output - changed).abs().max().item(), 1e-5)
        with self.assertRaises(ValueError):
            ArmFDecoder(3, 32, (4, 2, 2), channels=16, readout='reverse_dot_p2')

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
            heads = ['arm_f_attention', 'arm_f_linear', 'arm_f_query_dot', 'arm_f_reverse_dot', 'arm_f_sam_tail']
            if dimension == '2D':
                heads.append('arm_f_reverse_dot_p2')
                heads.append('arm_f_reverse_dot_p2_softmask')
            if dimension == '3D':
                heads.append('arm_e_multiscale_query_3d')
            for head in heads:
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
