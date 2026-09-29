import unittest
from unittest.mock import patch
import torch
from test_orgslot_model import tiny_model
from tools.token_intervention import TokenBank


class TokenInterventionTests(unittest.TestCase):
    def test_bank(self):
        bank = TokenBank(10, 0)
        bank.add('kidney', True, torch.ones(2, 4), 'train_a', 0)
        bank.add('kidney', True, torch.ones(2, 4)*3, 'train_b', 1)
        bank.add('kidney', False, torch.zeros(2, 4), 'train_c', 2)
        torch.testing.assert_close(bank.mean('kidney', True), torch.ones(2, 4)*2)
        self.assertEqual(bank.donor('kidney', True, ('train_a',))[1], 'train_b')

    def test_cached_readout_equivalence(self):
        torch.set_num_threads(2)
        model = tiny_model(slot_head_type='arm_e_multiscale_query', slot_head_channels=16).eval()
        model.requires_grad_(False)
        decoder = model.pixel_query_decoder
        orig = decoder.forward_pixels
        captured = {}
        def capture(*args, **kwargs):
            captured['p'] = orig(*args, **kwargs)
            return captured['p']
        with torch.inference_mode(), patch.object(decoder, 'forward_pixels', capture):
            output = model(torch.randn(2, 3, 32, 32), decode_reconstruction=False, return_diagnostics=True)
            for name in model.slot_names:
                slot = model.slot_bank.get_slot(name)
                logits = slot.forward_query_head(output['slot_tokens'][name], captured['p'], decoder, (32,32))[1]
                torch.testing.assert_close(logits, output['calibrated_logits'][name], rtol=1e-5, atol=1e-5)
                replacement = output['slot_tokens'][name].flip(0)
                swapped = slot.forward_query_head(replacement, captured['p'], decoder, (32,32))[1]
                self.assertTrue(torch.isfinite(swapped).all())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
