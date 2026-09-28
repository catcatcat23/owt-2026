import unittest
import torch
from test_orgslot_model import tiny_model


class AllSegTests(unittest.TestCase):
    def test_dropped_slot_gets_seg_gradient_without_changing_reconstruction(self):
        torch.set_num_threads(2)
        torch.manual_seed(12)
        model = tiny_model(slot_head_type='arm_f_sam_tail',slot_head_channels=16).eval()
        image = torch.rand(2,3,32,32)
        keep = torch.tensor([[True,False,True],[True,False,True]])
        original_keep = keep.clone()
        retained = keep.clone(); retained[:,0] = False
        all_heads = torch.ones_like(keep); all_heads[:,0] = False
        with torch.no_grad():
            reference = model(image,slot_keep_mask=keep,head_compute_mask=retained)
        output = model(image,slot_keep_mask=keep,head_compute_mask=all_heads,return_diagnostics=True)
        torch.testing.assert_close(keep,original_keep)
        torch.testing.assert_close(output['reconstruction'],reference['reconstruction'],rtol=0,atol=0)
        self.assertTrue(output['slot_compute_mask'][:,1].all())
        self.assertFalse(output['slot_keep_mask'][:,1].any())
        tokens = output['slot_tokens']['kidney']
        self.assertGreater(tokens.abs().sum().item(),0)
        tokens.retain_grad()
        logits = output['calibrated_logits']['kidney']
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.ones_like(logits))
        loss.backward()
        slot = model.slot_bank.get_slot('kidney')
        grads = [p.grad for p in slot.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads),0)
