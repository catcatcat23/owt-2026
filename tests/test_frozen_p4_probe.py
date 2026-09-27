import unittest
import torch
from ArmFDecoder import ArmFDecoder
from tools.frozen_p4_probe import FeatureCapture, probe_loss


class ProbeTests(unittest.TestCase):
    def test_capture_and_frozen_backward(self):
        torch.set_num_threads(2)
        d = ArmFDecoder(3,32,(2,2),channels=16,readout='attention').eval()
        d.requires_grad_(False)
        maps = [torch.randn(2,16,n,n) for n in (2,4,8)]
        tokens, identity = torch.randn(2,20,32), torch.randn(16)
        with torch.no_grad():
            expected=d.forward_mask(maps,tokens,identity,(32,32))
        capture=FeatureCapture(d)
        with torch.no_grad():
            actual=d.forward_mask(maps,tokens,identity,(32,32))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(capture.values['pre'],maps[2].flatten(2).transpose(1,2))
        head=torch.nn.Linear(16,1)
        target=torch.zeros(2,1,8,8);target[0,:,2:5,2:5]=1
        loss=sum(probe_loss(head(v).transpose(1,2).reshape(2,1,8,8),target) for v in capture.values.values())
        loss.backward()
        self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertTrue(all(p.grad is None for p in d.parameters()))
        capture.close()

    def test_reject_wrong_head(self):
        with self.assertRaises(ValueError):
            FeatureCapture(ArmFDecoder(3,32,(2,2),channels=16,readout='query_dot'))

    def test_empty_loss(self):
        x=torch.randn(2,1,8,8,requires_grad=True)
        probe_loss(x,torch.zeros_like(x)).backward()
        self.assertTrue(torch.isfinite(x.grad).all())
