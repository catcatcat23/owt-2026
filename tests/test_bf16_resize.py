import unittest

import torch
import torch.nn.functional as F

from OrganSlotEmbed import MultiScalePixelQueryDecoder2D, _interpolate_bf16_safe


class BF16ResizeTests(unittest.TestCase):
    def test_resize_preserves_dtype_values_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 3, 4, 4, dtype=dtype, requires_grad=True)
                y = _interpolate_bf16_safe(
                    x, size=(9, 9), mode="bilinear", align_corners=False
                )
                expected = F.interpolate(
                    x.float(), size=(9, 9), mode="bilinear", align_corners=False
                ).to(dtype)
                self.assertEqual(y.dtype, dtype)
                torch.testing.assert_close(y, expected, rtol=0, atol=0)
                y.float().square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(x.grad.abs().sum().item(), 0)

    @unittest.skipUnless(
        torch.cuda.is_available(), "Full BF16 decoder requires a CUDA worker"
    )
    def test_cross_attention_bf16_autocast_forward_backward(self):
        torch.manual_seed(7)
        decoder = MultiScalePixelQueryDecoder2D(
            3, 16, (2, 2), channels=16, query_refinement="cross_attn"
        ).cuda()
        images = torch.rand(2, 3, 32, 32, device="cuda", requires_grad=True)
        z = torch.rand(2, 4, 16, device="cuda", requires_grad=True)
        tokens = torch.rand(2, 3, 16, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pixels = decoder.forward_pixels(images, z)
            logits = decoder.forward_mask(
                pixels, tokens, torch.zeros(16, device="cuda"), (32, 32)
            )
        self.assertEqual(tuple(logits.shape), (2, 1, 32, 32))
        self.assertEqual(logits.dtype, torch.bfloat16)
        loss = logits.float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for value in (images, z, tokens):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
