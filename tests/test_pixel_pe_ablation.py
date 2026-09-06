import unittest
import torch
from OrganSlotEmbed import SharedPixelQueryDecoder


class PixelPETests(unittest.TestCase):
    def test_modes_shapes_gradients_and_checkpoint(self):
        for grid, modes in [((2, 2), ("none", "spatial")),
                            ((4, 2, 2), ("none", "spatial", "temporal", "spatiotemporal"))]:
            for mode in modes:
                model = SharedPixelQueryDecoder(16, grid, channels=16, pixel_pe=mode)
                z = torch.randn(2, int(torch.tensor(grid).prod()), 16, requires_grad=True)
                result = model.forward_pixels(z)
                self.assertEqual(tuple(result.shape), (2, 16, *grid[:-2], 8, 8))
                result.square().mean().backward()
                self.assertTrue(torch.isfinite(z.grad).all())
                for name, p in model.named_parameters():
                    if name.endswith("_pe"):
                        self.assertIsNotNone(p.grad)
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(float(p.grad.abs().sum()), 0)
                clone = SharedPixelQueryDecoder(16, grid, channels=16, pixel_pe=mode)
                clone.load_state_dict(model.state_dict(), strict=True)
                torch.testing.assert_close(result, clone.forward_pixels(z))

    def test_shared_initialization_unchanged(self):
        torch.manual_seed(3)
        baseline = SharedPixelQueryDecoder(16, (4, 2, 2), channels=16)
        for mode in ("spatial", "temporal", "spatiotemporal"):
            torch.manual_seed(3)
            candidate = SharedPixelQueryDecoder(16, (4, 2, 2), channels=16, pixel_pe=mode)
            for key, value in baseline.state_dict().items():
                torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)

    def test_axes_and_invalid_2d_temporal(self):
        model = SharedPixelQueryDecoder(16, (4, 2, 2), channels=16, pixel_pe="spatiotemporal")
        self.assertEqual(tuple(model.spatial_pe.unsqueeze(2).shape), (1, 16, 1, 2, 2))
        self.assertEqual(tuple(model.temporal_pe.shape), (1, 16, 4, 1, 1))
        with self.assertRaises(ValueError):
            SharedPixelQueryDecoder(16, (2, 2), channels=16, pixel_pe="temporal")
