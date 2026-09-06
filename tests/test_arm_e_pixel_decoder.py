import unittest

import torch

from OrganSlotEmbed import MultiScalePixelQueryDecoder2D


class ArmEPixelDecoderTests(unittest.TestCase):
    def test_448_multiscale_shapes_queries_and_gradients(self):
        torch.manual_seed(17)
        decoder = MultiScalePixelQueryDecoder2D(
            in_channels=3,
            embed_dim=32,
            grid_size=(28, 28),
            channels=16,
        )
        images = torch.rand(2, 3, 448, 448, requires_grad=True)
        z = torch.rand(2, 28 * 28, 32, requires_grad=True)
        pixels, scales = decoder.forward_pixels(
            images, z, return_intermediates=True
        )
        self.assertEqual(scales["p4_raw"].shape, (2, 16, 112, 112))
        self.assertEqual(scales["p8_raw"].shape, (2, 16, 56, 56))
        self.assertEqual(scales["p16"].shape, (2, 16, 28, 28))
        self.assertEqual(pixels.shape, (2, 16, 112, 112))

        base_queries = torch.randn(2, 16, requires_grad=True)
        outputs = {}
        for query_count in (1, 3, 8):
            queries = base_queries[:, None].expand(
                -1, query_count, -1
            ).contiguous()
            outputs[query_count] = decoder.forward_mask_from_query(
                pixels, queries, (448, 448)
            )
            self.assertEqual(
                outputs[query_count].shape,
                (2, query_count, 448, 448),
            )

        normal = decoder.forward_mask_from_query(
            pixels, base_queries, (448, 448)
        )
        zeroed = decoder.forward_mask_from_query(
            pixels, torch.zeros_like(base_queries), (448, 448)
        )
        shuffled = decoder.forward_mask_from_query(
            pixels, base_queries.flip(0), (448, 448)
        )
        self.assertTrue(torch.equal(zeroed, torch.zeros_like(zeroed)))
        self.assertGreater((normal - shuffled).abs().mean(), 1e-4)

        loss = outputs[8].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            decoder.spatial_stem.to_p2[0].weight.grad.abs().sum(), 0
        )
        self.assertGreater(decoder.pixel_proj.weight.grad.abs().sum(), 0)
        self.assertGreater(base_queries.grad.abs().sum(), 0)


if __name__ == "__main__":
    unittest.main()
