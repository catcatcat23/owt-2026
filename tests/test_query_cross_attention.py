import unittest
import torch
from OrganSlotEmbed import MultiScalePixelQueryDecoder2D
from test_orgslot_model import tiny_model


class QueryCrossAttentionTests(unittest.TestCase):
    def test_tiny_fixed_batch_can_reduce_loss(self):
        torch.manual_seed(11)
        decoder = MultiScalePixelQueryDecoder2D(3, 16, (2, 2), channels=16,
                                               query_refinement="cross_attn")
        images, z, tokens = torch.rand(2, 3, 32, 32), torch.rand(2, 4, 16), torch.rand(2, 2, 16)
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 8:24, 8:24] = 1
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=0.003)
        losses = []
        for _ in range(25):
            pixels = decoder.forward_pixels(images, z)
            logits = decoder.forward_mask(pixels, tokens, torch.zeros(16), (32, 32))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0] * 0.85)

    def test_baseline_initialization_and_strict_roundtrip(self):
        torch.manual_seed(4)
        baseline = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=16)
        torch.manual_seed(4)
        model = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=16,
                           query_refinement="cross_attn")
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        clone = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=16,
                           query_refinement="cross_attn")
        clone.load_state_dict(model.state_dict(), strict=True)
        image = torch.rand(2, 3, 32, 32)
        model.eval()
        clone.eval()
        a = model(image, decode_reconstruction=False)["calibrated_logits"]
        b = clone(image, decode_reconstruction=False)["calibrated_logits"]
        for key in a:
            torch.testing.assert_close(a[key], b[key])

    def test_attention_normalization_interventions_and_finite_backward(self):
        torch.manual_seed(7)
        decoder = MultiScalePixelQueryDecoder2D(3, 16, (2, 2), channels=16,
                                               query_refinement="cross_attn")
        pixels = decoder.forward_pixels(torch.rand(2, 3, 32, 32), torch.rand(2, 4, 16))
        tokens = torch.rand(2, 3, 16)
        query = decoder.forward_query(tokens, torch.zeros(16))
        normal, weights = decoder.query_cross_attention(query, pixels)
        uniform, _ = decoder.query_cross_attention(query, pixels, mode="uniform")
        bypass, _ = decoder.query_cross_attention(query, pixels, mode="bypass")
        self.assertEqual(tuple(weights.shape), (2, 4, 4, 4))
        torch.testing.assert_close(weights.sum((-1, -2)), torch.ones(2, 4))
        torch.testing.assert_close(bypass, query.float())
        self.assertGreater(float((normal - uniform).abs().max()), 0)
        logits = decoder.forward_mask_from_query(pixels, normal, (32, 32))
        logits.square().mean().backward()
        for name, p in decoder.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        self.assertGreater(float(decoder.query_cross_attention.k_proj.weight.grad.abs().sum()), 0)

    def test_tiny_joint_model_backward_with_partial_heads(self):
        model = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=16,
                           query_refinement="cross_attn")
        keep = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
        result = model(torch.rand(2, 3, 32, 32), slot_keep_mask=keep,
                       head_compute_mask=keep)
        loss = result["reconstruction"].square().mean()
        loss = loss + sum(v.square().mean() for v in result["calibrated_logits"].values())
        loss.backward()
        for name, p in model.pixel_query_decoder.query_cross_attention.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_query_depends_on_memory(self):
        decoder = MultiScalePixelQueryDecoder2D(3, 16, (2, 2), channels=16,
                                               query_refinement="cross_attn")
        query = torch.randn(1, 16)
        a, _ = decoder.query_cross_attention(query, torch.randn(1, 16, 8, 8))
        b, _ = decoder.query_cross_attention(query, torch.zeros(1, 16, 8, 8))
        self.assertGreater(float((a - b).abs().max()), 0)
