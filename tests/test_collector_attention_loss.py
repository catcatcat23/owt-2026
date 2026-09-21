import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from losses_orgslot import collector_attention_loss


def fixture(batch=2):
    scores = torch.zeros(batch, 2, 4, requires_grad=True)
    mask = torch.zeros(batch, 1, 4, 4)
    mask[0, 0, 0, 0] = 1  # one pixel survives patch pooling
    return scores, mask


def ddp_worker(rank, path):
    dist.init_process_group("gloo", init_method="file://" + path,
                            rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        for second_rank_positive in (False, True):
            scores, mask = fixture()
            if rank == 1:
                mask.zero_()
                if second_rank_positive:
                    mask[:, 0, 3, 3] = 1
            loss, stats = collector_attention_loss(
                {"organ": scores.softmax(-1)}, {"organ": mask}, ["organ"],
                torch.ones(2, 1, dtype=torch.bool), (2, 2))
            loss.backward()
            expected_count = 3 if second_rank_positive else 1
            assert stats["collector_valid_pairs"] == expected_count
            # Shared parameter gradient: sum batch then average DDP ranks.
            gradient = scores.grad.sum(0)
            dist.all_reduce(gradient)
            gradient /= 2
            reference = torch.zeros(1, 2, 4, requires_grad=True)
            a = reference.softmax(-1)
            target_loss = (1 - a[..., 0]).mean()
            if second_rank_positive:
                target_loss = (target_loss + 2 * (1 - a[..., 3]).mean()) / 3
            target_loss.backward()
            torch.testing.assert_close(gradient, reference.grad[0])
    finally:
        dist.destroy_process_group()


class CollectorAlignmentTests(unittest.TestCase):
    def test_inside_outside_and_tiny_foreground(self):
        _, mask = fixture(1)
        for weights, expected in (([1., 0, 0, 0], 0.), ([0., 1, 0, 0], 1.)):
            a = torch.tensor([[weights]], requires_grad=True)
            loss, stats = collector_attention_loss({"organ": a}, {"organ": mask},
                ["organ"], torch.ones(1, 1, dtype=torch.bool), (2, 2))
            self.assertEqual(loss.item(), expected)
            self.assertEqual(stats["collector_organ_support_fraction"].item(), .25)
            loss.backward()
            self.assertTrue(torch.isfinite(a.grad).all())

    def test_mixed_empty_missing_background_and_dropped(self):
        scores, mask = fixture()
        names = ["background", "organ", "unknown", "dropped"]
        attentions = {name: scores.softmax(-1) for name in names}
        keep = torch.ones(2, 4, dtype=torch.bool)
        keep[:, 3] = False
        loss, stats = collector_attention_loss(attentions,
            {"background": torch.ones_like(mask), "organ": mask, "dropped": mask},
            names, keep, (2, 2))
        self.assertEqual(loss.item(), .75)
        self.assertEqual(stats["collector_valid_pairs"].item(), 1)
        loss.backward()
        self.assertEqual(scores.grad[1].abs().sum().item(), 0)
        self.assertGreater(scores.grad[0].abs().sum().item(), 0)

    def test_empty_batch_and_3d_rejection(self):
        scores, mask = fixture()
        loss, stats = collector_attention_loss({"organ": scores.softmax(-1)},
            {"organ": mask * 0}, ["organ"], torch.ones(2, 1, dtype=torch.bool), (2, 2))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertEqual(stats["collector_valid_pairs"].item(), 0)
        with self.assertRaises(ValueError):
            collector_attention_loss({}, {}, [], torch.ones(1, 1), (4, 2, 2))

    def test_collector_can_learn_localization(self):
        from OrganEmbed import OrganCollector
        torch.manual_seed(0)
        collector = OrganCollector(4, 4, 2, 2)
        features = torch.eye(4).unsqueeze(0)
        _, mask = fixture(1)
        optimizer = torch.optim.Adam(collector.parameters(), lr=.05)
        values = []
        for _ in range(50):
            optimizer.zero_grad()
            _, attention = collector(features)
            loss, _ = collector_attention_loss({"organ": attention}, {"organ": mask},
                ["organ"], torch.ones(1, 1, dtype=torch.bool), (2, 2))
            loss.backward()
            self.assertTrue(torch.isfinite(collector.conv1.weight.grad).all())
            optimizer.step()
            values.append(loss.item())
        self.assertLess(values[-1], .05 * values[0])

    def test_arm_e_diagnostics_preserve_forward_and_checkpoint(self):
        from test_orgslot_model import tiny_model
        torch.manual_seed(0)
        model = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=32)
        model.eval()
        image = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.bool)
        original = model(image, keep, head_compute_mask=keep)
        diagnostic = model(image, keep, head_compute_mask=keep, return_diagnostics=True)
        torch.testing.assert_close(original["reconstruction"], diagnostic["reconstruction"], rtol=0, atol=0)
        for name in model.slot_names:
            torch.testing.assert_close(original["calibrated_logits"][name],
                                       diagnostic["calibrated_logits"][name], rtol=0, atol=0)
        masks = {name: torch.zeros(2, 1, 32, 32) for name in model.slot_names}
        masks["kidney"][0, 0, 0, 0] = 1
        loss, _ = collector_attention_loss(diagnostic["collector_attention"], masks,
                                           model.slot_names, keep, (2, 2))
        loss.backward()
        grad = model.slot_bank.get_slot("kidney").collector.conv2.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        clone = tiny_model(slot_head_type="arm_e_multiscale_query", slot_head_channels=32)
        clone.load_state_dict(model.state_dict(), strict=True)

    def test_ddp_unequal_and_empty_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_worker, args=(os.path.join(directory, "rdzv"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
