"""Private multi-scale interaction, frozen shared tail, no offline leakage."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_orgslot_model import tiny_model
from test_incremental44 import options
from util.orgslot_incremental44 import (
    BASE, OLD, NEW, configure_stage2, Incremental44Objective,
    install_slot_interactions, validate_incremental_architecture,
)


def model():
    net = tiny_model(specs=[{"name": n, "raw_class_id": i} for i, n in enumerate(BASE)],
                     slot_head_type="arm_f_sam_tail", slot_head_channels=16)
    net.pixel_query_decoder.configure_mask_supervision("soft_prior_aux")
    return net


def args_for_test():
    args = options()
    args.stage2_shared_segmentation = "slot_private"
    args.hsam_supervision = "soft_prior_aux"
    args.lambda_bg_seg = 0
    args.seg_loss_type = "small_organ"
    args.segmentation_unit = "slice"
    return args


class PrivateInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_private_stage2_gradients_old_invariance_and_evaluator(self):
        torch.manual_seed(6)
        net = model().eval()
        image = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            stage1 = net(image, decode_reconstruction=False)["slot_logits"]
        for i, name in enumerate(NEW, 5):
            net.append_slot(name, i, init_from="background")
        shared = copy.deepcopy(net).eval()
        with torch.no_grad():
            original = shared(image, decode_reconstruction=False)
        configure_stage2(net, "separation", "slot_private")
        objective = Incremental44Objective(net, args_for_test()).train()
        self.assertFalse(net.pixel_query_decoder.training)
        self.assertFalse(net.pixel_query_decoder.sam_tail.training)
        for n in OLD:
            self.assertFalse(net.slot_bank.get_slot(n).interaction.training)
        for n in NEW:
            self.assertTrue(net.slot_bank.get_slot(n).interaction.training)
        with torch.no_grad():
            migrated = net(image, decode_reconstruction=False)
        for n in OLD + NEW:
            torch.testing.assert_close(original["slot_logits"][n], migrated["slot_logits"][n], atol=0, rtol=0)
            for a, b in zip(original["coarse_logits"][n], migrated["coarse_logits"][n]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        copies = [net.slot_bank.get_slot(n).interaction.input_proj.weight for n in OLD + NEW]
        self.assertEqual(len({p.data_ptr() for p in copies}), 8)
        frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
        masks = {n: torch.zeros(2, 1, 32, 32) for n in NEW}
        for i, n in enumerate(NEW):
            masks[n][:, :, 2+6*i:6+6*i, 8:16] = 1
        probs = {n: torch.zeros_like(masks[NEW[0]]) for n in OLD}
        keep = torch.ones(2, 9, dtype=torch.bool)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
        loss, stats = objective(image, masks, probs, keep)
        torch.testing.assert_close(loss.detach(), stats["reconstruction"] + .01 *
            stats["segmentation"] + stats["weighted_coarse_segmentation"] + .1 * stats["background_separation"])
        loss.backward()
        for n in NEW:
            interaction = net.slot_bank.get_slot(n).interaction
            for module in (interaction.input_proj, *interaction.blocks):
                grads = [p.grad for p in module.parameters() if p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
        optimizer.step()
        for n, p in net.named_parameters():
            if n in frozen:
                self.assertIsNone(p.grad, n)
                self.assertTrue(torch.equal(p, frozen[n]), n)
        net.eval()
        with torch.no_grad():
            after = net(image, decode_reconstruction=False)["slot_logits"]
        for n in OLD:
            torch.testing.assert_close(stage1[n], after[n], atol=0, rtol=0)
        # A frozen tail must transmit final-mask gradients, not only aux gradients.
        optimizer.zero_grad(set_to_none=True)
        final_output = net(image, decode_reconstruction=False)
        final_output["slot_logits"][NEW[0]].square().mean().backward()
        grad = net.slot_bank.get_slot(NEW[0]).interaction.input_proj.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in net.pixel_query_decoder.sam_tail.parameters()))
        from tools.eval_common8_orgslot_reconstruction_threshold import build_model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            torch.save({"model": net.state_dict(), "epoch": 0, "args": {
                "input_size": 32, "incremental_stage": "stage2",
                "slot_head_type": "arm_f_sam_tail", "hsam_supervision": "soft_prior_aux",
                "stage2_shared_segmentation": "slot_private"}}, path)
            with patch("tools.eval_common8_orgslot_reconstruction_threshold.build_orgslot", return_value=shared):
                restored, report = build_model("orgslot", path, [], 32, "linear_sqrt")
            self.assertTrue(report["exact"])
            restored.eval()
            with torch.no_grad():
                prediction = restored(image, decode_reconstruction=False)["slot_logits"]
            for n in OLD + NEW:
                torch.testing.assert_close(prediction[n], after[n], atol=0, rtol=0)

    def test_aux_schema_for_no_retained_new_slots(self):
        net = model()
        for i, n in enumerate(NEW, 5):
            net.append_slot(n, i)
        configure_stage2(net, "separation", "slot_private")
        args = args_for_test()
        args.seg_supervision = "retained"
        objective = Incremental44Objective(net, args).train()
        image = torch.rand(1, 3, 32, 32)
        masks = {n: torch.zeros(1, 1, 32, 32) for n in NEW}
        probs = {n: torch.zeros_like(masks[NEW[0]]) for n in OLD}
        keep = torch.zeros(1, 9, dtype=torch.bool)
        keep[:, 0] = True
        loss, empty = objective(image, masks, probs, keep)
        loss.backward()
        keep[:] = True
        _, full = objective(image, masks, probs, keep)
        self.assertEqual(set(empty), set(full))
        self.assertEqual(float(empty["coarse_segmentation"]), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_reject_incompatible_architecture(self):
        validate_incremental_architecture({"slot_head_type": "arm_e_multiscale_query", "query_refinement": "cross_attn"})
        with self.assertRaises(ValueError):
            validate_incremental_architecture({"slot_head_type": "arm_f_sam_tail", "query_refinement": "none"})
        net = model()
        net.pixel_query_decoder.configure_mask_supervision("none")
        with self.assertRaises(ValueError):
            install_slot_interactions(net)

    def test_tiny_batch_loss_descends(self):
        # Random frozen tail/pixels: test optimization, not near-zero overfit.
        torch.manual_seed(8)
        net = model()
        for i, n in enumerate(NEW, 5):
            net.append_slot(n, i)
        configure_stage2(net, "separation", "slot_private")
        objective = Incremental44Objective(net, args_for_test()).train()
        image = torch.zeros(2, 3, 32, 32)
        image[:, :, 8:24, 8:24] = .7
        masks = {n: torch.zeros(2, 1, 32, 32) for n in NEW}
        masks[NEW[0]][:, :, 8:24, 8:24] = 1
        probs = {n: torch.zeros_like(masks[NEW[0]]) for n in OLD}
        keep = torch.ones(2, 9, dtype=torch.bool)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=.003)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss, _ = objective(image, masks, probs, keep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(sum(losses[-3:]) / 3, sum(losses[:3]) / 3, losses)


if __name__ == "__main__":
    unittest.main()
