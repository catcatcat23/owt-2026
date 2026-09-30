from types import SimpleNamespace
import unittest

import torch

from test_orgslot_model import tiny_model
from util.label_visibility import load_visibility_config, build_visible_binary_masks
from util.orgslot_incremental44 import (
    BASE, OLD, NEW, teacher_regions, region_mse, composition_target,
    configure_stage2, Incremental44Objective,
)
from datasets.orgslot_highres import OrganSlotHighResTransform
from tools.incremental44_final_checkpoint import validate_stage1_completion


def options(policy="separation"):
    return SimpleNamespace(background_policy=policy, teacher_low=.1, teacher_high=.9,
        seg_supervision="all", focal_alpha=.75, focal_gamma=2., tversky_alpha_fp=.3,
        tversky_beta_fn=.7, tversky_eps=1e-6, balanced_focal_weight=.5,
        hard_negative_ratio=.02, negative_slice_weight=.1, background_reduction="topk",
        background_preserve_weight=.1, lambda_seg=.01, lambda_background=.1)


class Incremental44Tests(unittest.TestCase):
    def test_stage1_completion_guard(self):
        checkpoint = {"args": {"incremental_stage": "stage1", "max_optimizer_updates": 59400},
                      "optimizer": {"state": {0: {"step": torch.tensor(59400.)}, 1: {"step": 2000}}}}
        validate_stage1_completion(checkpoint)
        checkpoint["optimizer"]["state"][0]["step"] = 59399
        with self.assertRaises(ValueError):
            validate_stage1_completion(checkpoint)
        checkpoint["args"]["incremental_stage"] = "offline"
        with self.assertRaises(ValueError):
            validate_stage1_completion(checkpoint)

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_visibility_and_roi_ignore_hidden_labels(self):
        classes, stages = load_visibility_config("configs/orgslot/common8_incremental44.json")
        for stage, visible, hidden in (("stage1", 4, (5, 6)), ("stage2", 6, (1, 2))):
            label = torch.zeros(3, 40, 40, dtype=torch.long)
            label[:, 10:20, 10:20] = visible
            label[:, 25:30, 25:30] = hidden[0]
            changed = label.clone()
            changed[changed == hidden[0]] = hidden[1]
            transform = OrganSlotHighResTransform(output_size=32, global_crop_size=32, roi_crop_size=24)
            image = torch.rand(3, 40, 40)
            outputs = []
            for target in (label, changed):
                torch.manual_seed(12)
                sample = transform({"image": image.clone(), "label": target,
                                    "roi_applied": True, "focus_class_id": visible})
                outputs.append((sample, build_visible_binary_masks(sample["label"], classes, stages[stage])))
            self.assertTrue(torch.equal(outputs[0][0]["image"], outputs[1][0]["image"]))
            for name in outputs[0][1]:
                self.assertTrue(torch.equal(outputs[0][1][name], outputs[1][1][name]))
            self.assertEqual(set(outputs[0][1]), set(BASE if stage == "stage1" else NEW))

    def test_regions_new_gt_priority_conflict_ignore_and_background(self):
        masks = {n: torch.zeros(1, 1, 1, 4) for n in NEW}
        masks[NEW[0]][..., 0] = 1
        probs = {n: torch.zeros(1, 1, 1, 4) for n in OLD}
        probs[OLD[0]][..., :3] = 1
        probs[OLD[1]][..., 2] = 1
        regions, valid = teacher_regions(masks, probs)
        self.assertEqual(valid.flatten().tolist(), [True, True, False, True])
        self.assertEqual(regions[OLD[0]].flatten().tolist(), [False, True, False, False])
        self.assertEqual(regions["background"].flatten().tolist(), [False, False, False, True])
        target = composition_target(torch.ones(1, 3, 1, 4), regions,
                                    torch.tensor([[1, 0, 0, 0, 0, 1, 0, 0, 0]]), BASE + NEW)
        self.assertEqual(target[0, 0, 0].tolist(), [1, 0, 0, 1])

    def test_empty_region_finite_backward(self):
        pred = torch.rand(1, 3, 4, 4, requires_grad=True)
        loss = region_mse(pred, torch.zeros_like(pred), torch.zeros(1, 1, 4, 4, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_stage2_update_preserves_old_logits_and_frozen_state(self):
        torch.manual_seed(9)
        net = tiny_model(specs=[{"name": n, "raw_class_id": i} for i, n in enumerate(BASE)],
                         slot_head_type="arm_e_multiscale_query", slot_head_channels=16,
                         query_refinement="cross_attn")
        image = torch.rand(2, 3, 32, 32)
        net.eval()
        with torch.no_grad():
            before_logits = net(image, decode_reconstruction=False)["slot_logits"]
        for i, name in enumerate(NEW, 5):
            net.append_slot(name, i, init_from="background")
        configure_stage2(net, "separation")
        frozen = {n: p.detach().clone() for n, p in net.named_parameters() if not p.requires_grad}
        objective = Incremental44Objective(net, options()).train()
        masks = {n: torch.zeros(2, 1, 32, 32) for n in NEW}
        for i, n in enumerate(NEW):
            masks[n][:, :, 2 + 6*i:6 + 6*i, 8:16] = 1
        probs = {n: torch.zeros_like(masks[NEW[0]]) for n in OLD}
        keep = torch.ones(2, 9, dtype=torch.bool)
        optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-3)
        loss, stats = objective(image, masks, probs, keep)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for n in ("background",) + NEW:
            slot = net.slot_bank.get_slot(n)
            grads = [p.grad for p in slot.collector.parameters() if p.requires_grad and p.grad is not None]
            self.assertTrue(grads, n)
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0, n)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads), n)
        optimizer.step()
        for n, p in net.named_parameters():
            if n in frozen:
                self.assertTrue(torch.equal(p, frozen[n]), n)
        net.eval()
        with torch.no_grad():
            after_logits = net(image, decode_reconstruction=False)["slot_logits"]
        for n in OLD:
            torch.testing.assert_close(before_logits[n], after_logits[n], atol=0, rtol=0)

    def test_background_policies(self):
        net = tiny_model(specs=[{"name": n, "raw_class_id": i} for i, n in enumerate(BASE + NEW)])
        for policy in ("frozen", "composition", "separation"):
            configure_stage2(net, policy)
            bg = net.slot_bank.get_slot("background")
            self.assertFalse(any(p.requires_grad for p in bg.head.parameters()))
            self.assertEqual(any(p.requires_grad for p in bg.tg_encoder.parameters()), policy != "frozen")

    def test_tiny_batch_optimization_descends_not_an_overfit_claim(self):
        torch.manual_seed(3)
        net = tiny_model(specs=[{"name": n, "raw_class_id": i} for i, n in enumerate(BASE + NEW)],
                         slot_head_type="arm_e_multiscale_query", slot_head_channels=16,
                         query_refinement="cross_attn")
        configure_stage2(net, "separation")
        objective = Incremental44Objective(net, options()).train()
        image = torch.zeros(2, 3, 32, 32)
        image[:, :, 8:24, 8:24] = .7
        masks = {n: torch.zeros(2, 1, 32, 32) for n in NEW}
        masks[NEW[0]][:, :, 8:24, 8:24] = 1
        probs = {n: torch.zeros_like(masks[NEW[0]]) for n in OLD}
        keep = torch.ones(2, 9, dtype=torch.bool)
        optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=.01)
        losses = []
        for _ in range(25):
            optimizer.zero_grad()
            loss, _ = objective(image, masks, probs, keep)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss))
        # Decoder is deliberately frozen and RANDOM in this unit fixture.
        # Test descent, not near-zero fit; real GPU preflight must use stage1.
        self.assertLess(losses[-1], losses[0], losses)


if __name__ == "__main__":
    unittest.main()
