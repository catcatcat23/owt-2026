import unittest
from types import SimpleNamespace

import torch

from engine_pretrain_orgslot import compute_incremental_minimal_objective
from engine_pretrain_orgslot import compute_base_objective
from engine_pretrain_orgslot_common8_a100 import (
    _segmentation_supervision_mask,
    _slot_keep_mask,
)
from test_orgslot_model import tiny_model
from losses_orgslot import (
    base_segmentation_loss,
    new_region_reconstruction_loss,
    old_confidence_suppression_loss,
)
from util.slot_tgr import (
    build_base_reconstruction_target,
    sample_base_slot_keep_mask,
    sample_retain_new_keep_mask,
)


class LossAndTGRTests(unittest.TestCase):
    def test_base_tgr_is_deterministic_per_sample_and_never_empty(self):
        indices = torch.arange(64)
        first = sample_base_slot_keep_mask(indices, epoch=5, slot_count=4, seed=7)
        second = sample_base_slot_keep_mask(indices, epoch=5, slot_count=4, seed=7)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.all(first.sum(1) >= 1))
        self.assertGreater(torch.unique(first, dim=0).shape[0], 1)

    def test_incremental_minimal_retains_every_slot(self):
        keep = sample_retain_new_keep_mask(
            torch.tensor([0, 1]), 3, ["background", "old", "new"], "new",
            minimal=True,
        )
        self.assertTrue(torch.all(keep))

    def test_reconstruction_target_zeroes_only_dropped_slot_pixels(self):
        image = torch.ones(2, 3, 4, 4)
        masks = {
            "background": torch.zeros(2, 1, 4, 4),
            "organ": torch.zeros(2, 1, 4, 4),
        }
        masks["organ"][:, :, :2, :2] = 1
        masks["background"] = 1 - masks["organ"]
        keep = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        target = build_base_reconstruction_target(
            image, masks, ["background", "organ"], keep
        )
        self.assertEqual(target[0, :, :2, :2].sum(), 0)
        self.assertEqual(target[0, :, 2:, 2:].sum(), 12)
        self.assertEqual(target[1, :, 2:, 2:].sum(), 0)
        self.assertEqual(target[1, :, :2, :2].sum(), 12)

    def test_dropped_logits_have_zero_gradient(self):
        retained = torch.zeros(2, 1, 4, 4, requires_grad=True)
        dropped = torch.ones(2, 1, 4, 4, requires_grad=True)
        logits = {"a": retained, "b": dropped}
        targets = {"a": torch.ones_like(retained), "b": torch.zeros_like(dropped)}
        keep = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        loss, _ = base_segmentation_loss(logits, targets, ["a", "b"], keep)
        loss.backward()
        self.assertGreater(retained.grad.abs().sum(), 0)
        self.assertTrue(
            dropped.grad is None or dropped.grad.abs().sum().item() == 0
        )

    def test_all_supervision_gives_dropped_logits_gradient(self):
        retained = torch.zeros(2, 1, 4, 4, requires_grad=True)
        dropped = torch.ones(2, 1, 4, 4, requires_grad=True)
        logits = {"a": retained, "b": dropped}
        targets = {"a": torch.ones_like(retained), "b": torch.zeros_like(dropped)}
        keep = torch.tensor([[1, 0], [1, 0]], dtype=torch.bool)
        supervision = _segmentation_supervision_mask("all", keep)
        loss, _ = base_segmentation_loss(
            logits, targets, ["a", "b"], supervision
        )
        loss.backward()
        self.assertGreater(retained.grad.abs().sum(), 0)
        self.assertGreater(dropped.grad.abs().sum(), 0)

    def test_segmentation_mode_does_not_change_reconstruction_target(self):
        image = torch.ones(2, 3, 4, 4)
        organ = torch.zeros(2, 1, 4, 4)
        organ[:, :, :2, :2] = 1
        masks = {"background": 1 - organ, "organ": organ}
        keep = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
        expected = build_base_reconstruction_target(
            image, masks, ["background", "organ"], keep
        )
        for mode in ("retained", "all"):
            _segmentation_supervision_mask(mode, keep)
            actual = build_base_reconstruction_target(
                image, masks, ["background", "organ"], keep
            )
            self.assertTrue(torch.equal(actual, expected), mode)

    def test_fixed_per_sample_mask_does_not_change_with_epoch(self):
        args = SimpleNamespace(tgr_mode="fixed_per_sample", seed=7)
        batch = {
            "image": torch.zeros(8, 3, 4, 4),
            "sample_index": torch.arange(8),
        }
        first = _slot_keep_mask(args, batch, 0, 4, torch.device("cpu"))
        later = _slot_keep_mask(args, batch, 99, 4, torch.device("cpu"))
        self.assertTrue(torch.equal(first, later))

    def test_new_region_reconstruction_ignores_outside(self):
        image = torch.zeros(1, 3, 4, 4)
        reconstruction = torch.zeros_like(image)
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :2, :2] = 1
        reconstruction[:, :, 2:, 2:] = 100
        self.assertEqual(new_region_reconstruction_loss(reconstruction, image, mask), 0)
        reconstruction[:, :, 0, 0] = 2
        self.assertGreater(new_region_reconstruction_loss(reconstruction, image, mask), 0)

    def test_suppression_detaches_frozen_logits_and_empty_is_finite(self):
        new_logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        old_logits = torch.full((1, 1, 4, 4), 10.0, requires_grad=True)
        new_mask = torch.zeros(1, 1, 4, 4)
        loss, mask = old_confidence_suppression_loss(
            new_logits, {"old": old_logits}, new_mask
        )
        self.assertTrue(mask.all())
        loss.backward()
        self.assertIsNone(old_logits.grad)
        empty_loss, empty_mask = old_confidence_suppression_loss(
            new_logits, {"old": torch.full_like(old_logits, -10)}, new_mask
        )
        self.assertFalse(empty_mask.any())
        self.assertTrue(torch.isfinite(empty_loss))

    def test_incremental_engine_rejects_old_or_full_masks(self):
        class _UnusedModel:
            slot_names = ("background", "old", "liver")

        batch = {
            "image": torch.rand(1, 3, 4, 4),
            "visible_masks": {
                "liver": torch.zeros(1, 1, 4, 4),
                "old": torch.zeros(1, 1, 4, 4),
            },
            "sample_index": torch.tensor([0]),
        }
        with self.assertRaisesRegex(ValueError, "exactly the current new slot"):
            compute_incremental_minimal_objective(
                _UnusedModel(), batch, "liver", epoch=0
            )

    def test_incremental_engine_builds_detached_old_confidence_teacher(self):
        model = tiny_model()
        model.append_slot("liver", 4, init_from="background")
        with torch.no_grad():
            model.slot_bank.get_slot("kidney").calibration_bias.fill_(10)
        batch = {
            "image": torch.rand(1, 3, 32, 32),
            "visible_masks": {"liver": torch.zeros(1, 1, 32, 32)},
            "sample_index": torch.tensor([0]),
        }
        loss, stats = compute_incremental_minimal_objective(
            model,
            batch,
            "liver",
            epoch=0,
            lambda_sup=1.0,
            old_slot_names=("kidney",),
        )
        self.assertTrue(stats["suppression_mask"].all())
        self.assertGreater(float(stats["suppression_loss"]), 0)
        loss.backward()
        self.assertIsNone(
            model.slot_bank.get_slot("kidney").calibration_bias.grad
        )

    def test_calibration_parameters_receive_base_gradients(self):
        model = tiny_model()
        image = torch.rand(2, 3, 32, 32)
        masks = {
            name: torch.zeros(2, 1, 32, 32)
            for name in model.slot_names
        }
        masks["background"].fill_(1)
        batch = {
            "image": image,
            "visible_masks": masks,
            "sample_index": torch.tensor([0, 1]),
        }
        loss, _ = compute_base_objective(model, batch, 0, lambda_lpips=0)
        loss.backward()
        retained = sample_base_slot_keep_mask(
            batch["sample_index"], 0, len(model.slot_names)
        ).any(dim=0)
        for index, name in enumerate(model.slot_names):
            gradient = model.slot_bank.get_slot(name).calibration_scale.grad
            if retained[index]:
                self.assertIsNotNone(gradient)


if __name__ == "__main__":
    unittest.main()
