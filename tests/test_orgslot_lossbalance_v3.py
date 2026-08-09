import unittest

import torch

from util.orgslot_lossbalance_v3 import (
    build_class_frequency_weights,
    state_separated_mask_roi_l2,
)


class OrganSlotLossBalanceV3Tests(unittest.TestCase):
    def test_kept_and_removed_masks_are_routed_separately(self):
        target = torch.zeros((2, 3, 1, 2))
        pred = torch.zeros_like(target, requires_grad=True)
        with torch.no_grad():
            pred[0, :, :, 0] = 1.0
            pred[0, :, :, 1] = 2.0
            pred[1, :, :, 0] = 3.0
            pred[1, :, :, 1] = 4.0
        masks = {
            "background": torch.zeros((2, 1, 1, 2)),
            "class1": torch.tensor([[[[1.0, 0.0]]]]).repeat(2, 1, 1, 1),
            "class2": torch.tensor([[[[0.0, 1.0]]]]).repeat(2, 1, 1, 1),
        }
        stats = state_separated_mask_roi_l2(
            pred, target, masks, ("background", "class1", "class2"),
            torch.tensor([[True, True, False], [True, False, True]]),
            torch.tensor([0.0, 2.0, 3.0]),
        )
        self.assertAlmostEqual(stats["positive"]["loss"].item(), 25.0)
        self.assertAlmostEqual(stats["removed"]["loss"].item(), 6.5)
        stats["positive"]["loss"].backward()
        self.assertTrue(torch.count_nonzero(pred.grad[0, :, :, 0]))
        self.assertTrue(torch.count_nonzero(pred.grad[1, :, :, 1]))
        self.assertEqual(torch.count_nonzero(pred.grad[0, :, :, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(pred.grad[1, :, :, 0]).item(), 0)

    def test_absent_foreground_is_excluded(self):
        pred = torch.ones((1, 3, 2, 2), requires_grad=True)
        masks = {
            "background": torch.ones((1, 1, 2, 2)),
            "organ": torch.zeros((1, 1, 2, 2)),
        }
        stats = state_separated_mask_roi_l2(
            pred, torch.zeros_like(pred), masks, ("background", "organ"),
            torch.tensor([[True, True]]), torch.tensor([0.0, 1.0]),
        )
        self.assertEqual(stats["positive"]["loss"].item(), 0.0)
        stats["positive"]["loss"].backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_frequency_weights_have_unit_expected_mass(self):
        weights = build_class_frequency_weights([50, 25], 100, alpha=0.5)
        self.assertEqual(weights[0].item(), 0.0)
        expected_mass = (50 * weights[1] + 25 * weights[2]) / 100
        self.assertAlmostEqual(expected_mass.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
