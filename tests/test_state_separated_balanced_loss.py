import unittest

import torch

from util.state_separated_balanced_loss import state_separated_roi_l2


class StateSeparatedRoiL2Test(unittest.TestCase):
    def test_kept_and_removed_classes_are_routed_separately(self):
        target = torch.zeros((2, 3, 1, 2))
        pred = torch.zeros_like(target, requires_grad=True)
        label = torch.zeros((2, 3, 1, 2), dtype=torch.long)
        label[:, :, :, 0] = 1
        label[:, :, :, 1] = 2
        with torch.no_grad():
            pred[0, :, :, 0] = 1.0
            pred[0, :, :, 1] = 2.0
            pred[1, :, :, 0] = 3.0
            pred[1, :, :, 1] = 4.0

        stats = state_separated_roi_l2(
            pred,
            target,
            label,
            class_keep_mask=torch.tensor([
                [True, True, False],
                [True, False, True],
            ]),
            class_weights=torch.tensor([0.0, 2.0, 3.0]),
        )

        # Positive: (2 * 1^2 + 3 * 4^2) / B = 25.
        self.assertAlmostEqual(stats["positive"]["loss"].item(), 25.0)
        # Removed monitoring is unweighted: (2^2 + 3^2) / B = 6.5.
        self.assertAlmostEqual(stats["removed"]["loss"].item(), 6.5)
        self.assertEqual(stats["positive"]["class_counts"].tolist(), [0, 1, 1])
        self.assertEqual(stats["removed"]["class_counts"].tolist(), [0, 1, 1])
        self.assertAlmostEqual(stats["positive"]["mass"].item(), 2.5)
        self.assertAlmostEqual(stats["removed"]["mass"].item(), 1.0)
        stats["positive"]["loss"].backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.count_nonzero(pred.grad[0, :, :, 0]))
        self.assertTrue(torch.count_nonzero(pred.grad[1, :, :, 1]))
        self.assertEqual(torch.count_nonzero(pred.grad[0, :, :, 1]).item(), 0)
        self.assertEqual(torch.count_nonzero(pred.grad[1, :, :, 0]).item(), 0)

    def test_absent_and_background_classes_are_excluded(self):
        pred = torch.ones((1, 3, 2, 2), requires_grad=True)
        target = torch.zeros_like(pred)
        label = torch.zeros((1, 3, 2, 2), dtype=torch.long)
        stats = state_separated_roi_l2(
            pred,
            target,
            label,
            class_keep_mask=torch.tensor([[False, True, False]]),
            class_weights=torch.tensor([0.0, 1.0, 2.0]),
        )
        self.assertEqual(stats["positive"]["valid_samples"].item(), 0)
        self.assertEqual(stats["removed"]["valid_samples"].item(), 0)
        self.assertEqual(stats["positive"]["loss"].item(), 0.0)
        self.assertEqual(stats["removed"]["loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(stats["positive"]["loss"]))
        stats["positive"]["loss"].backward()
        torch.testing.assert_close(pred.grad, torch.zeros_like(pred))

    def test_short_window_3d(self):
        pred = torch.ones((1, 3, 4, 2, 2), requires_grad=True)
        target = torch.zeros_like(pred)
        label = torch.ones((1, 3, 4, 2, 2), dtype=torch.long)
        stats = state_separated_roi_l2(
            pred,
            target,
            label,
            class_keep_mask=torch.tensor([[True, True]]),
            class_weights=torch.tensor([0.0, 1.0]),
        )
        self.assertAlmostEqual(stats["positive"]["loss"].item(), 1.0)
        self.assertEqual(stats["positive"]["class_counts"].tolist(), [0, 1])

    def test_class_keep_mask_shape_and_dtype_are_checked(self):
        pred = torch.zeros((1, 3, 2, 2))
        target = torch.zeros_like(pred)
        label = torch.zeros_like(pred, dtype=torch.long)
        weights = torch.tensor([0.0, 1.0])
        with self.assertRaises(ValueError):
            state_separated_roi_l2(
                pred, target, label, torch.ones(1, 3, dtype=torch.bool), weights
            )
        with self.assertRaises(ValueError):
            state_separated_roi_l2(
                pred, target, label, torch.ones(1, 2), weights
            )


if __name__ == "__main__":
    unittest.main()
