import unittest

import torch

from util.frequency_balanced_loss import (
    build_class_frequency_weights,
    frequency_balanced_roi_l2,
)


class FrequencyWeightTest(unittest.TestCase):
    def test_alpha_one_matches_inverse_positive_frequency(self):
        weights = build_class_frequency_weights(
            [4, 1], dataset_size=4, alpha=1.0, max_weight_ratio=10.0
        )
        torch.testing.assert_close(weights, torch.tensor([0.0, 0.5, 2.0]))
        expected_mass = (4 * weights[1] + weights[2]) / 4
        self.assertAlmostEqual(expected_mass.item(), 1.0)

    def test_ratio_cap_is_applied_before_scale_normalization(self):
        weights = build_class_frequency_weights(
            [100, 25], dataset_size=100, alpha=0.5, max_weight_ratio=1.5
        )
        self.assertAlmostEqual((weights[2] / weights[1]).item(), 1.5)
        expected_mass = (100 * weights[1] + 25 * weights[2]) / 100
        self.assertAlmostEqual(expected_mass.item(), 1.0)

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            build_class_frequency_weights([4, 0], dataset_size=4)
        with self.assertRaises(ValueError):
            build_class_frequency_weights([5, 1], dataset_size=4)


class FrequencyBalancedRoiL2Test(unittest.TestCase):
    def test_dataset_frequency_balances_class_means(self):
        target = torch.zeros((4, 3, 1, 2))
        pred = torch.zeros_like(target, requires_grad=True)
        label = torch.zeros((4, 3, 1, 2), dtype=torch.long)

        label[:, :, :, 0] = 1
        label[0, :, :, 1] = 2
        with torch.no_grad():
            pred[:, :, :, 0] = 1.0
            pred[0, :, :, 1] = 2.0

        weights = build_class_frequency_weights(
            [4, 1], dataset_size=4, alpha=1.0, max_weight_ratio=10.0
        )
        loss, per_class, counts, valid_samples, weighted_mass = (
            frequency_balanced_roi_l2(pred, target, label, weights)
        )

        # Equivalent full-dataset objective: (class-1 mean + class-2 mean) / 2.
        self.assertAlmostEqual(loss.item(), (1.0 + 4.0) / 2.0)
        self.assertAlmostEqual(per_class[1].item(), 1.0)
        self.assertAlmostEqual(per_class[2].item(), 4.0)
        self.assertEqual(counts.tolist(), [3, 4, 1])
        self.assertEqual(valid_samples.item(), 4)
        self.assertAlmostEqual(weighted_mass.item(), 1.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)

    def test_short_window_3d_shape(self):
        pred = torch.ones((1, 3, 4, 2, 2), requires_grad=True)
        target = torch.zeros_like(pred)
        label = torch.ones((1, 3, 4, 2, 2), dtype=torch.long)
        weights = build_class_frequency_weights(
            [1], dataset_size=1, alpha=0.5, max_weight_ratio=4.0
        )
        loss, per_class, counts, valid_samples, weighted_mass = (
            frequency_balanced_roi_l2(pred, target, label, weights)
        )
        self.assertAlmostEqual(loss.item(), 1.0)
        self.assertAlmostEqual(per_class[1].item(), 1.0)
        self.assertEqual(counts.tolist(), [0, 1])
        self.assertEqual(valid_samples.item(), 1)
        self.assertAlmostEqual(weighted_mass.item(), 1.0)


if __name__ == "__main__":
    unittest.main()
