import unittest
import torch
from datasets.orgslot_highres import OrganSlotHighResTransform


class FullFieldTest(unittest.TestCase):
    def sample(self, size):
        label = torch.zeros(1, size, size, dtype=torch.long)
        for ys, xs in [(slice(0, 20), slice(0, 20)), (slice(-20, None), slice(-20, None))]:
            label[:, ys, xs] = 4
        return {'image': label.expand(3, -1, -1).float() / 4, 'label': label}

    def test_full_field_keeps_both_edges_for_small_and_large_images(self):
        transform = OrganSlotHighResTransform(training=False, full_field=True)
        for size in (305, 501):
            result = transform(self.sample(size))
            self.assertEqual(result['label'].shape, (1, 448, 448))
            self.assertGreater(result['label'][0, :50, :50].sum().item(), 0)
            self.assertGreater(result['label'][0, -50:, -50:].sum().item(), 0)

    def test_old_default_still_rejects_small_image(self):
        with self.assertRaises(ValueError):
            OrganSlotHighResTransform(training=False)(self.sample(305))

    def test_roi_keeps_focus_at_edge(self):
        sample = self.sample(501)
        sample['label'][:, -20:, -20:] = 0
        sample.update(focus_class_id=4, roi_applied=True)
        transform = OrganSlotHighResTransform(full_field=True, flip_probability=0,
            rotation_probability=0, gamma_probability=0, photometric_operations=0)
        for seed in range(5):
            torch.manual_seed(seed)
            result = transform(sample)
            self.assertGreaterEqual(result['label'].eq(4).sum().item(), 400)

    def test_legacy_default_matches_explicit_false(self):
        a = OrganSlotHighResTransform()
        b = OrganSlotHighResTransform(full_field=False)
        torch.manual_seed(2)
        x = a(self.sample(501))
        torch.manual_seed(2)
        y = b(self.sample(501))
        self.assertTrue(torch.equal(x['image'], y['image']))
        self.assertTrue(torch.equal(x['label'], y['label']))
