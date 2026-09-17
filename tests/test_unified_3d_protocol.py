import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools import eval_common8_orgslot_unified_3d as unified
from tools.eval_common8_orgslot_reconstruction_threshold import (
    new_counter, postprocess_slice, update_counter,
)


class ProtocolTests(unittest.TestCase):
    def test_calibration_prefix_and_case_separation(self):
        train = [('train', index) for index in range(8)]
        test = [('test', 0)]
        self.assertEqual(unified.scoring_keys('calibrate', train, test, 3), set(train[:3]))
        with self.assertRaises(ValueError):
            unified.scoring_keys('calibrate', train, [('train', 99)], 3)
        with self.assertRaises(ValueError):
            unified.scoring_keys('head', test, test + [('test', 1)], 3)

    def test_slice_counts_identical_to_2d(self):
        score = np.zeros((16, 16), dtype=np.float32)
        score[2:10, 2:10] = 0.8
        score[0, 0] = 0.5
        target = score > 0.7
        counters = {'binary_raw': {1: {}}, 'binary_post': {1: {}}}
        unified.count_slice(counters, 'case', 1, score, target, 0.5, 'binary')
        for mode, prediction in [('raw', score > 0.5),
                                 ('post', postprocess_slice(score > 0.5, 20, 1))]:
            expected = new_counter()
            update_counter(expected, prediction, target)
            self.assertEqual(counters['binary_' + mode][1]['case'], expected)

    def test_reconstruction_complement_and_channel_clamp(self):
        class Model:
            slot_names = ['background', 'organ']

            def forward_encoder(self, image):
                return image, None

        image = torch.tensor([1., 3.]).reshape(1, 2, 1, 1, 1)
        masks = []

        def decode(model, canvases, keep):
            masks.append(keep.clone())
            return torch.full_like(image, 2.)

        with patch.object(unified, 'slot_canvases', return_value={}), \
                patch.object(unified, 'reconstruct_from_keep', side_effect=decode):
            scores = unified.recon_scores(Model(), image, [1], {1: 1})
        self.assertTrue(torch.equal(masks[0], torch.tensor([[False, True]])))
        self.assertTrue(torch.equal(masks[1], ~masks[0]))
        self.assertEqual(scores['direct'].item(), 2.)
        self.assertEqual(scores['indirect'].item(), 0.5)


if __name__ == '__main__':
    unittest.main()
