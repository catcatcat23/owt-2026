import unittest
from collections import Counter
from types import SimpleNamespace

import torch

import OWT_models
import OWT_models_psem
from util.per_sample_mask_schedule import (
    MODE_BACKGROUND_ONLY,
    MODE_DIRECT_NEGATIVE,
    MODE_DIRECT_POSITIVE,
    MODE_LEAVE_ONE_OUT,
    MODE_ORGANS_ONLY,
    MODE_RANDOM_SUBSET,
    MODE_WHOLE,
    exhaustive_present_label_schedule,
    masked_reconstruction_target,
    present_class_mask,
    query_matched_schedule,
)


def model_args(dataset_type):
    is_3d = dataset_type == "3D"
    return SimpleNamespace(
        arch_version="v11",
        dataset_type=dataset_type,
        temp_stride=1 if is_3d else 0,
        fix_frame=4 if is_3d else 0,
        LA=True,
        organ_token_total=6,
        token_factor=2,
        num_classes_with_bg=3,
        loss_version=["L2"],
        text_encoding="None",
    )


def build_model(model_class, dataset_type="2D"):
    return model_class(
        img_size=32,
        patch_size=16,
        in_chans=3,
        embed_dim=32,
        depth=2,
        num_heads=4,
        decoder_embed_dim=32,
        decoder_depth=1,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=lambda dim: torch.nn.LayerNorm(dim, eps=1e-6),
        norm_pix_loss=False,
        model_args=model_args(dataset_type),
    )


class ScheduleTests(unittest.TestCase):
    def test_exhaustive_cycle_uses_only_present_classes(self):
        present = torch.tensor([[True, False, True, False, True]])
        sample_indices = torch.tensor([7])
        observed = set()

        for epoch in range(7):
            keep, drop, _, cycles = exhaustive_present_label_schedule(
                present, sample_indices, epoch
            )
            self.assertEqual(cycles.item(), 7)
            self.assertTrue(torch.equal(keep | drop, present))
            self.assertFalse(torch.any(keep & drop))
            self.assertEqual(keep[:, [1, 3]].sum().item(), 0)
            self.assertEqual(drop[:, [1, 3]].sum().item(), 0)
            self.assertGreater(keep.sum().item(), 0)
            observed.add(tuple(drop[0, [0, 2, 4]].tolist()))

        self.assertEqual(len(observed), 7)
        self.assertNotIn((True, True, True), observed)

    def test_batch_samples_receive_independent_masks(self):
        present = torch.tensor(
            [
                [True, True, False, False],
                [True, False, True, True],
            ]
        )
        keep, drop, _, cycles = exhaustive_present_label_schedule(
            present, torch.tensor([11, 23]), epoch=3
        )
        self.assertTrue(torch.equal(keep | drop, present))
        self.assertEqual(cycles.tolist(), [3, 7])
        self.assertTrue(torch.all(keep.sum(dim=1) > 0))

    def test_target_matches_each_samples_keep_mask(self):
        labels = torch.tensor(
            [
                [[[0, 1], [2, 1]]] * 3,
                [[[2, 2], [0, 1]]] * 3,
            ]
        )
        images = torch.ones(2, 3, 2, 2)
        keep = torch.tensor(
            [
                [True, False, True],
                [False, True, True],
            ]
        )
        target = masked_reconstruction_target(images, labels, keep)
        self.assertEqual(target[0, 0].tolist(), [[1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(target[1, 0].tolist(), [[1.0, 1.0], [0.0, 1.0]])

    def test_present_mask_supports_3d(self):
        labels = torch.zeros(2, 3, 4, 2, 2, dtype=torch.long)
        labels[0, :, 1, 0, 0] = 2
        labels[1, :, 2, 1, 1] = 1
        present = present_class_mask(labels, 3)
        self.assertEqual(
            present.tolist(),
            [[True, False, True], [True, True, False]],
        )



class PaddingTests(unittest.TestCase):
    def test_tgenc_padding_does_not_change_valid_outputs(self):
        torch.manual_seed(0)
        block = OWT_models_psem.PaddedBlockLA(
            dim=8, num_heads=2, mlp_ratio=2, qkv_bias=True
        ).eval()
        sample_short = torch.randn(1, 2, 8)
        sample_long = torch.randn(1, 4, 8)
        padded = torch.cat(
            [
                torch.nn.functional.pad(sample_short, (0, 0, 0, 2)),
                sample_long,
            ],
            dim=0,
        )
        valid = torch.tensor(
            [[True, True, False, False], [True, True, True, True]]
        )

        batched = block(padded, valid)
        separate = block(sample_short, torch.ones(1, 2, dtype=torch.bool))
        self.assertTrue(torch.allclose(batched[:1, :2], separate, atol=1e-6))
        self.assertEqual(batched[0, 2:].abs().sum().item(), 0.0)

    def test_aher_padding_does_not_change_output(self):
        torch.manual_seed(1)
        aher = OWT_models_psem.PaddedAHER(8, 6, 4, 5).eval()
        sample = torch.randn(1, 2, 8)
        padded = torch.nn.functional.pad(sample, (0, 0, 0, 3))
        valid = torch.tensor([[True, True, False, False, False]])

        padded_output, _ = aher(padded, valid)
        separate_output, _ = aher(
            sample, torch.ones(1, 2, dtype=torch.bool)
        )
        self.assertTrue(
            torch.allclose(padded_output, separate_output, atol=1e-6)
        )


class ModelTests(unittest.TestCase):
    def test_parameter_keys_and_shapes_match_original_owt(self):
        torch.manual_seed(2)
        baseline = build_model(OWT_models.MaskedAutoencoderViT)
        psem = build_model(OWT_models_psem.PSEMMaskedAutoencoderViT)
        baseline_state = baseline.state_dict()
        psem_state = psem.state_dict()

        self.assertEqual(set(baseline_state), set(psem_state))
        for key in baseline_state:
            self.assertEqual(baseline_state[key].shape, psem_state[key].shape)

    def _forward_backward(self, dataset_type):
        torch.manual_seed(3)
        model = build_model(
            OWT_models_psem.PSEMMaskedAutoencoderViT, dataset_type
        )
        if dataset_type == "2D":
            images = torch.rand(2, 3, 32, 32)
            labels = torch.zeros(2, 3, 32, 32, dtype=torch.long)
            labels[0, :, :16] = 1
            labels[1, :, 16:] = 2
        else:
            images = torch.rand(2, 3, 4, 32, 32)
            labels = torch.zeros(2, 3, 4, 32, 32, dtype=torch.long)
            labels[0, :, :, :16] = 1
            labels[1, :, :, 16:] = 2

        present = present_class_mask(labels, 3)
        keep, _, _, _, _ = query_matched_schedule(
            present, torch.tensor([0, 1]), epoch=0
        )
        target = masked_reconstruction_target(images, labels, keep)
        loss, pred, middle = model(
            images,
            middle={"image_target": target, "class_keep_mask": keep},
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pred.shape, images.shape)
        self.assertEqual(
            middle["valid_token_mask"].shape[0], images.shape[0]
        )
        loss.backward()
        self.assertIsNotNone(model.organ_embed.conv1.weight.grad)
        self.assertTrue(torch.isfinite(model.organ_embed.conv1.weight.grad).all())

    def test_2d_forward_backward(self):
        self._forward_backward("2D")

    def test_3d_forward_backward(self):
        self._forward_backward("3D")


class QueryMatchedScheduleTests(unittest.TestCase):
    def setUp(self):
        self.present = torch.tensor([[True, True, False, True, False]])
        self.sample_indices = torch.tensor([9])

    def test_cycle_has_expected_mode_weights(self):
        modes = []
        slots = []
        for epoch in range(20):
            _, _, slot, mode, _ = query_matched_schedule(
                self.present, self.sample_indices, epoch
            )
            slots.append(slot.item())
            modes.append(mode.item())

        self.assertEqual(set(slots), set(range(20)))
        self.assertEqual(
            Counter(modes),
            Counter({
                MODE_DIRECT_POSITIVE: 4,
                MODE_DIRECT_NEGATIVE: 4,
                MODE_LEAVE_ONE_OUT: 5,
                MODE_WHOLE: 2,
                MODE_BACKGROUND_ONLY: 1,
                MODE_ORGANS_ONLY: 1,
                MODE_RANDOM_SUBSET: 3,
            }),
        )

    def test_consecutive_samples_cover_all_slots(self):
        present = self.present.expand(20, -1).clone()
        _, _, slots, _, _ = query_matched_schedule(
            present, torch.arange(20), epoch=7
        )
        self.assertEqual(sorted(slots.tolist()), list(range(20)))

    def test_direct_positive_and_negative_targets(self):
        labels = torch.tensor([[[[0, 1], [1, 0]]] * 3])
        images = torch.ones(1, 3, 2, 2)
        present = present_class_mask(labels, 3)
        observed = {}

        for epoch in range(20):
            keep, _, _, modes, classes = query_matched_schedule(
                present, torch.tensor([4]), epoch
            )
            mode = modes.item()
            if mode in (MODE_DIRECT_POSITIVE, MODE_DIRECT_NEGATIVE):
                observed.setdefault(
                    mode,
                    (keep.clone(), classes.item(),
                     masked_reconstruction_target(images, labels, keep)),
                )

        positive_keep, positive_class, positive_target = observed[
            MODE_DIRECT_POSITIVE
        ]
        self.assertEqual(positive_class, 1)
        self.assertEqual(positive_keep.tolist(), [[False, True, False]])
        self.assertEqual(
            positive_target[0, 0].tolist(), [[0.0, 1.0], [1.0, 0.0]]
        )

        negative_keep, negative_class, negative_target = observed[
            MODE_DIRECT_NEGATIVE
        ]
        self.assertEqual(negative_class, 2)
        self.assertEqual(negative_keep.tolist(), [[False, False, True]])
        self.assertEqual(negative_target.abs().sum().item(), 0.0)

    def test_exact_inference_query_masks_are_scheduled(self):
        observed = {}
        for epoch in range(20):
            keep, _, _, modes, classes = query_matched_schedule(
                self.present, self.sample_indices, epoch
            )
            observed.setdefault(
                modes.item(), (keep[0].clone(), classes.item())
            )

        leave_keep, leave_class = observed[MODE_LEAVE_ONE_OUT]
        self.assertGreater(leave_class, 0)
        self.assertEqual(leave_keep.sum().item(), 4)
        self.assertFalse(leave_keep[leave_class].item())
        self.assertTrue(torch.all(observed[MODE_WHOLE][0]))
        self.assertEqual(
            observed[MODE_BACKGROUND_ONLY][0].tolist(),
            [True, False, False, False, False],
        )
        self.assertEqual(
            observed[MODE_ORGANS_ONLY][0].tolist(),
            [False, True, True, True, True],
        )
        self.assertGreater(observed[MODE_RANDOM_SUBSET][0].sum().item(), 0)


class QueryMatchedTinyOverfitTests(unittest.TestCase):
    def test_positive_and_negative_queries_overfit(self):
        torch.manual_seed(17)
        model = build_model(OWT_models_psem.PSEMMaskedAutoencoderViT)
        model.train()
        images = torch.rand(2, 3, 32, 32)
        labels = torch.zeros(2, 3, 32, 32, dtype=torch.long)
        labels[0, :, :16] = 1
        keep = torch.tensor([
            [False, True, False],
            [False, True, False],
        ])
        target = masked_reconstruction_target(images, labels, keep)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3e-3, weight_decay=0.0
        )

        with torch.no_grad():
            initial_loss, initial_pred, _ = model(
                images,
                middle={"image_target": target, "class_keep_mask": keep},
            )
            initial_negative_energy = initial_pred[1].square().mean().item()

        for _ in range(25):
            optimizer.zero_grad()
            loss, _, _ = model(
                images,
                middle={"image_target": target, "class_keep_mask": keep},
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            final_loss, final_pred, _ = model(
                images,
                middle={"image_target": target, "class_keep_mask": keep},
            )
            final_negative_energy = final_pred[1].square().mean().item()

        self.assertLess(final_loss.item(), initial_loss.item() * 0.8)
        self.assertLess(
            final_negative_energy, initial_negative_energy * 0.8
        )


if __name__ == "__main__":
    unittest.main()
