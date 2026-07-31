import unittest
from types import SimpleNamespace

import torch

import OWT_models
import OWT_models_lossbalance_v3


def model_args(dataset_type):
    is_3d = dataset_type == '3D'
    return SimpleNamespace(
        arch_version='v11',
        dataset_type=dataset_type,
        temp_stride=1 if is_3d else 0,
        fix_frame=4 if is_3d else 0,
        LA=True,
        organ_token_total=6,
        token_factor=2,
        num_classes_with_bg=3,
        loss_version=['L2'],
        text_encoding='None',
        positive_roi_loss_weight=0.25,
        roi_class_weights=[0.0, 1.0, 1.0],
    )


def build_model(model_class, dataset_type='2D'):
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


class LossBalanceV3ModelTest(unittest.TestCase):
    def test_state_dict_keys_and_shapes_match_original_owt(self):
        baseline = build_model(OWT_models.MaskedAutoencoderViT)
        v3 = build_model(
            OWT_models_lossbalance_v3.StateSeparatedLossMaskedAutoencoderViT
        )
        baseline_state = baseline.state_dict()
        v3_state = v3.state_dict()
        self.assertEqual(set(baseline_state), set(v3_state))
        for key in baseline_state:
            self.assertEqual(baseline_state[key].shape, v3_state[key].shape)
        self.assertNotIn('roi_class_weights', v3_state)

    def _forward_backward(self, dataset_type):
        torch.manual_seed(7)
        model = build_model(
            OWT_models_lossbalance_v3.StateSeparatedLossMaskedAutoencoderViT,
            dataset_type,
        )
        if dataset_type == '2D':
            images = torch.rand(2, 3, 32, 32)
            labels = torch.zeros(2, 3, 32, 32, dtype=torch.long)
            labels[0, :, :16] = 1
            labels[1, :, 16:] = 2
        else:
            images = torch.rand(2, 3, 4, 32, 32)
            labels = torch.zeros(2, 3, 4, 32, 32, dtype=torch.long)
            labels[0, :, :, :16] = 1
            labels[1, :, :, 16:] = 2

        target = images.clone()
        target[labels == 2] = 0
        loss, pred, middle = model(
            images,
            mask_ratio=0.5,
            middle={
                'image_target': target,
                'random_selected_class': [2],
                'label': labels,
                'class_keep_mask': torch.tensor(
                    [[True, True, False], [True, True, False]]
                ),
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(pred.shape, images.shape)
        self.assertTrue(torch.isfinite(middle['positive_roi_loss']))
        self.assertTrue(torch.isfinite(middle['positive_weighted_mass']))
        loss.backward()
        grad = model.organ_embed.conv1.weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_tiny_batch_overfits(self):
        torch.manual_seed(11)
        model = build_model(
            OWT_models_lossbalance_v3.StateSeparatedLossMaskedAutoencoderViT
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        images = torch.rand(1, 3, 32, 32)
        labels = torch.ones(1, 3, 32, 32, dtype=torch.long)
        labels[:, :, :, 16:] = 2
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            loss, _, _ = model(
                images,
                mask_ratio=0.0,
                middle={
                    'image_target': images,
                    'random_selected_class': [],
                    'label': labels,
                    'class_keep_mask': torch.ones(
                        1, 3, dtype=torch.bool
                    ),
                },
            )
            self.assertTrue(torch.isfinite(loss))
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0] * 0.8)

    def test_2d_forward_backward(self):
        self._forward_backward('2D')

    def test_3d_forward_backward(self):
        self._forward_backward('3D')


if __name__ == '__main__':
    unittest.main()
