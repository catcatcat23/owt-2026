import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from OWT_models_orgslot import OrganSlotMaskedAutoencoderViT
from util.checkpoint_orgslot import (
    compare_parameter_hashes,
    hash_frozen_parameters,
)


def model_args(dataset_type="2D", slot_count=3):
    return SimpleNamespace(
        LA=True,
        arch_version="v1",
        dataset_type=dataset_type,
        token_factor=2,
        organ_token_total=2 * slot_count,
        fix_frame=4,
        temp_stride=1,
        loss_version="L2",
        text_encoding="None",
    )


def tiny_model(dataset_type="2D", specs=None, slot_tg_depth=1):
    if specs is None:
        specs = [
            {"name": "background", "raw_class_id": 0},
            {"name": "kidney", "raw_class_id": 1},
            {"name": "spleen", "raw_class_id": 2},
        ]
    return OrganSlotMaskedAutoencoderViT(
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
        norm_layer=nn.LayerNorm,
        model_args=model_args(dataset_type, len(specs)),
        slot_specs=specs,
        slot_tg_depth=slot_tg_depth,
    )


class OrganSlotModelTests(unittest.TestCase):
    def test_2d_forward_backward_shapes(self):
        torch.manual_seed(1)
        model = tiny_model()
        images = torch.rand(2, 3, 32, 32, requires_grad=True)
        keep = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        output = model(images, keep, return_diagnostics=True)
        self.assertEqual(output["reconstruction"].shape, images.shape)
        for name in model.slot_names:
            self.assertEqual(output["slot_logits"][name].shape, (2, 1, 32, 32))
            self.assertEqual(output["slot_tokens"][name].shape, (2, 2, 32))
            self.assertEqual(output["slot_canvases"][name].shape, (2, 4, 32))
        loss = output["reconstruction"].mean()
        loss += sum(value.mean() for value in output["slot_logits"].values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_3d_forward_backward_shapes(self):
        torch.manual_seed(2)
        model = tiny_model("3D")
        images = torch.rand(1, 3, 4, 32, 32)
        output = model(images)
        self.assertEqual(output["reconstruction"].shape, images.shape)
        for logits in output["slot_logits"].values():
            self.assertEqual(logits.shape, (1, 1, 4, 32, 32))
        output["reconstruction"].mean().backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_per_sample_fusion_is_exact_and_isolated(self):
        model = tiny_model()
        canvases = {
            "background": torch.ones(2, 4, 32),
            "kidney": torch.full((2, 4, 32), 2.0),
            "spleen": torch.full((2, 4, 32), 4.0),
        }
        keep = torch.tensor([[1, 1, 0], [0, 0, 1]], dtype=torch.bool)
        actual = model.fuse_canvases(canvases, keep)
        raw = torch.stack([
            (canvases["background"][0] + canvases["kidney"][0]) / (2 ** 0.5),
            canvases["spleen"][1],
        ])
        self.assertTrue(torch.allclose(actual, model.fusion_norm(raw)))
        keep_changed = keep.clone()
        keep_changed[1] = torch.tensor([1, 1, 1])
        changed = model.fuse_canvases(canvases, keep_changed)
        self.assertTrue(torch.equal(actual[0], changed[0]))
        with self.assertRaisesRegex(ValueError, "retain at least one"):
            model.fuse_canvases(canvases, torch.zeros_like(keep))

    def test_append_copy_preserves_all_existing_tensors(self):
        model = tiny_model()
        with torch.no_grad():
            background_slot = model.slot_bank.get_slot("background")
            background_slot.calibration_scale.fill_(2.0)
            background_slot.calibration_bias.fill_(3.0)
        before = copy.deepcopy(model.state_dict())
        model.append_slot("liver", 4, init_from="background")
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        background = model.slot_bank.get_slot("background").state_dict()
        liver = model.slot_bank.get_slot("liver").state_dict()
        for key in background:
            if key not in {"calibration_scale", "calibration_bias"}:
                self.assertTrue(torch.equal(background[key], liver[key]), key)
        self.assertEqual(float(liver["calibration_scale"]), 1.0)
        self.assertEqual(float(liver["calibration_bias"]), 0.0)

    def test_incremental_step_changes_only_new_slot(self):
        model = tiny_model()
        model.append_slot("liver", 4, init_from="background")
        report = model.freeze_for_incremental(
            old_slots=("kidney", "spleen"),
            new_slots=("liver",),
            background_policy="frozen",
        )
        self.assertEqual(report["trainable"], report["per_slot"]["liver"]["trainable"])
        before = hash_frozen_parameters(model)
        liver_before = {
            name: value.detach().clone()
            for name, value in model.slot_bank.get_slot("liver").named_parameters()
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        images = torch.rand(2, 3, 32, 32)
        frozen_logits_before = {
            name: value.detach().clone()
            for name, value in model(images)["slot_logits"].items()
            if name != "liver"
        }
        loss = model(images)["slot_logits"]["liver"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(compare_parameter_hashes(before, hash_frozen_parameters(model))["match"])
        self.assertTrue(any(
            not torch.equal(value, dict(model.slot_bank.get_slot("liver").named_parameters())[name])
            for name, value in liver_before.items()
        ))
        frozen_logits_after = model(images)["slot_logits"]
        for name, value in frozen_logits_before.items():
            self.assertTrue(torch.equal(value, frozen_logits_after[name]), name)

    def test_incremental_baseline_trainable_scopes(self):
        model = tiny_model()
        model.append_slot("liver", 4, init_from="background")
        report = model.freeze_for_head_only("liver")
        liver = model.slot_bank.get_slot("liver")
        expected = sum(parameter.numel() for parameter in liver.head.parameters())
        self.assertEqual(report["trainable"], expected)
        self.assertFalse(liver.calibration_scale.requires_grad)
        self.assertTrue(all(
            parameter.requires_grad for parameter in liver.head.parameters()
        ))
        report = model.unfreeze_all()
        self.assertEqual(report["trainable"], report["total"])


if __name__ == "__main__":
    unittest.main()
