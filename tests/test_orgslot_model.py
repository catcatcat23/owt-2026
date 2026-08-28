import copy
from types import SimpleNamespace
import unittest

import torch
import torch.nn as nn

from losses_orgslot import base_segmentation_loss
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


def tiny_model(
    dataset_type="2D",
    specs=None,
    slot_tg_depth=1,
    fusion_mode="post_layernorm",
    img_size=32,
    fusion_reference_count=None,
    slot_head_type="linear",
    slot_head_channels=128,
):
    if specs is None:
        specs = [
            {"name": "background", "raw_class_id": 0},
            {"name": "kidney", "raw_class_id": 1},
            {"name": "spleen", "raw_class_id": 2},
        ]
    return OrganSlotMaskedAutoencoderViT(
        img_size=img_size,
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
        fusion_mode=fusion_mode,
        fusion_reference_count=fusion_reference_count,
        slot_head_type=slot_head_type,
        slot_head_channels=slot_head_channels,
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

    def test_336_resolution_forward_backward(self):
        torch.manual_seed(3)
        model = tiny_model(img_size=336, fusion_mode="linear_sqrt")
        images = torch.rand(1, 3, 336, 336)
        keep = torch.tensor([[1, 1, 1]], dtype=torch.bool)
        output = model(images, keep)
        self.assertEqual(output["reconstruction"].shape, images.shape)
        self.assertEqual(model.patch_embed.num_patches, 21 * 21)
        loss = output["reconstruction"].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_multiscale_head_runs_only_selected_rows(self):
        model = tiny_model(
            slot_head_type="multiscale_conv", slot_head_channels=32
        )
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        seen_batch_sizes = {}
        seen_slot_batch_sizes = {}
        handles = []
        for index, name in enumerate(model.slot_names):
            def record(_module, inputs, _output, slot_name=name):
                seen_batch_sizes[slot_name] = inputs[0].shape[0]
            def record_slot(_module, inputs, _output, slot_name=name):
                seen_slot_batch_sizes[slot_name] = inputs[0].shape[0]
            slot = model.slot_bank.get_slot(name)
            handles.append(slot.head.register_forward_hook(record))
            handles.append(slot.collector.register_forward_hook(record_slot))
        output = model(
            images,
            slot_keep_mask=keep,
            head_compute_mask=keep,
            decode_reconstruction=False,
        )
        for handle in handles:
            handle.remove()
        for index, name in enumerate(model.slot_names):
            expected = int(keep[:, index].sum())
            self.assertEqual(seen_batch_sizes[name], expected)
            self.assertEqual(seen_slot_batch_sizes[name], expected)
            dropped_rows = ~keep[:, index]
            self.assertTrue(torch.equal(
                output["calibrated_logits"][name][dropped_rows],
                torch.zeros_like(output["calibrated_logits"][name][dropped_rows]),
            ))
            self.assertEqual(
                output["calibrated_logits"][name].shape, (2, 1, 32, 32)
            )

        masks = {
            name: torch.zeros(2, 1, 32, 32)
            for name in model.slot_names
        }
        masks["background"].fill_(1)
        loss, _ = base_segmentation_loss(
            output["calibrated_logits"],
            masks,
            model.slot_names,
            keep,
            background_weight=0,
            loss_type="focal",
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(
            model.slot_bank.get_slot("kidney").head.proj.weight.grad
        )

    def test_query_dot_uses_shared_pixels_and_slot_queries(self):
        torch.manual_seed(5)
        model = tiny_model(slot_head_type="query_dot", slot_head_channels=16)
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        output = model(
            images,
            slot_keep_mask=keep,
            head_compute_mask=keep,
            decode_reconstruction=False,
        )
        for index, name in enumerate(model.slot_names):
            logits = output["calibrated_logits"][name]
            self.assertEqual(logits.shape, (2, 1, 32, 32))
            dropped_rows = ~keep[:, index]
            self.assertTrue(torch.equal(
                logits[dropped_rows], torch.zeros_like(logits[dropped_rows])
            ))

        loss = sum(
            logits.square().mean()
            for logits in output["calibrated_logits"].values()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            model.pixel_query_decoder.pixel_proj.weight.grad.abs().sum(), 0
        )
        self.assertGreater(model.patch_embed.proj.weight.grad.abs().sum(), 0)
        for index, name in enumerate(model.slot_names):
            slot = model.slot_bank.get_slot(name)
            if keep[:, index].any():
                self.assertGreater(slot.head.embedding.grad.abs().sum(), 0)
            self.assertTrue(all(
                parameter.grad is None for parameter in slot.aher.parameters()
            ))
        self.assertIsNone(model.decoder_pred.weight.grad)

    def test_multi_query_uses_all_token_queries(self):
        torch.manual_seed(6)
        model = tiny_model(
            slot_head_type="multi_query_dot", slot_head_channels=16
        )
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool)
        output = model(
            images,
            slot_keep_mask=keep,
            head_compute_mask=keep,
            decode_reconstruction=False,
        )
        for index, name in enumerate(model.slot_names):
            logits = output["calibrated_logits"][name]
            self.assertEqual(logits.shape, (2, 1, 32, 32))
            dropped_rows = ~keep[:, index]
            self.assertTrue(torch.equal(
                logits[dropped_rows], torch.zeros_like(logits[dropped_rows])
            ))

        loss = sum(
            logits.square().mean()
            for logits in output["calibrated_logits"].values()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            model.pixel_query_decoder.query_proj.weight.grad.abs().sum(), 0
        )
        self.assertGreater(
            model.pixel_query_decoder.pixel_proj.weight.grad.abs().sum(), 0
        )
        for index, name in enumerate(model.slot_names):
            slot = model.slot_bank.get_slot(name)
            if keep[:, index].any():
                self.assertGreater(slot.head.embedding.grad.abs().sum(), 0)
            self.assertTrue(all(
                parameter.grad is None for parameter in slot.aher.parameters()
            ))
        self.assertIsNone(model.decoder_pred.weight.grad)

    def test_multi_query_matches_single_query_for_identical_tokens(self):
        torch.manual_seed(7)
        single = tiny_model(
            slot_head_type="query_dot", slot_head_channels=16
        ).pixel_query_decoder
        multi = tiny_model(
            slot_head_type="multi_query_dot", slot_head_channels=16
        ).pixel_query_decoder
        multi.load_state_dict(single.state_dict(), strict=True)
        pixels = torch.randn(2, 16, 8, 8)
        one_token = torch.randn(2, 1, 32)
        tokens = one_token.expand(-1, 4, -1).clone()
        identity = torch.randn(16)
        expected = single.forward_mask(pixels, tokens, identity, (32, 32))
        actual = multi.forward_mask(pixels, tokens, identity, (32, 32))
        self.assertTrue(torch.allclose(
            actual, expected, atol=1e-6, rtol=1e-6
        ))


    def test_decode_heads_false_skips_every_head(self):
        model = tiny_model(slot_head_type="multiscale_conv")
        output = model(
            torch.rand(1, 3, 32, 32),
            decode_heads=False,
            decode_reconstruction=True,
        )
        self.assertEqual(output["slot_logits"], {})
        self.assertEqual(output["calibrated_logits"], {})
        self.assertIn("reconstruction", output)

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

    def test_3d_multiscale_head_forward_backward_shapes(self):
        torch.manual_seed(4)
        model = tiny_model(
            "3D", slot_head_type="multiscale_conv", slot_head_channels=32
        )
        images = torch.rand(1, 3, 4, 32, 32)
        output = model(images, decode_reconstruction=False)
        for logits in output["slot_logits"].values():
            self.assertEqual(logits.shape, (1, 1, 4, 32, 32))
        loss = sum(logits.mean() for logits in output["slot_logits"].values())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        head = model.slot_bank.get_slot("kidney").head
        self.assertIsNotNone(head.proj.weight.grad)

    def test_fixed_fusion_2d_and_3d_forward_backward(self):
        for dataset_type in ("2D", "3D"):
            with self.subTest(dataset_type=dataset_type):
                model = tiny_model(
                    dataset_type, fusion_mode="linear_fixed_sqrt"
                )
                shape = (1, 3, 32, 32)
                if dataset_type == "3D":
                    shape = (1, 3, 4, 32, 32)
                images = torch.rand(*shape)
                output = model(images)
                self.assertEqual(output["reconstruction"].shape, images.shape)
                output["reconstruction"].mean().backward()
                self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_all_head_probe_skips_decoder_and_freezes_feature_path(self):
        model = tiny_model()
        report = model.freeze_for_all_heads(train_calibration=True)
        expected = 0
        for name in model.slot_names:
            slot = model.slot_bank.get_slot(name)
            expected += sum(parameter.numel() for parameter in slot.head.parameters())
            expected += slot.calibration_scale.numel()
            expected += slot.calibration_bias.numel()
        self.assertEqual(report["trainable"], expected)

        images = torch.rand(2, 3, 32, 32)
        output = model(images, decode_reconstruction=False)
        self.assertNotIn("reconstruction", output)
        loss = sum(
            logits.mean() for logits in output["calibrated_logits"].values()
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
            else:
                self.assertIsNone(parameter.grad, name)

    def test_all_slot_segmentation_updates_dropped_slot_and_backbone(self):
        model = tiny_model()
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool)
        output = model(
            images,
            slot_keep_mask=keep,
            decode_reconstruction=False,
        )
        masks = {
            name: torch.zeros(2, 1, 32, 32)
            for name in model.slot_names
        }
        masks["background"].fill_(1)
        masks["kidney"][:, :, :16] = 1
        masks["spleen"][:, :, 16:] = 1
        supervision = torch.ones_like(keep)
        loss, _ = base_segmentation_loss(
            output["calibrated_logits"],
            masks,
            model.slot_names,
            supervision,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        dropped_head = model.slot_bank.get_slot("kidney").head.proj.weight
        before = dropped_head.detach().clone()
        optimizer.zero_grad()
        loss.backward()
        self.assertGreater(dropped_head.grad.abs().sum(), 0)
        dropped_aher = model.slot_bank.get_slot("kidney").aher
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in dropped_aher.parameters()
        ))
        self.assertGreater(model.patch_embed.proj.weight.grad.abs().sum(), 0)
        self.assertIsNone(model.decoder_pred.weight.grad)
        optimizer.step()
        self.assertFalse(torch.equal(before, dropped_head.detach()))

    def test_retained_segmentation_does_not_update_dropped_head(self):
        model = tiny_model()
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [1, 0, 1]], dtype=torch.bool)
        output = model(
            images,
            slot_keep_mask=keep,
            decode_reconstruction=False,
        )
        masks = {
            name: torch.zeros(2, 1, 32, 32)
            for name in model.slot_names
        }
        loss, _ = base_segmentation_loss(
            output["calibrated_logits"], masks, model.slot_names, keep
        )
        loss.backward()
        gradient = model.slot_bank.get_slot("kidney").head.proj.weight.grad
        self.assertTrue(
            gradient is None or gradient.abs().sum().item() == 0
        )

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

    def test_linear_sqrt_fusion_bypasses_post_layernorm(self):
        model = tiny_model(fusion_mode="linear_sqrt")
        canvases = {
            "background": torch.randn(2, 4, 32),
            "kidney": torch.randn(2, 4, 32),
            "spleen": torch.randn(2, 4, 32),
        }
        keep = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
        expected = torch.stack([
            (canvases["background"][0] + canvases["kidney"][0])
            / (2 ** 0.5),
            (canvases["kidney"][1] + canvases["spleen"][1])
            / (2 ** 0.5),
        ])
        actual = model.fuse_canvases(canvases, keep)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model.fusion_norm.parameters()
        ))

    def test_linear_fixed_sqrt_uses_stable_reference_count(self):
        model = tiny_model(
            fusion_mode="linear_fixed_sqrt", fusion_reference_count=3
        )
        canvases = {
            "background": torch.randn(2, 4, 32),
            "kidney": torch.randn(2, 4, 32),
            "spleen": torch.randn(2, 4, 32),
        }
        keep_all = torch.ones(2, 3, dtype=torch.bool)
        keep_without_kidney = keep_all.clone()
        keep_without_kidney[:, 1] = False
        fused_all = model.fuse_canvases(canvases, keep_all)
        fused_without = model.fuse_canvases(canvases, keep_without_kidney)
        expected_delta = canvases["kidney"] / (3 ** 0.5)
        self.assertTrue(torch.allclose(
            fused_all - fused_without, expected_delta, rtol=1e-6, atol=1e-6
        ))

        reference_count = model.fusion_reference_count
        model.append_slot("liver", 4, init_from="background")
        self.assertEqual(model.fusion_reference_count, reference_count)
        self.assertEqual(model.parameter_report()["fusion_reference_count"], 3)

    def test_fixed_fusion_does_not_change_slot_head_logits(self):
        dynamic = tiny_model(fusion_mode="linear_sqrt").eval()
        fixed = tiny_model(fusion_mode="linear_fixed_sqrt").eval()
        fixed.load_state_dict(dynamic.state_dict(), strict=True)
        images = torch.rand(2, 3, 32, 32)
        keep = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.bool)
        with torch.no_grad():
            dynamic_output = dynamic(images, keep)
            fixed_output = fixed(images, keep)
        for name in dynamic.slot_names:
            self.assertTrue(torch.equal(
                dynamic_output["slot_logits"][name],
                fixed_output["slot_logits"][name],
            ))
        self.assertFalse(torch.equal(
            dynamic_output["canvas"], fixed_output["canvas"]
        ))

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
