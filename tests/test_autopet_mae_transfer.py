import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.mae_manifest import split_records_by_case
from models_mae_transfer import ArchitectureMatchedMAE
from OWT_models_orgslot import OrganSlotMaskedAutoencoderViT
from util.mae_transfer import load_mae_transfer_checkpoint


def _target_model(img_size=64):
    model_args = SimpleNamespace(
        LA=True,
        arch_version="v11",
        dataset_type="2D",
        token_factor=2,
        organ_token_total=4,
        fix_frame=0,
        temp_stride=0,
        loss_version=["L2"],
        text_encoding="None",
    )
    return OrganSlotMaskedAutoencoderViT(
        img_size=img_size,
        patch_size=16,
        embed_dim=48,
        depth=4,
        num_heads=4,
        decoder_embed_dim=48,
        decoder_depth=2,
        decoder_num_heads=4,
        model_args=model_args,
        slot_specs=[
            {"name": "background", "raw_class_id": 0},
            {"name": "organ", "raw_class_id": 1},
        ],
        slot_tg_depth=1,
        fusion_mode="linear_sqrt",
        fusion_reference_count=2,
        slot_head_type="linear",
    )


class AutoPETMAETransferTests(unittest.TestCase):
    def test_mae_forward_backward_is_finite(self):
        model = ArchitectureMatchedMAE(
            img_size=32,
            patch_size=16,
            embed_dim=48,
            encoder_depth=2,
            encoder_heads=4,
            decoder_dim=48,
            decoder_depth=2,
            decoder_heads=4,
        )
        images = torch.rand(2, 3, 32, 32)
        loss, prediction, mask = model(images, mask_ratio=0.75)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(prediction.shape), (2, 4, 768))
        self.assertEqual(tuple(mask.shape), (2, 4))
        loss.backward()
        self.assertIsNotNone(model.patch_embed.proj.weight.grad)

    def test_mae_can_overfit_one_tiny_batch(self):
        torch.manual_seed(3)
        model = ArchitectureMatchedMAE(
            img_size=32,
            patch_size=16,
            embed_dim=32,
            encoder_depth=1,
            encoder_heads=4,
            decoder_dim=32,
            decoder_depth=1,
            decoder_heads=4,
        )
        images = torch.rand(2, 3, 32, 32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(40):
            # Fix the masking pattern so this test measures learnability rather
            # than variation from seeing a different reconstruction target.
            torch.manual_seed(11)
            loss, _, _ = model(images, mask_ratio=0.75)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], 0.35 * losses[0])

    def test_patient_split_is_disjoint_and_deterministic(self):
        records = []
        for case in range(10):
            for slice_index in range(3):
                records.append(
                    {
                        "image_pth": "/tmp/case_{0}/case_{0}_{1}.jpg".format(
                            case, slice_index
                        )
                    }
                )
        train_a, val_a = split_records_by_case(records, 0.2, seed=7)
        train_b, val_b = split_records_by_case(records, 0.2, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        train_cases = {Path(row["image_pth"]).parent.name for row in train_a}
        val_cases = {Path(row["image_pth"]).parent.name for row in val_a}
        self.assertFalse(train_cases & val_cases)
        self.assertEqual(len(val_cases), 2)

    def test_encoder_only_transfer_leaves_shared_decoder_unchanged(self):
        source = ArchitectureMatchedMAE(
            img_size=32,
            patch_size=16,
            embed_dim=48,
            encoder_depth=2,
            encoder_heads=4,
            decoder_dim=48,
            decoder_depth=2,
            decoder_heads=4,
        )
        target = _target_model(img_size=64)
        decoder_before = {
            key: value.clone()
            for key, value in target.state_dict().items()
            if key.startswith(("decoder_blocks.", "decoder_norm.", "decoder_pred."))
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "mae.pth"
            torch.save({"model": source.state_dict()}, checkpoint_path)
            report = load_mae_transfer_checkpoint(target, checkpoint_path, "encoder")
        self.assertEqual(report["scope"], "encoder")
        self.assertTrue(
            torch.equal(
                source.state_dict()["blocks.0.attn.qkv.weight"],
                target.state_dict()["blocks1.0.attn.qkv.weight"],
            )
        )
        for key, value in decoder_before.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_incomplete_encoder_checkpoint_is_rejected(self):
        source = ArchitectureMatchedMAE(
            img_size=32,
            patch_size=16,
            embed_dim=48,
            encoder_depth=1,
            encoder_heads=4,
            decoder_dim=48,
            decoder_depth=2,
            decoder_heads=4,
        )
        target = _target_model(img_size=64)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "incomplete_mae.pth"
            torch.save({"model": source.state_dict()}, checkpoint_path)
            with self.assertRaisesRegex(RuntimeError, "every required target tensor"):
                load_mae_transfer_checkpoint(
                    target, checkpoint_path, "encoder_decoder"
                )

    def test_encoder_decoder_transfer_is_exact_and_slots_stay_new(self):
        source = ArchitectureMatchedMAE(
            img_size=32,
            patch_size=16,
            embed_dim=48,
            encoder_depth=2,
            encoder_heads=4,
            decoder_dim=48,
            decoder_depth=2,
            decoder_heads=4,
        )
        target = _target_model(img_size=64)
        slot_before = {
            key: value.clone()
            for key, value in target.state_dict().items()
            if key.startswith("slot_bank.")
        }
        pos_before = target.pos_embed.clone()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "mae.pth"
            torch.save(
                {
                    "model": source.state_dict(),
                    "epoch": 3,
                    "optimizer_updates": 12,
                    "input_size": 32,
                },
                checkpoint_path,
            )
            report = load_mae_transfer_checkpoint(
                target, checkpoint_path, "encoder_decoder"
            )
        self.assertEqual(report["source_input_size"], 32)
        self.assertEqual(report["target_input_size"], 64)
        self.assertGreater(report["loaded_tensor_count"], 0)
        self.assertTrue(
            torch.equal(
                source.state_dict()["blocks.0.attn.qkv.weight"],
                target.state_dict()["blocks1.0.attn.qkv.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                source.state_dict()["decoder_pred.weight"],
                target.state_dict()["decoder_pred.weight"],
            )
        )
        self.assertTrue(torch.equal(pos_before, target.pos_embed))
        for key, value in slot_before.items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
