import tempfile
import unittest

import torch

import OWT_models
from test_orgslot_model import model_args, tiny_model
from util.checkpoint_orgslot import (
    load_orgslot_base_checkpoint,
    load_original_owt_initialization,
    save_orgslot_checkpoint,
    visible_slot_transfer,
)


class CheckpointTests(unittest.TestCase):
    def test_original_owt_forward_remains_runnable(self):
        args = model_args("2D", slot_count=3)
        original = OWT_models.MaskedAutoencoderViT(
            img_size=32, patch_size=16, in_chans=3,
            embed_dim=32, depth=2, num_heads=4,
            decoder_embed_dim=32, decoder_depth=1, decoder_num_heads=4,
            mlp_ratio=2, model_args=args,
        )
        images = torch.rand(1, 3, 32, 32)
        loss, prediction, _ = original(
            images,
            middle={"image_target": images, "random_selected_class": [1]},
        )
        self.assertEqual(prediction.shape, images.shape)
        self.assertTrue(torch.isfinite(loss))

    def test_exact_orgslot_save_and_load(self):
        source = tiny_model()
        target = tiny_model()
        source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
        target_optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
        source(torch.rand(1, 3, 32, 32))["reconstruction"].mean().backward()
        source_optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/model.pth"
            save_orgslot_checkpoint(path, source, source_optimizer, epoch=3)
            checkpoint, report = load_orgslot_base_checkpoint(
                target, path, optimizer=target_optimizer
            )
        self.assertEqual(checkpoint["epoch"], 3)
        self.assertFalse(report["missing_keys"])
        self.assertFalse(report["unexpected_keys"])
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)
        self.assertEqual(
            len(source_optimizer.state_dict()["state"]),
            len(target_optimizer.state_dict()["state"]),
        )

    def test_incremental_checkpoint_resume(self):
        source = tiny_model()
        target = tiny_model()
        source.append_slot("liver", 4, init_from="background")
        target.append_slot("liver", 4, init_from="background")
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/incremental.pth"
            save_orgslot_checkpoint(path, source, epoch=1)
            _, report = load_orgslot_base_checkpoint(target, path)
        self.assertFalse(report["missing_keys"])
        self.assertEqual(target.slot_names, source.slot_names)

    def test_original_owt_collector_is_sliced_by_slot(self):
        args = model_args("2D", slot_count=3)
        original = OWT_models.MaskedAutoencoderViT(
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
            model_args=args,
        )
        target = tiny_model()
        with torch.no_grad():
            weight = original.organ_embed.conv2.weight
            weight.copy_(torch.arange(weight.numel()).reshape_as(weight))
        report = load_original_owt_initialization(
            target,
            original.state_dict(),
            ["background", "kidney", "spleen"],
        )
        self.assertFalse(report["shape_mismatches"])
        accounted = set(report["loaded_source_keys"]) | set(report["unexpected_keys"])
        accounted |= {
            key for key in original.state_dict()
            if key.startswith(("organ_embed.", "blocks2.", "decoder_embed."))
        }
        self.assertEqual(accounted, set(original.state_dict()))
        for index, name in enumerate(target.slot_names):
            actual = target.slot_bank.get_slot(name).collector.conv2.weight
            expected = original.organ_embed.conv2.weight[index * 2:(index + 1) * 2]
            self.assertTrue(torch.equal(actual, expected), name)
            self.assertTrue(torch.equal(
                target.slot_bank.get_slot(name).aher.sp_linear1.weight,
                original.decoder_embed.sp_linear1.weight,
            ))

    def test_visible_transfer_skips_future_slot(self):
        source = tiny_model()
        source.append_slot("future", 9, init_from="background")
        target = tiny_model()
        target.append_slot("future", 9, init_from="background")
        future_before = {
            key: value.clone()
            for key, value in target.slot_bank.get_slot("future").state_dict().items()
        }
        with torch.no_grad():
            source.slot_bank.get_slot("future").calibration_bias.fill_(7)
            source.slot_bank.get_slot("kidney").calibration_bias.fill_(3)
        report = visible_slot_transfer(
            target, {"model": source.state_dict()},
            visible_slots=("background", "kidney", "spleen"),
        )
        self.assertIn("future", report["slots_intentionally_skipped"])
        self.assertEqual(float(target.slot_bank.get_slot("kidney").calibration_bias), 3)
        for key, value in future_before.items():
            self.assertTrue(torch.equal(value, target.slot_bank.get_slot("future").state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
