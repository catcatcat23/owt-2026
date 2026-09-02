from types import SimpleNamespace
import tempfile
import unittest

import torch

from main_pretrain_orgslot_common8_a100 import _load_initial_checkpoint
from test_orgslot_model import tiny_model


class ArmECheckpointTests(unittest.TestCase):
    def test_arm_c_checkpoint_only_misses_new_spatial_modules(self):
        torch.manual_seed(29)
        arm_c = tiny_model(
            slot_head_type="query_dot", slot_head_channels=16
        )
        torch.manual_seed(31)
        arm_e = tiny_model(
            slot_head_type="arm_e_multiscale_query", slot_head_channels=16
        )
        saved_args = {
            "input_size": 32,
            "token_factor": 2,
            "slot_tg_depth": 1,
            "fusion_mode": "post_layernorm",
            "fusion_reference_count": 3,
            "slot_head_type": "query_dot",
        }
        requested = SimpleNamespace(
            **saved_args,
        )
        requested.slot_head_type = "arm_e_multiscale_query"

        with tempfile.NamedTemporaryFile(suffix=".pth") as checkpoint_file:
            torch.save(
                {
                    "model": arm_c.state_dict(),
                    "args": saved_args,
                    "epoch": 802,
                },
                checkpoint_file.name,
            )
            report = _load_initial_checkpoint(
                arm_e, checkpoint_file.name, requested
            )

        self.assertFalse(report["exact"])
        self.assertTrue(report["compatible_missing_keys"])
        allowed_prefixes = (
            "pixel_query_decoder.spatial_stem.",
            "pixel_query_decoder.p8_proj.",
            "pixel_query_decoder.p4_proj.",
        )
        self.assertTrue(all(
            key.startswith(allowed_prefixes)
            for key in report["compatible_missing_keys"]
        ))
        arm_c_state = arm_c.state_dict()
        arm_e_state = arm_e.state_dict()
        for key, value in arm_c_state.items():
            self.assertTrue(torch.equal(value, arm_e_state[key]), key)


if __name__ == "__main__":
    unittest.main()
