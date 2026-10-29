"""CPU regression for persistent groups, freezing, objective and strict evaluator reload."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_orgslot_model import tiny_model
from test_incremental_sam_private import args_for_test
from util.orgslot_incremental44 import Incremental44Objective
from util.orgslot_incremental2222 import GROUPS, NAMES, seen, install_group, restore_groups, configure_group


def model(stage):
    net = tiny_model(specs=[{'name': n, 'raw_class_id': i} for i, n in enumerate(seen(stage))],
                     slot_head_type='arm_f_sam_tail', slot_head_channels=16)
    net.pixel_query_decoder.configure_mask_supervision('soft_prior_aux')
    return net


class FourStageTests(unittest.TestCase):
    def test_published_order_and_visibility(self):
        import json
        from util.label_visibility import load_visibility_config, build_visible_binary_masks
        from tools.eval_common8_orgslot_reconstruction_threshold import parse_class_configuration
        config = Path('configs/orgslot/common8_incremental2222.json')
        classes, stages = load_visibility_config(config)
        expected = (("spleen", "right_kidney"), ("left_kidney", "gallbladder"),
                    ("esophagus", "pancreas"), ("liver", "stomach"))
        self.assertEqual(GROUPS, expected)
        labels = torch.arange(9).reshape(1, 3, 3)
        for stage in range(1, 5):
            masks = build_visible_binary_masks(labels, classes, stages['stage'+str(stage)])
            self.assertEqual(set(masks), set(seen(1) if stage == 1 else GROUPS[stage-1]))
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'classes.json'
                data = json.loads(config.read_text())
                data.update(incremental_protocol='2222', stage_index=stage,
                            stages={'base':{'stage':'evaluation','visible_slots':list(seen(stage))}})
                path.write_text(json.dumps(data))
                specs, names = parse_class_configuration(path)
                self.assertEqual(list(names.values()), list(seen(stage)))

    def test_all_transitions_and_evaluator(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        net = model(1).eval()
        image = torch.rand(2, 3, 32, 32)
        for stage in (2, 3, 4):
            net.eval()
            with torch.no_grad():
                before = net(image, decode_reconstruction=False)['slot_logits']
            for name in GROUPS[stage - 1]:
                net.append_slot(name, NAMES.index(name) + 1, init_from='background')
            install_group(net, stage)
            configure_group(net, stage)
            self.assertEqual(len(net.stage_pixel_decoders), stage - 1)
            a, b = GROUPS[stage - 1]
            self.assertIs(net.slot_bank.get_slot(a).interaction, net.slot_bank.get_slot(b).interaction)
            args = args_for_test()
            args.stage2_shared_segmentation = 'frozen'
            objective = Incremental44Objective(net, args).train()
            frozen = {n: p.clone() for n, p in net.named_parameters() if not p.requires_grad}
            masks = {n: torch.zeros(2, 1, 32, 32) for n in GROUPS[stage - 1]}
            masks[a][:, :, 5:12, 6:14] = 1
            masks[b][:, :, 16:24, 19:28] = 1
            probs = {n: torch.zeros(2, 1, 32, 32) for n in seen(stage - 1)[1:]}
            keep = torch.ones(2, len(seen(stage)), dtype=torch.bool)
            optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=.001)
            loss, full = objective(image, masks, probs, keep)
            loss.backward()
            grads = [p.grad for p in net.stage_pixel_decoders['stage'+str(stage)].parameters() if p.grad is not None]
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            optimizer.step()
            for n, p in net.named_parameters():
                if n in frozen:
                    self.assertTrue(torch.equal(p, frozen[n]), n)
            net.eval()
            with torch.no_grad():
                after = net(image, decode_reconstruction=False)['slot_logits']
            for name in seen(stage - 1)[1:]:
                torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)
            args.seg_supervision = 'retained'
            keep[:, 1:] = False
            _, empty = objective(image, masks, probs, keep)
            self.assertEqual(set(full), set(empty))
            self.assertTrue(torch.isfinite(empty['loss']))
            fresh = model(stage)
            from tools.eval_common8_orgslot_reconstruction_threshold import build_model
            with tempfile.TemporaryDirectory() as d:
                path = Path(d)/'checkpoint.pth'
                torch.save({'model': net.state_dict(), 'epoch': 0, 'args': {
                    'input_size':32, 'incremental_protocol':'2222','stage_index':stage,
                    'incremental_stage':'stage'+str(stage),'slot_head_type':'arm_f_sam_tail',
                    'hsam_supervision':'soft_prior_aux'}}, path)
                with patch('tools.eval_common8_orgslot_reconstruction_threshold.build_orgslot', return_value=fresh):
                    restored, report = build_model('orgslot', path, [], 32, 'linear_sqrt')
                self.assertTrue(report['exact'])
                with torch.no_grad():
                    actual = restored.eval()(image, decode_reconstruction=False)['slot_logits']
                for name in seen(stage)[1:]:
                    torch.testing.assert_close(actual[name], after[name], atol=0, rtol=0)

if __name__ == '__main__':
    unittest.main()
