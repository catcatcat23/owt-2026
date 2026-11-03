"""CPU checks for variable class groups and exact restoration; no GPU preflight."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_orgslot_model import tiny_model
from test_incremental_sam_private import args_for_test
from util.orgslot_incremental2222 import protocol_groups, seen, install_group, configure_group
from util.orgslot_incremental44 import Incremental44Objective
from tools.eval_common8_orgslot_reconstruction_threshold import parse_class_configuration, build_model

class SuiteTests(unittest.TestCase):
    def test_visibility_all_protocols(self):
        from util.label_visibility import load_visibility_config, build_visible_binary_masks
        for protocol in ('pcdd_44','pcdd_422','pcdd_41111','pcdd_71','pcdd_offline'):
            classes, stages=load_visibility_config('configs/orgslot/'+protocol+'.json')
            groups=protocol_groups(protocol)
            self.assertEqual(sum(map(len,groups)),8)
            for stage,group in enumerate(groups,1):
                masks=build_visible_binary_masks(torch.arange(9).reshape(1,3,3),classes,stages['stage'+str(stage)])
                self.assertEqual(set(masks),set(seen(1,protocol) if stage==1 else group))
                with tempfile.TemporaryDirectory() as root:
                    path=Path(root)/'classes.json'
                    data=json.loads(Path('configs/orgslot/'+protocol+'.json').read_text())
                    data.update(incremental_protocol=protocol,stage_index=stage,stages={'base':{'stage':'evaluation','visible_slots':list(seen(stage,protocol))}})
                    path.write_text(json.dumps(data));_,names=parse_class_configuration(path)
                    self.assertEqual(list(names.values()),list(seen(stage,protocol)))

    def test_single_new_class_and_loader(self):
        torch.set_num_threads(1)
        protocol='pcdd_71'
        def make(stage):
            net=tiny_model(specs=[{'name':n,'raw_class_id':i} for i,n in enumerate(seen(stage,protocol))],slot_head_type='arm_f_sam_tail',slot_head_channels=16)
            net.pixel_query_decoder.configure_mask_supervision('soft_prior_aux')
            return net
        net=make(1).eval();image=torch.rand(2,3,32,32)
        with torch.no_grad():before=net(image,decode_reconstruction=False)['slot_logits']
        net.append_slot('stomach',8,init_from='background');install_group(net,2,protocol);configure_group(net,2,protocol=protocol)
        frozen={n:p.detach().clone() for n,p in net.named_parameters() if not p.requires_grad}
        args=args_for_test();args.stage2_shared_segmentation='frozen'
        objective=Incremental44Objective(net,args)
        masks={'stomach':torch.ones(2,1,32,32)}
        probs={n:torch.zeros(2,1,32,32) for n in seen(1,protocol)[1:]}
        optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001)
        loss,_=objective(image,masks,probs,torch.ones(2,9,dtype=torch.bool));loss.backward();optimizer.step()
        for n,p in net.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]),n)
        net.eval()
        with torch.no_grad():after=net(image,decode_reconstruction=False)['slot_logits']
        for name in seen(1,protocol)[1:]:torch.testing.assert_close(before[name],after[name],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'checkpoint.pth'
            torch.save({'model':net.state_dict(),'epoch':0,'args':{'input_size':32,'incremental_protocol':protocol,'stage_index':2,'incremental_stage':'stage2','slot_head_type':'arm_f_sam_tail','hsam_supervision':'soft_prior_aux'}},path)
            with patch('tools.eval_common8_orgslot_reconstruction_threshold.build_orgslot',return_value=make(2)):
                restored,report=build_model('orgslot',path,[],32,'linear_sqrt')
            self.assertTrue(report['exact'])
            with torch.no_grad():loaded=restored(image,decode_reconstruction=False)['slot_logits']
            for name in after:torch.testing.assert_close(after[name],loaded[name],rtol=0,atol=0)

if __name__=='__main__':unittest.main()
