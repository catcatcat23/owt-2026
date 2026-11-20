import unittest
import numpy as np
import torch
from datasets.orgslot_highres import body_center_window
from tools.eval_common8_orgslot_heads_3d import accumulate_native_slab

class NativeGeometryTests(unittest.TestCase):
    def test_constant_slab_matches_2d_window(self):
        x=torch.zeros(3,550,580);x[:,90:440,100:490]=0.7
        self.assertEqual(body_center_window(x),body_center_window(x[:,None].repeat(1,4,1,1)))

    def test_padding_overlap_and_full_gt(self):
        sums=np.zeros((1,1,3,3),np.float32);counts=np.zeros((1,3,3),np.uint16)
        target=np.ones((1,3,3),np.int64);targets=[None]
        # Window starts outside native image; bottom row remains uncovered GT.
        box=(-1,0,3,3,3,3)
        for value in (0.2,0.8):
            accumulate_native_slab(sums,counts,targets,{7:0},[7],np.full((1,1,3,3),value),target,box)
        score=sums/np.maximum(counts,1)[None]
        np.testing.assert_allclose(score[0,0,:2],0.5)
        np.testing.assert_array_equal(score[0,0,2],0)
        np.testing.assert_array_equal(targets[0],target[0])

if __name__=='__main__':unittest.main()
