import unittest
from datasets.orgslot_highres import OrganAwareAppendDataset
class Raw:
    sample_record_indices=tuple(range(8))
    records=[{'image_pth':'/data/case%d/slice_%d.jpg'%(i//4,i)} for i in range(8)]
    def __len__(self):return 8
    def __getitem__(self,i):return {'index':i}
class TestAppend(unittest.TestCase):
    def test_globals_preserved_and_extra_targets_valid(self):
        d=OrganAwareAppendDataset(Raw(),lambda x:x,{'1':[0,4],'2':[1,2,5]},[1,2],.5)
        self.assertEqual(len(d),12)
        for i in range(8):
            s=d[i];self.assertEqual(s['index'],i);self.assertFalse(s['roi_applied']);self.assertEqual(s['focus_class_id'],-1)
        for i in range(8,12):
            s=d[i];self.assertTrue(s['roi_applied']);self.assertIn(s['index'],d.roi_indices[s['focus_class_id']])
        self.assertEqual([d[i]['focus_class_id'] for i in range(8,12)],[1,2,1,2])
if __name__=='__main__':unittest.main()
