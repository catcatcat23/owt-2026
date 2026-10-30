"""Check unmodified WORD center-crop448 geometry on BTCV without resizing/padding."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
p = argparse.ArgumentParser(__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
summary = json.loads((a.root/'metadata/preprocess_summary.json').read_text())
assert np.allclose(summary['config']['spacing_mm'], [0.7,0.7,2])
expected = json.loads((Path(__file__).resolve().parents[1]/'configs/orgslot/btcv_offline_seed42.json').read_text())
report = {'crop_size':448,'spacing_mm':[0.7,0.7,2],'cases':{}}
for split in ('Training','Test'):
    rows=list(csv.DictReader(open(a.root/'csv'/('BTCV_'+split+'_2D_native07072.csv'))))
    assert sorted({Path(x['image_pth']).parent.name for x in rows}) == sorted(expected['splits'][split])
    for row in rows:
        label=cv2.imread(row['mask_pth'],0)
        if label is None: raise FileNotFoundError(row['mask_pth'])
        case=Path(row['image_pth']).parent.name
        h,w=label.shape
        item=report['cases'].setdefault(case, {'split':split,'shape_xy':[h,w],'fits':min(h,w)>=448,'lost_voxels_by_class':[0]*8,'clipped_slices_by_class':[0]*8,'erased_slices_by_class':[0]*8})
        if not item['fits']: continue
        y,x=(h-448)//2,(w-448)//2
        before=np.bincount(label.ravel(),minlength=9)[1:9]
        after=np.bincount(label[y:y+448,x:x+448].ravel(),minlength=9)[1:9]
        for c in range(8):
            item['lost_voxels_by_class'][c]+=int(before[c]-after[c])
            item['clipped_slices_by_class'][c]+=int(after[c]<before[c])
            item['erased_slices_by_class'][c]+=int(before[c]>0 and after[c]==0)
report['undersized_cases']=[k for k,v in report['cases'].items() if not v['fits']]
report['clipped_cases']=[k for k,v in report['cases'].items() if sum(v['lost_voxels_by_class'])]
report['compatible']=not report['undersized_cases'] and not report['clipped_cases']
output=a.root/'metadata/fixed_crop_audit.json'
output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not report['compatible']:
    raise SystemExit('STOP: exact WORD crop has size/coverage conflicts; do not submit until explicitly resolved')
