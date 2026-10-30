"""Audit native BTCV masks for full-field resize disappearance and ROI feasibility."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
p = argparse.ArgumentParser(__doc__)
p.add_argument('root', type=Path)
a = p.parse_args()
report = {}
for split in ('Training', 'Test'):
    rows = list(csv.DictReader(open(a.root/'csv'/('BTCV_'+split+'_2D_native113.csv'))))
    stats = {str(c): {'positive_slices': 0, 'lost_positive_slices_after_resize': 0,
                      'max_bbox_extent': 0} for c in range(1, 9)}
    for row in rows:
        label = cv2.imread(row['mask_pth'], 0)
        if label is None:
            raise FileNotFoundError(row['mask_pth'])
        h, w = label.shape
        side = max(h, w, 384)
        canvas = np.pad(label, (((side-h)//2, side-h-(side-h)//2),
                               ((side-w)//2, side-w-(side-w)//2)))
        resized = cv2.resize(canvas, (448,448), interpolation=cv2.INTER_NEAREST)
        for c in range(1,9):
            ys,xs = np.where(label == c)
            if len(ys):
                s = stats[str(c)]
                s['positive_slices'] += 1
                s['lost_positive_slices_after_resize'] += int(not np.any(resized == c))
                s['max_bbox_extent'] = max(s['max_bbox_extent'], int(ys.max()-ys.min()+1), int(xs.max()-xs.min()+1))
    report[split] = {'slices': len(rows), 'classes': stats,
                     'global_crop_truncation': False,
                     'roi_focus_bbox_fits_384_before_augmentation': all(stats[str(c)]['max_bbox_extent'] <= 384 for c in (4,5,6))}
out = a.root/'metadata'/'full_field_audit.json'
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
