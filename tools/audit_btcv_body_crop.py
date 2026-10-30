"""Audit image-only body512 windows; labels are used only for coverage counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from datasets.orgslot_highres import body_center_window, crop_with_padding


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    cases, manifests = {}, {}
    for split in ('Training', 'Test'):
        path = args.root / 'csv' / ('BTCV_' + split + '_2D_native07072.csv')
        manifests[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in csv.DictReader(path.open()):
            image = cv2.imread(row['image_pth'], 0)
            if image is None:
                raise FileNotFoundError(row['image_pth'])
            box = body_center_window(torch.from_numpy(image.copy()).float().unsqueeze(0) / 255)
            gt = cv2.imread(row['mask_pth'], 0)
            if gt is None or gt.shape != image.shape:
                raise ValueError(row)
            cropped = crop_with_padding(torch.from_numpy(gt.copy()), *box).numpy()
            full = np.bincount(gt.ravel(), minlength=9)[1:9]
            kept = np.bincount(cropped.ravel(), minlength=9)[1:9]
            name = Path(row['mask_pth']).parent.name
            c = cases.setdefault(name, dict(split=split, slices=0, full=[0]*8, lost=[0]*8))
            c['slices'] += 1
            c['full'] = (np.array(c['full']) + full).tolist()
            c['lost'] = (np.array(c['lost']) + full-kept).tolist()
    report = dict(policy='body_center512_v1', manifests=manifests, cases=cases,
                  body_clipped_cases=[k for k,v in cases.items() if sum(v['lost'])])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(cases=len(cases), slices=sum(v['slices'] for v in cases.values()),
                         clipped=report['body_clipped_cases'])))
    if report['body_clipped_cases']:
        raise RuntimeError('Body window loses GT; do not submit')


if __name__ == '__main__':
    main()
