#!/usr/bin/env python3
"""Count samples/windows containing each foreground label in an OWT CSV."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--dataset-type', choices=('2D', '3D'), required=True)
    parser.add_argument('--fix-frame', type=int, default=4)
    return parser.parse_args()


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f'cannot read mask: {path}')
    return mask


def window_paths(start_path, fix_frame):
    path = Path(start_path)
    stem, frame_text = path.stem.rsplit('_', 1)
    start_frame = int(frame_text)
    return [
        path.with_name(f'{stem}_{frame}{path.suffix}')
        for frame in range(start_frame, start_frame + fix_frame)
    ]


def present_labels(mask_path, dataset_type, fix_frame):
    if dataset_type == '2D':
        return np.unique(read_mask(mask_path))
    masks = [read_mask(path) for path in window_paths(mask_path, fix_frame)]
    return np.unique(np.stack(masks))


def main():
    args = parse_args()
    if args.num_classes <= 0:
        raise ValueError('--num-classes must be positive')
    if args.dataset_type == '3D' and args.fix_frame <= 0:
        raise ValueError('--fix-frame must be positive')

    counts = np.zeros(args.num_classes, dtype=np.int64)
    dataset_size = 0
    with open(args.csv, newline='', encoding='utf-8') as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames is None or 'mask_pth' not in rows.fieldnames:
            raise ValueError('CSV must contain a mask_pth column')
        for row in rows:
            dataset_size += 1
            labels = present_labels(
                row['mask_pth'], args.dataset_type, args.fix_frame
            )
            for class_id in labels:
                class_id = int(class_id)
                if 1 <= class_id <= args.num_classes:
                    counts[class_id - 1] += 1

    result = {
        'csv': str(Path(args.csv).resolve()),
        'dataset_type': args.dataset_type,
        'fix_frame': args.fix_frame if args.dataset_type == '3D' else None,
        'dataset_size': dataset_size,
        'foreground_class_ids': list(range(1, args.num_classes + 1)),
        'positive_sample_counts': counts.tolist(),
        'positive_sample_frequencies': (counts / dataset_size).tolist(),
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
