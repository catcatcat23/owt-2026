"""3D slab inference with the existing 2D calibration and scoring protocol."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.orgslot_manifest import parse_case_and_slice
from tools.eval_common8_orgslot_heads import (
    add_volume_ratios, load_class_thresholds, parse_threshold_sweep,
    select_thresholds, threshold_label,
)
from tools.eval_common8_orgslot_heads_3d import (
    accumulate_slab, build_3d_dataset, build_3d_model,
)
from tools.eval_common8_orgslot_reconstruction_threshold import (
    atomic_json, new_counter, parse_class_configuration, postprocess_slice,
    reconstruct_from_keep, sha256, slot_canvases, summarize, update_counter,
)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-csv', required=True)
    parser.add_argument('--test-reference-csv', required=True)
    parser.add_argument('--preprocess-summary', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--class-map', default='configs/orgslot/common8_offline.json')
    parser.add_argument('--mode', choices=('calibrate', 'head', 'reconstruction'), required=True)
    parser.add_argument('--class-thresholds')
    parser.add_argument('--threshold-sweep', default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--calibration-samples', type=int, default=6000)
    parser.add_argument('--expected-cases', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()


def manifest_keys(path):
    with open(path, newline='') as handle:
        keys = [parse_case_and_slice(row['image_pth']) for row in csv.DictReader(handle)]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate case/slice in manifest')
    return keys


def scoring_keys(mode, data_keys, test_keys, calibration_samples):
    if mode == 'calibrate':
        if {case for case, _ in data_keys} & {case for case, _ in test_keys}:
            raise ValueError('calibration/test case overlap')
        if not 0 < calibration_samples <= len(data_keys):
            raise ValueError('invalid calibration sample count')
        return set(data_keys[:calibration_samples])
    if set(data_keys) != set(test_keys):
        raise ValueError('test evaluation must cover the complete reference manifest')
    return set(data_keys)


def recon_scores(model, image, class_ids, id_to_index):
    encoded, _ = model.forward_encoder(image)
    canvases = slot_canvases(model, encoded)
    direct_scores, indirect_scores = [], []
    for class_id in class_ids:
        keep = torch.zeros(image.shape[0], len(model.slot_names), dtype=torch.bool, device=image.device)
        keep[:, id_to_index[class_id]] = True
        direct = reconstruct_from_keep(model, canvases, keep)
        complement = reconstruct_from_keep(model, canvases, ~keep)
        direct_scores.append(direct.mean(dim=1).float())
        indirect_scores.append((image - complement).clamp_min(0).mean(dim=1).float())
    return {'direct': torch.stack(direct_scores, dim=1),
            'indirect': torch.stack(indirect_scores, dim=1)}


def count_slice(counters, case, class_id, score, target, threshold, prefix):
    raw = score > threshold
    for suffix, prediction in [('raw', raw), ('post', postprocess_slice(raw, 20, 1))]:
        counter = counters[prefix + '_' + suffix][class_id].setdefault(case, new_counter())
        update_counter(counter, prediction, target)


def main():
    args = parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    data_keys, test_keys = manifest_keys(args.data_csv), manifest_keys(args.test_reference_csv)
    selected_keys = scoring_keys(args.mode, data_keys, test_keys, args.calibration_samples)
    if len({case for case, _ in data_keys}) != args.expected_cases:
        raise ValueError('unexpected manifest case count')
    spacing = json.loads(Path(args.preprocess_summary).read_text())['config']['spacing_mm']
    if not np.allclose(spacing, [0.7, 0.7, 2.0], rtol=0, atol=1e-6):
        raise ValueError('spacing mismatch')
    checkpoint_hash = sha256(Path(args.checkpoint))
    test_hash = sha256(Path(args.test_reference_csv))
    specs, names = parse_class_configuration(Path(args.class_map))
    class_ids = tuple(range(1, 9))
    threshold_sets = {'binary': {class_id: 0.5 for class_id in class_ids}}
    candidates = parse_threshold_sweep(args.threshold_sweep)
    if args.mode == 'calibrate' and (not candidates or not np.isfinite(candidates).all()):
        raise ValueError('calibration requires finite threshold candidates')
    if args.mode == 'calibrate':
        if args.class_thresholds:
            raise ValueError('external thresholds forbidden during calibration')
        threshold_sets.update({'sweep_' + threshold_label(value): dict.fromkeys(class_ids, value)
                               for value in candidates})
    elif args.mode == 'head':
        if not args.class_thresholds:
            raise ValueError('head evaluation requires completed calibration')
        provenance = json.loads(Path(args.class_thresholds).read_text())
        if provenance.get('checkpoint_sha256') != checkpoint_hash or provenance.get('test_reference_sha256') != test_hash:
            raise ValueError('threshold checkpoint/test identity mismatch')
        if provenance.get('split') != 'train_calibration' or not provenance.get('test_cases_disjoint'):
            raise ValueError('invalid calibration provenance')
        threshold_sets['selected'] = load_class_thresholds(args.class_thresholds, class_ids)
        if not np.isfinite(list(threshold_sets['selected'].values())).all():
            raise ValueError('nonfinite calibrated threshold')
    elif args.class_thresholds:
        raise ValueError('reconstruction uses the fixed legacy threshold 0.02')
    prefixes = ('direct', 'indirect') if args.mode == 'reconstruction' else tuple(threshold_sets)
    counters = {prefix + '_' + suffix: {class_id: {} for class_id in class_ids}
                for prefix in prefixes for suffix in ('raw', 'post')}
    dataset, case_indices, case_slices = build_3d_dataset(args.data_csv, 448, 448, 4)
    if set(data_keys) != {(case, index) for case, indices in case_slices.items() for index in indices}:
        raise ValueError('3D dataset does not cover the manifest exactly')
    if not torch.cuda.is_available():
        raise RuntimeError('formal inference requires allocated CUDA')
    model, load_report = build_3d_model(args.checkpoint, specs, 448, 'linear_sqrt', 4, 1)
    model.cuda().eval()
    id_to_name = {int(spec['raw_class_id']): spec['name'] for spec in specs}
    id_to_index = {int(spec['raw_class_id']): index for index, spec in enumerate(specs)}
    cases = sorted({case for case, _ in selected_keys})
    out.mkdir(parents=True)
    config = dict(vars(args), checkpoint_sha256=checkpoint_hash, test_reference_sha256=test_hash,
                  data_csv_sha256=sha256(Path(args.data_csv)), scoring_slices=len(selected_keys),
                  scoring_cases=cases, geometry='448 deterministic center crop',
                  slab_fusion='mean scores per original slice before thresholding',
                  postprocess='2D postprocess_slice: min_size=20, opening_radius=1',
                  reconstruction='Direct mean channels; Indirect clamp_min(input-complement,0) then mean channels; threshold=0.02',
                  metric='existing 2D summarize: case_dice_presence_mean', amp_dtype='fp16')
    atomic_json(out / 'resolved_config.json', config)
    atomic_json(out / 'checkpoint_load.json', load_report)
    atomic_json(out / 'scoring_keys.json', sorted(selected_keys))
    started = time.time()
    processed = 0
    with torch.inference_mode():
        for case_number, case in enumerate(cases, 1):
            indices = case_slices[case]
            positions = {value: index for index, value in enumerate(indices)}
            shape = (len(class_ids), len(indices), 448, 448)
            sources = ('direct', 'indirect') if args.mode == 'reconstruction' else ('head',)
            sums = {source: np.zeros(shape, np.float32) for source in sources}
            counts = {source: np.zeros(len(indices), np.uint16) for source in sources}
            targets = [None] * len(indices)
            loader = DataLoader(Subset(dataset, case_indices[case]), batch_size=args.batch_size,
                                shuffle=False, num_workers=args.workers, pin_memory=True)
            for batch in loader:
                if any(str(value) != case for value in batch['case_id']):
                    raise ValueError('mixed-case inference batch')
                image = batch['image'].cuda(non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    if args.mode == 'reconstruction':
                        scores = recon_scores(model, image, class_ids, id_to_index)
                    else:
                        output = model(image, slot_keep_mask=torch.ones(image.shape[0], len(specs),
                            dtype=torch.bool, device=image.device), decode_reconstruction=False, decode_heads=True)
                        scores = {'head': torch.stack([torch.sigmoid(output['calibrated_logits'][id_to_name[class_id]][:, 0])
                                                     for class_id in class_ids], dim=1)}
                for source, values in scores.items():
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError('nonfinite ' + source)
                    values = values.float().cpu().numpy()
                    for sample in range(len(image)):
                        accumulate_slab(sums[source], counts[source], targets, positions,
                            batch['slice_indices'][sample].numpy(), values[sample], batch['full_label'][sample, 0].numpy())
            for source in sources:
                if np.any(counts[source] == 0):
                    raise ValueError('uncovered original slices')
                sums[source] /= counts[source][None, :, None, None]
            for position, slice_index in enumerate(indices):
                if (case, slice_index) not in selected_keys:
                    continue
                for offset, class_id in enumerate(class_ids):
                    target = targets[position] == class_id
                    if args.mode == 'reconstruction':
                        for source in sources:
                            count_slice(counters, case, class_id, sums[source][offset, position], target, 0.02, source)
                    else:
                        for prefix, thresholds in threshold_sets.items():
                            count_slice(counters, case, class_id, sums['head'][offset, position], target, thresholds[class_id], prefix)
                processed += 1
            atomic_json(out / 'progress.json', dict(processed=processed, total=len(selected_keys),
                processed_cases=case_number, total_cases=len(cases), complete=False))
            print(json.dumps(dict(case=case, processed=processed, total=len(selected_keys))), flush=True)
            del sums, targets
    if processed != len(selected_keys):
        raise ValueError('incomplete evaluation')
    metrics, rows, records = summarize(counters, class_ids, names, 'OrganSlot 3D ' + args.mode)
    add_volume_ratios(metrics, rows)
    split = 'train_calibration' if args.mode == 'calibrate' else 'test'
    for record in records:
        record['split'] = split
    if args.mode == 'calibrate':
        atomic_json(out / 'selected_thresholds.json', dict(thresholds=select_thresholds(metrics, class_ids, candidates),
            candidates=candidates, split=split, test_cases_disjoint=True, checkpoint_sha256=checkpoint_hash,
            test_reference_sha256=test_hash, calibration_csv_sha256=config['data_csv_sha256'],
            samples=processed, selection_metric='case_dice_presence_mean after 2D postprocessing'))
    with open(out / 'per_case.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / 'per_case.jsonl', 'w') as handle:
        for record in records:
            handle.write(json.dumps(record) + '\n')
    atomic_json(out / 'results.json', dict(metrics=metrics, checkpoint_load=load_report, split=split,
        cases=len(cases), samples=processed, complete_split=True, complete_test_set=split == 'test',
        elapsed_seconds=time.time() - started))
    atomic_json(out / 'progress.json', dict(processed=processed, total=processed, complete=True))


if __name__ == '__main__':
    main()
