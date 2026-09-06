"""Checkpoint-only diagnostics; never modifies training or formal result paths."""
import csv
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import expit
import torch
from torch.utils.data import DataLoader, Subset

FUSIONS = ('mean_prob', 'mean_logit', 'center_only')
SOURCES = ('raw', 'calibrated', 'reconstruction')


def center_position(slice_indices, anchor):
    """Interior center is T//2 (index 2 for T=4); clamped edges use anchor.

    The dataset has one anchor per original slice. Selecting that anchor's
    temporal position covers every boundary exactly once, without averaging.
    """
    positions = np.flatnonzero(np.asarray(slice_indices) == int(anchor))
    if len(positions) != 1:
        raise ValueError('anchor must occur exactly once in its slab')
    return int(positions[0])


def probability_statistics(probabilities, target):
    q = np.percentile(probabilities, [0, 1, 10, 50, 90, 99, 100])
    foreground = probabilities[target]
    background = probabilities[~target]
    return dict(zip(('min', 'p1', 'p10', 'median', 'p90', 'p99', 'max'), map(float, q)),
                gt_foreground_mean=float(foreground.mean()) if foreground.size else float('nan'),
                gt_background_mean=float(background.mean()) if background.size else float('nan'),
                background_above_05=float((background > .5).mean()) if background.size else float('nan'),
                foreground_above_05=float((foreground > .5).mean()) if foreground.size else float('nan'))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    # Import here so helpers stay cheap to test and the formal evaluator can dispatch.
    from tools.eval_common8_orgslot_heads_3d import (
        build_3d_dataset, build_3d_model, parse_class_configuration, load_thresholds,
        accumulate_slab, overlap_metrics, postprocess_volume, atomic_json, sha256,
    )
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(out)
    if args.input_size != 448 or args.global_crop_size != 448:
        raise ValueError('diagnostics use the same deterministic 448 crop')
    if not torch.cuda.is_available():
        raise RuntimeError('checkpoint inference requires allocated CUDA')
    preprocess = json.loads(Path(args.preprocess_summary).read_text())
    if not np.allclose(preprocess['config']['spacing_mm'], args.expected_spacing, rtol=0, atol=1e-6):
        raise ValueError('spacing mismatch')
    if args.binary_threshold != .5 or args.class_thresholds is not None:
        raise ValueError('diagnostics fix threshold at .5; no test-set tuning')
    spacing = tuple(args.expected_spacing[::-1])
    specs, names = parse_class_configuration(Path(args.class_map))
    ids = tuple(int(x) for x in args.class_ids.split(','))
    if len(set(ids)) != len(ids) or any(x == 0 or x not in names for x in ids):
        raise ValueError('invalid foreground class IDs')
    id_to_name = {int(s['raw_class_id']): s['name'] for s in specs}
    dataset, case_indices, case_slices = build_3d_dataset(
        args.test_csv, args.input_size, args.global_crop_size, args.fix_frame)
    if len(case_indices) != args.expected_cases:
        raise ValueError('unexpected manifest case count')
    cases = sorted(case_indices)
    if args.diagnostic_case_limit:
        cases = cases[:args.diagnostic_case_limit]
    model, load_report = build_3d_model(args.checkpoint, specs, args.input_size,
        args.fusion_mode, args.fix_frame, args.temp_stride)
    device = torch.device(args.device)
    model.to(device).eval()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'checkpoint_load.json', load_report)
    calibration = {n: {'scale': float(model.slot_bank.get_slot(n).calibration_scale),
                       'bias': float(model.slot_bank.get_slot(n).calibration_bias)}
                   for n in id_to_name.values()}
    atomic_json(out/'calibration.json', calibration)
    config = dict(vars(args), cases_selected=cases,
        checkpoint_sha256=sha256(Path(args.checkpoint)),
        test_csv_sha256=sha256(Path(args.test_csv)),
        complete_test_set=len(cases)==len(case_indices),
        center_rule='T//2=2 for interior T=4; clamped boundary uses anchor position; exactly one prediction per original slice',
        threshold_rule='head > 0.5; reconstruction channel mean > 0.02 (legacy fixed threshold)',
        probability_precision='sigmoid in native output dtype, then float32 accumulation (legacy-compatible)',
        geometry='deterministic center crop, not native full field of view',
        surface_metrics='omitted in diagnostics; overlap metrics only')
    atomic_json(out/'resolved_config.json', config)
    rows, stats_rows, z_rows, data_rows = [], [], [], []
    started = time.time()
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.amp_dtype]
    for case_number, case in enumerate(cases, 1):
        indices = case_slices[case]
        if any(b-a != 1 for a,b in zip(indices, indices[1:])):
            raise ValueError('noncontiguous slice numbering')
        position = {s: i for i,s in enumerate(indices)}
        shape = (len(ids), len(indices), args.input_size, args.input_size)
        sums = {source: {fusion: np.zeros(shape, np.float32) for fusion in FUSIONS} for source in SOURCES}
        counts = np.zeros(len(indices), np.uint16)
        center_counts = np.zeros(len(indices), np.uint16)
        targets = [None]*len(indices)
        loader = DataLoader(Subset(dataset, case_indices[case]), batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers, pin_memory=True)
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader):
                image = batch['image'].to(device)
                if image.shape[1:3] != (3,args.fix_frame):
                    raise ValueError('image axes disagree with B,C,T,H,W')
                if not torch.equal(image[:,0],image[:,1]) or not torch.equal(image[:,1],image[:,2]):
                    raise ValueError('3D grayscale channels differ')
                keep = torch.ones(image.shape[0],len(specs),device=device,dtype=torch.bool)
                with torch.cuda.amp.autocast(enabled=not args.no_amp and dtype!=torch.float32, dtype=dtype):
                    output = model(image, slot_keep_mask=keep, decode_reconstruction=False,
                                   decode_heads=True, return_diagnostics=True)
                    recon = []
                    for c in ids:
                        selected = torch.zeros_like(keep)
                        selected[:, model.slot_names.index(id_to_name[c])] = True
                        canvas = model.fuse_canvases(output['slot_canvases'], selected, model.slot_names)
                        reconstruction = model.forward_decoder(canvas).mean(dim=1)
                        recon.append(reconstruction.float())
                logits = {source: torch.stack([output[key][id_to_name[c]][:,0] for c in ids], dim=1)
                    for source,key in [('raw','slot_logits'),('calibrated','calibrated_logits')]}
                reconstruction_scores = torch.stack(recon, dim=1)
                logits['reconstruction'] = torch.logit(reconstruction_scores.clamp(1e-7, 1-1e-7))
                if case_number == 1 and batch_number == 0:
                    precision = []
                    reference = logits['calibrated'].float()
                    for control in ('bf16','fp32'):
                        with torch.cuda.amp.autocast(enabled=control!='fp32',dtype=torch.bfloat16):
                            other = model(image,slot_keep_mask=keep,decode_reconstruction=False,decode_heads=True)
                        for c in ids:
                            a=reference[:,ids.index(c)]
                            b=other['calibrated_logits'][id_to_name[c]][:,0].float()
                            precision.append(dict(class_id=c,reference=args.amp_dtype,control=control,
                                logit_abs_mean=float((a-b).abs().mean()),logit_abs_max=float((a-b).abs().max()),
                                threshold_disagreement=float(((a>0)!=(b>0)).float().mean()),
                                reference_foreground=float((a>0).float().mean()),control_foreground=float((b>0).float().mean())))
                        del other
                    atomic_json(out/'precision_control.json',precision)
                    atomic_json(out/'input_check.json',dict(shape=list(image.shape),minimum=float(image.min()),maximum=float(image.max()),channels_identical=True))
                labels = batch['full_label'][:,0].numpy()
                slab_indices = batch['slice_indices'].numpy()
                for source in SOURCES:
                    if not bool(torch.isfinite(logits[source]).all()):
                        raise FloatingPointError('nonfinite '+source+' logits')
                    probability = torch.sigmoid(logits[source]).float().cpu().numpy()
                    values = logits[source].float().cpu().numpy()
                    for b in range(len(image)):
                        if str(batch['case_id'][b]) != case:
                            raise ValueError('mixed-case batch')
                        t_center = center_position(slab_indices[b], batch['slice_index'][b])
                        anchor_pos = position[int(batch['slice_index'][b])]
                        if source == 'raw':
                            accumulate_slab(sums[source]['mean_prob'],counts,targets,position,slab_indices[b],probability[b],labels[b])
                            center_counts[anchor_pos] += 1
                        else:
                            for t,s in enumerate(slab_indices[b]):
                                sums[source]['mean_prob'][:,position[int(s)]] += probability[b,:,t]
                        for t,s in enumerate(slab_indices[b]):
                            sums[source]['mean_logit'][:,position[int(s)]] += values[b,:,t]
                        sums[source]['center_only'][:,anchor_pos] = probability[b,:,t_center]
                if batch_number % args.print_freq == 0:
                    print(json.dumps(dict(case=case,slabs=(batch_number+1)*args.batch_size,total=len(case_indices[case]))),flush=True)
        if np.any(counts==0) or not np.all(center_counts==1):
            raise ValueError('fusion has missing or duplicated slices')
        target_volume = np.stack(targets)
        for source in SOURCES:
            sums[source]['mean_prob'] /= counts[None,:,None,None]
            sums[source]['mean_logit'] /= counts[None,:,None,None]
            expit(sums[source]['mean_logit'],out=sums[source]['mean_logit'])
        for offset,c in enumerate(ids):
            target = target_volume==c
            positive = target.reshape(len(indices),-1).any(1)
            slab_positive = []
            mixed_empty = 0
            for anchor in range(len(indices)):
                start=min(max(anchor-args.fix_frame//2,0),len(indices)-args.fix_frame)
                window=positive[start:start+args.fix_frame]
                slab_positive.append(bool(window.any()))
                if window.any():
                    mixed_empty += int((~window).sum())
            data_rows.append(dict(case_id=case,class_id=c,positive_slices=int(positive.sum()),negative_slices=int((~positive).sum()),positive_slabs=sum(slab_positive),negative_slabs=len(indices)-sum(slab_positive),empty_slice_occurrences_in_positive_slabs=mixed_empty,coverage_min=int(counts.min()),coverage_max=int(counts.max())))
            for fusion in FUSIONS:
                predictions = {}
                for source in SOURCES:
                    p = sums[source][fusion][offset]
                    binary = p > (.02 if source == 'reconstruction' else .5)
                    post = postprocess_volume(binary,args.min_component_voxels,args.opening_radius_mm,spacing)
                    predictions[source]=(binary,post)
                    info = dict(case_id=case,class_id=c,class_name=names[c],logit_source=source,slab_fusion=fusion)
                    stat = dict(info,**probability_statistics(p,target))
                    stat['prediction_to_target_volume_ratio']=float(binary.sum()/target.sum()) if target.any() else float('nan')
                    stats_rows.append(stat)
                    for mode,pred in [('head_raw3d',binary),('head_post3d',post)]:
                        metrics=overlap_metrics(pred,target)
                        metrics['prediction_to_target_volume_ratio']=metrics['prediction_voxels']/metrics['target_voxels'] if metrics['target_voxels'] else float('nan')
                        rows.append(dict(info,mode=mode,**metrics))
                raw=predictions['raw'][0]
                cal,post=predictions['calibrated']
                recon_raw,recon_post=predictions['reconstruction']
                for t,s in enumerate(indices):
                    z_rows.append(dict(case_id=case,class_id=c,class_name=names[c],slab_fusion=fusion,slice_index=s,gt_voxels=int(target[t].sum()),raw_predicted_voxels=int(raw[t].sum()),calibrated_predicted_voxels=int(cal[t].sum()),postprocessed_voxels=int(post[t].sum()),reconstruction_voxels=int(recon_raw[t].sum()),reconstruction_post_voxels=int(recon_post[t].sum()),raw_mean_probability=float(sums['raw'][fusion][offset,t].mean()),calibrated_mean_probability=float(sums['calibrated'][fusion][offset,t].mean())))
            print(json.dumps(dict(case=case,class_id=c,statistics_complete=True)),flush=True)
        write_csv(out/'per_case.csv',rows)
        write_csv(out/'probability_statistics.csv',stats_rows)
        write_csv(out/'per_slice.csv',z_rows)
        write_csv(out/'data_diagnostics.csv',data_rows)
        atomic_json(out/'progress.json',dict(processed_cases=case_number,total_cases=len(cases),complete=False))
        del sums
    summaries=[]
    for source in SOURCES:
        for fusion in FUSIONS:
            for mode in ('head_raw3d','head_post3d'):
                selected=[r for r in rows if r['logit_source']==source and r['slab_fusion']==fusion and r['mode']==mode]
                present=[r for r in selected if r['target_voxels']>0]
                summaries.append(dict(logit_source=source,slab_fusion=fusion,mode=mode,
                    **{key:float(np.mean([r[key] for r in present])) for key in ('dice','iou','precision','recall','prediction_to_target_volume_ratio')}))
    atomic_json(out/'results.json',dict(cases=cases,complete_test_set=len(cases)==len(case_indices),complete=True,elapsed_seconds=time.time()-started,summaries=summaries))
    atomic_json(out/'progress.json',dict(processed_cases=len(cases),total_cases=len(cases),complete=True))
