"""Per-organ Offline differences and external PCDD group references; never invent class scores."""
import argparse
import csv
import json
import math
from pathlib import Path

SIZES = {'2222': (2,2,2,2), 'pcdd_44': (4,4), 'pcdd_422': (4,2,2),
         'pcdd_41111': (4,1,1,1,1), 'pcdd_71': (7,1), 'pcdd_offline': (8,)}
PCDD = {'WORD': {'pcdd_44': (84.09,82.75,83.42), 'pcdd_422': (83.95,83.04,83.50),
                  '2222': (90.17,65.97,72.02), 'pcdd_71': (71.51,82.59,72.89)},
        'BTCV': {'pcdd_44': (92.51,88.69,90.60), 'pcdd_422': (92.45,88.48,90.46),
                  'pcdd_41111': (92.27,88.83,90.55), 'pcdd_71': (90.15,93.37,90.55)}}
MODES=('binary_raw','binary_post','selected_raw','selected_post')

def load(path, ids):
    d=json.loads(path.read_text())
    if not d.get('complete_split') or d.get('split')!='test' or not d['checkpoint_load']['exact']:
        raise ValueError('Require complete exact test evaluation')
    rows=list(csv.DictReader(path.with_name('per_case.csv').open()))
    cohort=None
    for mode in MODES:
        metrics=d['metrics'][mode]
        if set(int(k) for k in metrics if k.isdigit() and int(k))!=set(ids):
            raise ValueError('Unexpected organ set')
        for i in ids:
            selected=[r for r in rows if r['mode']==mode and int(r['class_id'])==i]
            current={r['case_id']:int(r['slices']) for r in selected}
            if not current or len(current)!=len(selected) or (cohort is not None and current!=cohort):
                raise ValueError('Inconsistent case cohort')
            cohort=current
            value=float(metrics[str(i)]['case_dice_presence_mean'])
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid Dice')
    return d,cohort


def report(result, protocol, stage, dataset, offline):
    sizes=SIZES[protocol]
    if not 1<=stage<=len(sizes):raise ValueError('Invalid stage')
    ids=list(range(1,sum(sizes[:stage])+1));old=ids[:sizes[0]];new=ids[sizes[0]:]
    d,cohort=load(result,ids)
    baseline=None
    if offline.exists():
        baseline,base_cohort=load(offline,list(range(1,9)))
        if base_cohort!=cohort:raise ValueError('Offline test cohort mismatch; refuse difference')
    mean=lambda values,keys:sum(values[k] for k in keys)/len(keys) if keys else None
    output={'dataset':dataset,'protocol':protocol,'stage':stage,'source':str(result),
            'offline_source':str(offline),'offline_status':'available; verify training/initialization protocol alongside report' if baseline else 'pending',
            'cohort':cohort,'pcdd_source':'https://arxiv.org/html/2511.07749v1#S4',
            'pcdd_comparison':'external reference, not matched split/preprocessing/architecture; no verified per-organ values',
            'pcdd_final_groups':PCDD[dataset].get(protocol) if stage==len(sizes) else None,'scores':{}}
    for mode in MODES:
        values={i:100*d['metrics'][mode][str(i)]['case_dice_presence_mean'] for i in ids}
        off={i:100*baseline['metrics'][mode][str(i)]['case_dice_presence_mean'] for i in ids} if baseline else None
        groups={k:mean(values,keys) for k,keys in [('old',old),('new',new),('all',ids)]}
        entry={'groups':groups,'per_organ':{str(i):{'name':d['metrics'][mode][str(i)]['class_name'], 'dice':values[i],
               'offline':off[i] if off else None,'delta_offline_pp':values[i]-off[i] if off else None,
               'pcdd':None,'delta_pcdd_pp':None} for i in ids}}
        if off:
            entry['offline_groups']={k:mean(off,keys) for k,keys in [('old',old),('new',new),('all',ids)]}
            entry['delta_offline_pp']={k:groups[k]-v if v is not None else None for k,v in entry['offline_groups'].items()}
        if output['pcdd_final_groups']:
            entry['delta_pcdd_reference_pp']={k:groups[k]-v for k,v in zip(('old','new','all'),output['pcdd_final_groups'])}
        output['scores'][mode]=entry
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('result',type=Path);p.add_argument('--protocol',choices=SIZES,required=True)
    p.add_argument('--stage',type=int,required=True);p.add_argument('--dataset',choices=PCDD,required=True);p.add_argument('--offline',type=Path,required=True)
    a=p.parse_args();print(json.dumps(report(a.result,a.protocol,a.stage,a.dataset,a.offline),indent=2,allow_nan=False))
