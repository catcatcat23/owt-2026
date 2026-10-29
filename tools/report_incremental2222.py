"""PCDD 2-2: Old=initial two, New=all subsequent learned organs, not last pair."""
import argparse
import csv
import json
import math
from pathlib import Path


def report(path, stage):
    r = json.loads(path.read_text())
    if not r['complete_split'] or r['split'] != 'test' or not r['checkpoint_load']['exact']:
        raise ValueError('Require complete, exact test evaluation')
    rows = list(csv.DictReader(path.with_name('per_case.csv').open()))
    expected = set(range(1, 2*stage+1))
    result = {'stage': stage, 'source': str(path), 'old_definition': [1,2],
              'new_definition': list(range(3, 2*stage+1)), 'scores': {}}
    reference = None
    for mode in ('binary_raw','binary_post','selected_raw','selected_post'):
        metrics = r['metrics'][mode]
        if {int(k) for k in metrics if str(k).isdigit() and int(k)} != expected:
            raise ValueError('Unexpected classes')
        scores = {i: float(metrics[str(i)]['case_dice_presence_mean'])*100 for i in expected}
        if not all(math.isfinite(v) and 0 <= v <= 100 for v in scores.values()):
            raise ValueError('Invalid Dice')
        for i in expected:
            selected = [x for x in rows if x['mode']==mode and int(x['class_id'])==i]
            cohort = {x['case_id']:int(x['slices']) for x in selected}
            if len(selected)!=24 or len(cohort)!=24 or (reference is not None and reference!=cohort):
                raise ValueError('Inconsistent 24-case cohort')
            reference=cohort
        mean=lambda ids: sum(scores[i] for i in ids)/len(ids)
        result['scores'][mode]={'old2':mean([1,2]), 'new_cumulative':mean(list(range(3,2*stage+1))) if stage>1 else None,
            'all_seen':mean(list(expected)), 'per_organ':scores,
            'pairs': {'stage'+str(s):mean([2*s-1,2*s]) for s in range(1,stage+1)}}
    result['cohort']=reference
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('result',type=Path);p.add_argument('stage',type=int,choices=range(1,5))
    a=p.parse_args();print(json.dumps(report(a.result,a.stage),indent=2))
