"""Repartition the same 30 native cases; never regenerate or overwrite pixels."""
import csv, json, random, hashlib, argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--seed',type=int,default=43);a=p.parse_args()
assert not a.output.exists(), 'refuse overwrite'
rows=[]
for split in ['Training','Test']:
 with open(a.source/'csv'/('BTCV_'+split+'_2D_native07072.csv')) as f:rows+=list(csv.DictReader(f))
by={}
for r in rows:by.setdefault(Path(r['image_pth']).parent.name,[]).append(r)
assert len(by)==30
cases=sorted(by);random.Random(a.seed).shuffle(cases);test=set(cases[:6]);train=set(cases[6:]);assert not train&test
(a.output/'csv').mkdir(parents=True);(a.output/'metadata').mkdir()
report={'seed':a.seed,'algorithm':'sorted case IDs, random.Random(seed).shuffle; first6 test','splits':{'Training':sorted(train),'Test':sorted(test)},'manifest_sha256':{}}
for split,ids in [('Training',train),('Test',test)]:
 path=a.output/'csv'/('BTCV_'+split+'_2D_native07072.csv')
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader()
  for c in sorted(ids):w.writerows(by[c])
 report['manifest_sha256'][split]=hashlib.sha256(path.read_bytes()).hexdigest()
summary=json.loads((a.source/'metadata/preprocess_summary.json').read_text());summary['config']['runtime_final_input_xy']=[512,512]
for d in summary['case_details']:d['split']='Test' if d['case_id'] in test else 'Training'
summary['previous_split_counts']=summary.pop('splits', None)
summary['splits']={name:{'case_count':len(ids),'slice_count':sum(len(by[c]) for c in ids)} for name,ids in [('Training',train),('Test',test)]}
summary['split_override']=report
(a.output/'metadata/preprocess_summary.json').write_text(json.dumps(summary,indent=2))
(a.output/'split.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
