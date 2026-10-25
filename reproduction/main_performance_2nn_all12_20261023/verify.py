from pathlib import Path
import csv,json,collections,math
R=Path(__file__).resolve().parent;O=R/'results'
rows=list(csv.DictReader(open(O/'cells.csv')));thresholds=list(csv.DictReader(open(O/'thresholds.csv')))
assert len(rows)==7200 and len(thresholds)==1800
by=collections.defaultdict(list)
for r in rows:
 for k in ['precision','recall','f1','accuracy']:assert math.isfinite(float(r[k])) and -1e-12<=float(r[k])<=1+1e-12
 by[tuple(r[k] for k in ['model','backbone','dataset','policy','anchor'])].append(r)
for g in by.values():
 assert {r['condition'] for r in g}=={'baseline','saliency','nearest2'}
 for k in ['probe_threshold','judge_threshold','validation_n','test_n']:assert len({r[k] for r in g})==1
prior=list(csv.DictReader(open(R.parent/'two_nn_all12_20261023/results/cells.csv')));ix={tuple(r[k] for k in ['model','backbone','dataset','condition']):r for r in prior};count=0
for r in rows:
 if r['scope']=='local_full' and r['policy']=='fixed50' and r['condition'] in ['baseline','nearest2']:
  old=ix[tuple(r[k] for k in ['model','backbone','dataset','condition'])]
  for k in ['precision','recall','f1','accuracy']:assert abs(float(r[k])-float(old[k]))<1e-12
  count+=1
v=json.load(open(O/'VERIFIED.json'));v.update(shared_threshold_groups_checked=len(by),fixed_local_rows_replayed=count,score_inputs=json.load(open(R/'SCORE_INPUTS.json')));(O/'VERIFIED.json').write_text(json.dumps(v,indent=2))
for r in csv.DictReader(open(O/'summary.csv')):
 if r['scope']=='combined' and r['dimension']=='overall' and r['policy'] in ['fixed50','average_of_three']:print(r['cohort'],r['policy'],*[round(float(r[k])*100,3) for k in ['baseline_f1','saliency_f1','nearest2_f1','nn_minus_control_f1']])
print('Verified',len(by),'shared groups',count,'prior rows')
