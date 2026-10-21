"""Synthetic analytical checks: class coverage, census counts and untouched sample metrics."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
from missing_strata import select_supplement
HERE=Path(__file__).parent
def row(i,y):return dict(sample_id=str(i),dataset='synthetic',true_label=y)
pop=[row(0,0),row(1,1),row(2,1)]
assert select_supplement(pop,pop)[0]==[]
assert select_supplement(pop,pop[1:])[0]==pop[:1]
assert select_supplement(pop,pop[:1])[0]==pop[1:]
assert select_supplement([],[])[0]==[]
assert select_supplement(pop,[])[0]==pop
for missing_label in [0,1]:
 with tempfile.TemporaryDirectory() as tmp:
  r=Path(tmp);sample=[row('s1',1-missing_label),row('s2',1-missing_label)];supp=[row('extra',missing_label)]
  # Sample the observed class in full; both class-specific outcomes are exact.
  b=dict(tp=2 if missing_label==0 else 1,fp=1 if missing_label==0 else 2,fn=1,tn=3,f1=.5)
  (r/'private.json').write_text(json.dumps(sample));(r/'weighted_results.json').write_text(json.dumps(dict(results={'synthetic':dict(test=b)})))
  (r/'SUPPLEMENT_MANIFEST.json').write_text(json.dumps(dict(frozen=dict(rows=supp))))
  for sub,rr in [('judges',sample),('supplement/judges',supp)]:
   p=r/sub/'qwen_baseline/run';p.mkdir(parents=True);(p/'COMPLETED').write_text('{}')
   # Judge accepts every case: the full cascade must equal the original probe counts.
   (p/'judgments.jsonl').write_text(''.join(json.dumps(dict(x,prediction=1))+'\n' for x in rr))
  subprocess.run([sys.executable,str(HERE/'evaluate.py'),'qwen_baseline'],env=dict(os.environ,EXPERIMENT_ROOT=tmp),check=True,stdout=subprocess.DEVNULL)
  out=json.loads((r/'qwen_baseline_results.json').read_text())['results']['synthetic']['qwen_baseline']
  assert all(out['cascade_estimate'][k]==b[k] for k in ['tp','fp','fn','tn'])
  assert out['cascade_f1_ci95'][0]==out['cascade_f1_ci95'][1]
  assert out['sampled_flags']==2 and out['supplemental_flags']==1
  assert out['flagged_sample_metrics']['tp']+out['flagged_sample_metrics']['fp']==2
  # Missing supplementary judgments must fail instead of inventing an acceptance rate.
  (r/'supplement/judges/qwen_baseline/run/COMPLETED').unlink()
  p=subprocess.run([sys.executable,str(HERE/'evaluate.py'),'qwen_baseline'],env=dict(os.environ,EXPERIMENT_ROOT=tmp),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  assert p.returncode!=0
print('PASS: both missing classes, census selection, empty populations, exact cascade counts, fixed census intervals, unchanged original sample metrics, fail-closed missing judgments.')
