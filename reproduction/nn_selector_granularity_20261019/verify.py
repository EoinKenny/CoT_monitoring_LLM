from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
done=json.loads((O/'COMPLETED.json').read_text());freeze=json.loads((O/'SELECTION_FROZEN.json').read_text())
for p,v in done['inputs'].items():assert sha(Path(p))==v
for p,v in freeze['files'].items():assert sha(Path(p))==v
for p,v in freeze['models'].items():assert sha(Path(p))==v
for p,v in done['outputs'].items():assert sha(O/p)==v
c=pd.read_csv(O/'model_choices.csv');v=pd.read_csv(O/'validation_candidates.csv');cuts=pd.read_csv(O/'validation_cutoffs.csv');counts=pd.read_csv(O/'training_counts.csv');r=pd.read_csv(O/'cell_metrics.csv')
assert len(counts)==2480 and len(c)==9920
assert len(counts[counts.scope=='cell'])==2000
assert not c.duplicated(['model','policy','scope','group','variant']).any()
assert not r.duplicated(['model','policy','backbone','dataset','method']).any()
assert np.isfinite(r[['f1','precision','recall','accuracy','nn_rate']]).all().all()
assert ((r.f1>=-1e-9)&(r.f1<=1+1e-9)).all()
assert (counts.training_flips==counts.include+counts.exclude).all()
assert (counts.constant_final==((counts.include==0)|(counts.exclude==0))).all()
# Reproduce the old pooled candidate CV scores, not just test metrics.
old=pd.read_csv(P/'nn_knn_gate_20261019/results/validation_candidates.csv');q=v[v.scope=='pool_all'].merge(old,on=['model','policy','candidate'],validate='one_to_one');knnerror=float(abs(q.validation_f1-q.selection_validation_f1).max());assert len(q)==320 and knnerror<1e-10
oldlog=pd.read_csv(P/'nn_learned_gate_20261019/results/validation_candidates.csv');q=v[v.scope=='pool_all'].merge(oldlog,on=['model','policy','candidate'],validate='one_to_one');logerror=float(abs(q.validation_f1-q.selection_validation_f1).max());assert len(q)==80 and logerror<1e-10
# Every selected candidate must attain its family maximum within tolerance.
groups={k:g for k,g in v.groupby(['model','policy','scope','group'])}
for row in c.itertuples():
 h=groups[row.model,row.policy,row.scope,row.group];prefix='knn_' if row.variant=='knn' else row.variant+'_';h=h[h.candidate.str.startswith(prefix)];assert abs(row.validation_f1-h.validation_f1.max())<1e-10
# Verify whole aggregate support: toxicity pooling never mixed into all-10 averages.
s=pd.read_csv(O/'summary.csv');assert not s[(s.cohort=='all')&s.method.str.startswith('pool_toxicity')].shape[0]
for row in s[s.dimension=='overall'].itertuples():
 z=r[r.method==row.method];z=z[z.dataset.isin(['aegis','beavertails','toxicchat','toxigen'])] if row.cohort=='toxicity4' else z
 z=z[z.policy!='fixed50'] if row.policy=='anchor_average' else z[z.policy==row.policy]
 assert abs(z.f1.mean()-row.f1)<1e-12
receipt=dict(status='passed',model_bundles=2480,selected_configurations=len(c),old_knn_cv_replay_rows=320,old_knn_cv_max_error=knnerror,old_logistic_cv_replay_rows=80,old_logistic_cv_max_error=logerror,pooled_logistic_test_replay_rows=done['pooled_logistic_replay_rows'],pooled_logistic_test_max_error=done['pooled_logistic_max_error'],checks='Input, frozen configuration and model hashes; complete cell coverage; selected settings maximize validation family score; original pooled CV and logistic test replay; all overall aggregates recomputed.')
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
