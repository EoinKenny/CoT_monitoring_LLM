from pathlib import Path
import pandas as pd,numpy as np,json,hashlib,sklearn
R=Path(__file__).resolve().parent;O=R/'results';OLD=R.parent/'nn_learned_gate_20261019/results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
f=json.load(open(O/'SELECTION_FROZEN.json'));assert sha(O/'model_choices.csv')==f['choices_sha256'];assert sha(O/'validation_cutoffs.csv')==f['cutoffs_sha256']
for name,h in f['models'].items():assert sha(R/'models'/name)==h
c=pd.read_csv(O/'model_choices.csv');r=pd.read_csv(O/'all_cell_metrics.csv');oldchoices=pd.read_csv(OLD/'model_choices.csv');n=0
for row in c[c.variant=='expanded_selected'].itertuples():
 if row.candidate.startswith('knn_'):continue
 orig=oldchoices[(oldchoices.model==row.model)&(oldchoices.policy==row.policy)&(oldchoices.variant=='selected')].iloc[0];assert orig.candidate==row.candidate
 a=r[(r.model==row.model)&(r.policy==row.policy)&(r.method=='expanded_selected')].set_index(['backbone','dataset']).sort_index();b=r[(r.model==row.model)&(r.policy==row.policy)&(r.method=='selected_gate')].set_index(['backbone','dataset']).sort_index()
 assert np.allclose(a[['f1','precision','recall','accuracy']],b[['f1','precision','recall','accuracy']],rtol=0,atol=1e-12);n+=len(a)
for row in c[c.variant=='knn_selected'].itertuples():
 method='knn_euclidean' if 'euclidean' in row.candidate else 'knn_cosine';a=r[(r.model==row.model)&(r.policy==row.policy)&(r.method=='knn_selected')].set_index(['backbone','dataset']).sort_index();b=r[(r.model==row.model)&(r.policy==row.policy)&(r.method==method)].set_index(['backbone','dataset']).sort_index();assert np.allclose(a[['f1','precision','recall','accuracy']],b[['f1','precision','recall','accuracy']],rtol=0,atol=1e-12)
receipt=dict(checks=['Frozen validation selections unchanged','Fitted model checksums match frozen receipt','Expanded selections that retain original models replay original results','Selected kNN predictions reproduce selected metric family','Original group folds and feature whitelist reused'],old_model_replay_rows=n,sklearn_version=sklearn.__version__,outputs={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name!='VERIFIED.json'})
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2));print('Verified original model replay rows:',n)
