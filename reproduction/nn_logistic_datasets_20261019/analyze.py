from pathlib import Path
import json,gzip,hashlib
import pandas as pd,numpy as np
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';OLD=P/'nn_learned_gate_20261019/results'
s=pd.read_csv(OLD/'summary.csv');v=pd.read_csv(OLD/'validation_candidates.csv');v=v[(v.policy=='fixed50')&(v.candidate=='logistic_C0.1')].copy();v['training_dont_include']=v.training_flips-v.training_helpful
f=pd.read_csv(P/'nn_mechanisms_20261015/results/joined_features.csv.gz');scores=np.load(P/'api_results_20261015/results/scores.npz');val=f[f.split=='val'];trainrows=[]
for m in scores.files:
 ix=val.index.to_numpy();ok=(scores[m][ix]>=0).all(axis=1);g=val[ok];ix=g.index.to_numpy();ref=g.probe_probability.to_numpy()>=.5;bp=ref&(scores[m][ix,0]>=50);npred=ref&(scores[m][ix,2]>=50);flip=bp!=npred;y=g.true_label.to_numpy();h=g[flip].copy();h['model']=m;h['source_index']=h.index;h['include']=(npred[flip]==y[flip]).astype(int);trainrows.append(h[['model','source_index','backbone','dataset','cluster','include']]);assert len(h)==int(v[v.model==m].training_flips.iloc[0])
t=pd.concat(trainrows,ignore_index=True);n=t.groupby('dataset').agg(training_pairs=('source_index','size'),unique_contexts=('source_index','nunique'),unique_targets=('cluster','nunique'),include=('include','sum')).reset_index();n['dont_include']=n.training_pairs-n.include
n['available_validation_contexts']=n.dataset.map(val.groupby('dataset').size());n['available_test_contexts']=n.dataset.map(f[f.split=='test'].groupby('dataset').size())
rows=[]
for pol in ['fixed50','anchor_average']:
 z=s[(s.cohort=='all')&(s.dimension=='dataset')&(s.policy==pol)]
 for ds,g in z.groupby('group'):
  g=g.set_index('method');rows.append(dict(dataset=ds,policy=pol,control_f1=g.loc['control','f1'],always_nn_f1=g.loc['always_nn','f1'],cosine_f1=g.loc['cosine_gate','f1'],logistic_f1=g.loc['logistic_gate','f1'],delta_always=g.loc['logistic_gate','f1']-g.loc['always_nn','f1'],delta_cosine=g.loc['logistic_gate','f1']-g.loc['cosine_gate','f1'],precision=g.loc['logistic_gate','precision'],recall=g.loc['logistic_gate','recall'],accuracy=g.loc['logistic_gate','accuracy']))
z=pd.DataFrame(rows)
for name,x in [('dataset_performance',z),('training_by_dataset',n),('training_by_judge',v[['model','validation_rows','training_flips','training_helpful','training_dont_include']])]:x.to_csv(O/(name+'.csv'),index=False)
receipt=dict(retrained=False,new_llm_calls=0,selector='One pooled logistic model per judge/policy, trained across all10 datasets and5 backbones. Hyperparameter and per-cell cutoffs selected through grouped validation CV.',primary_policy='Probe50 and judge50',validation_contexts=len(val),validation_distinct_targets=val.cluster.nunique(),training_judge_context_pairs=len(t),training_unique_contexts=t.source_index.nunique(),training_distinct_targets=t.cluster.nunique(),test_contexts=int((f.split=='test').sum()),include_training_pairs=int(t.include.sum()),dont_include_training_pairs=int(len(t)-t.include.sum()),sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'summary.csv',OLD/'validation_candidates.csv',P/'api_results_20261015/results/scores.npz']})
(O/'COMPLETED.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));print(n.to_string(index=False));print(z.to_string(index=False))
