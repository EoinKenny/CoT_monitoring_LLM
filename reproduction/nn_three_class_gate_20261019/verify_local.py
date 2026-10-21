from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;O=R/'results';P=R.parent/'nn_embedding_gate_20261019';f=pd.read_csv(P/'inputs/joined_features.csv.gz');counts=[]
for marker in O.glob('*/*/*/COMPLETED.json'):
 rel=marker.relative_to(O);b,m,pol,_=rel.parts;rec=json.loads(marker.read_text());old=json.loads((P/'results'/rel).read_text())
 for name,h in rec['files'].items():assert hashlib.sha256((marker.parent/name).read_bytes()).hexdigest()==h
 lab=pd.read_csv(marker.parent/'training_labels.csv');oldids=np.load(P/'results'/b/m/pol/'selected_oof.npz')['source_index'];assert np.array_equal(lab.source_index.to_numpy(),oldids);assert f.iloc[lab.source_index].split.eq('val').all();assert len(lab)==rec['training_rows']==old['validation_rows']
 assert int((lab.utility_class==1).sum())==rec['helpful']==old['helpful'];assert int((lab.utility_class==-1).sum())==rec['harmful']==old['harmful'];assert int((lab.utility_class==0).sum())==rec['unchanged']==len(lab)-old['training_disagreements'];assert not lab.source_index.duplicated().any();counts.append(len(lab))
assert len(counts)==200
for n,h in json.loads((O/'EVALUATION_COMPLETED.json').read_text())['files'].items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h
if (O/'ANALYSIS_COMPLETED.json').exists():
 for n,h in json.loads((O/'ANALYSIS_COMPLETED.json').read_text())['artifacts'].items():assert hashlib.sha256((O/n).read_bytes()).hexdigest()==h
p=pd.read_csv(O/'test_predictions.csv.gz');assert not p.duplicated(['model','policy','source_index']).any();a=pd.read_csv(O/'cell_metrics.csv');assert a.groupby(['policy','method']).size().eq(200).all();assert np.isfinite(a[['f1','precision','recall','accuracy']].to_numpy()).all();v=dict(verified_groups=200,all_validation_membership_exact=True,helpful_harmful_labels_match_parent=True,unchanged_labels_complete=True,all_training_rows_validation_only=True,checkpoint_and_evaluation_hashes=True,finite_metrics=True,test_cases=p.source_index.nunique(),test_targets=p.cluster.nunique(),fixed50_judge_context_rows=int((p.policy=='fixed50').sum()));(O/'VERIFIED.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
