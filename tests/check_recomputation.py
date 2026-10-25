"""Compare regenerated numeric tables with the frozen paper release."""
from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
a=ROOT/'data/paper/plots';b=ROOT/'outputs/recomputed_check/recomputed/plots';checked={}
for filename,keys,values in [
 ('main_cells.csv.gz',['model','backbone','dataset','policy','anchor','condition'],['precision','recall','f1','accuracy','probe_threshold','judge_threshold']),
 ('ablation_cells.csv.gz',['model','backbone','dataset','condition'],['precision','recall','f1','accuracy']),
 ('main_curves.csv.gz',['policy','dimension','group','condition','operating_index'],['mean_test_recall','mean_test_precision']),
 ('label_cells.csv.gz',['judge','backbone','dataset','label','outcome'],['rate']),
 ('label_prevalence.json',['label','outcome'],['mean','lo','hi']),
 ('contrast_bins.json',['scope','left','right'],['helped','harmed','net','net_lo','net_hi'])]:
 reader=pd.read_json if filename.endswith('.json') else pd.read_csv
 x=reader(a/filename).set_index(keys).sort_index();y=reader(b/filename).set_index(keys).sort_index()
 assert x.index.is_unique and y.index.is_unique
 # Original table may omit unsupported outcome groups; neither treats them as zero.
 common=x.index.intersection(y.index)
 assert len(common)==len(x),(filename,len(common),len(x))
 delta=np.abs(x.loc[common,values].to_numpy()-y.loc[common,values].to_numpy())
 assert np.allclose(x.loc[common,values],y.loc[common,values],atol=1e-7,rtol=1e-10,equal_nan=True),(filename,np.nanmax(delta))
 checked[filename]=dict(rows=len(common),max_absolute_difference=float(np.nanmax(delta)))
x=json.loads((a/'trace_joints.json').read_text());y=json.loads((b/'trace_joints.json').read_text())
for key,value in y.items():assert np.allclose(x[key],value,atol=1e-12,rtol=0),(key,x[key],value)
checked['trace_joints.json']={'tables_checked':len(y)}
(ROOT/'docs/release/NUMERICAL_VALIDATION.json').write_text(json.dumps(checked,indent=2)+'\n')
print(json.dumps(checked,indent=2))
