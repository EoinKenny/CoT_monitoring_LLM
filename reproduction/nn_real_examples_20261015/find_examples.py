from pathlib import Path
import json,gzip,hashlib,numpy as np,pandas as pd
R=Path(__file__).resolve().parent;M=R.parent/'nn_mechanisms_20261015';A=R.parent/'api_results_20261015'
manifest=json.load(open(M/'EXPORT_COMPLETED.json'));assert hashlib.sha256((M/'features.jsonl.gz').read_bytes()).hexdigest()==manifest['files']['features.jsonl.gz']
f=pd.read_csv(M/'results/joined_features.csv.gz');s=np.load(A/'results/scores.npz');models=s.files;strong=[models.index(x) for x in ['qwen','gemma','qwen122','claude4.8-opus','claude4.5-sonnet','gpt-astra','gpt-4o']]
# Confirm rows match the authoritative frozen score order.
design=json.load(gzip.open(A/'frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b in design['samples'] for r in design['samples'][b]]
assert list(zip(f.backbone,f.sample_id))==[(r['backbone'],r['sample_id']) for r in cases]
b=np.stack([s[m][:,0] for m in models],axis=1);n=np.stack([s[m][:,2] for m in models],axis=1);y=f.true_label.to_numpy()[:,None]
valid=np.stack([(s[m]>=0).all(axis=1) for m in models],axis=1).all(axis=1)
bc=(b>=50)==y;nc=(n>=50)==y;helped=~bc&nc;hurt=bc&~nc
f['helped']=helped.sum(axis=1);f['hurt']=hurt.sum(axis=1);f['strong_helped']=helped[:,strong].sum(axis=1);f['strong_hurt']=hurt[:,strong].sum(axis=1);f['base_correct']=bc.sum(axis=1);f['nn_correct']=nc.sum(axis=1)
f['judge_scores']=[{m:dict(control=int(b[i,j]),nearest=int(n[i,j]),change='helped' if helped[i,j] else 'hurt' if hurt[i,j] else 'unchanged') for j,m in enumerate(models)} for i in range(len(f))]
f['combined_chars']=f.target_chars+f.neighbor_chars
rows=f[valid & f.split.eq('test') & f.hurt.eq(0) & f.strong_helped.ge(2)].copy()
rows=rows.sort_values(['dataset','strong_helped','helped','nn_correct','combined_chars','sample_id','backbone'],ascending=[True,False,False,False,True,True,True])
rows.drop(columns='judge_scores').to_csv(R/'eligible_cases.csv',index=False)
top=rows.groupby('dataset',sort=False).head(5);want=set(zip(top.backbone,top.sample_id));texts={}
for line in gzip.open(M/'features.jsonl.gz','rt'):
 r=json.loads(line);key=(r['backbone'],r['sample_id'])
 if key in want:texts[key]=r
out=[]
for _,r in top.iterrows():
 z=texts[r.backbone,r.sample_id];out.append(dict(z,helped=int(r.helped),hurt=int(r.hurt),strong_helped=int(r.strong_helped),base_correct=int(r.base_correct),nn_correct=int(r.nn_correct),judge_scores=r.judge_scores))
(R/'candidates.json').write_text(json.dumps(out,indent=2,ensure_ascii=False))
print(rows.groupby('dataset').size().to_string());print('\nTOP CANDIDATES')
for ds,g in top.groupby('dataset'):
 print(ds,g[['backbone','sample_id','helped','strong_helped','base_correct','nn_correct','target_chars','neighbor_chars']].to_string(index=False))
