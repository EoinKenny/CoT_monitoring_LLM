from pathlib import Path
import pandas as pd,numpy as np,json,gzip
R=Path(__file__).resolve().parent;M=R.parent/'nn_mechanisms_20261015';d=pd.read_csv(R.parent/'nn_boundary_analysis_20261015/results/cases.csv.gz');d=d[d.flip].copy()
# Mainstream judges for interpretability; require neighbour alignment matching the observed dominant pattern.
strong=['qwen','gemma','qwen122','claude4.8-opus','claude4.5-sonnet','gpt-astra','gpt-4o']
z=d[d.model.isin(strong)];out=[]
for (flag,cat),g in z.groupby(['flag_type','category']):
 aligned=int(cat=='wrong_to_correct');g=g[g.aligned.eq(aligned)]
 for (bb,sid),v in g.groupby(['backbone','sample_id']):
  r=v.iloc[0];allr=d[(d.backbone==bb)&(d.sample_id==sid)];out.append(dict(flag_type=flag,category=cat,backbone=bb,sample_id=sid,dataset=r.dataset,cosine=float(r.nearest_cosine),true_label=int(r.true_label),nearest_label=int(r.nearest_label),nearest_prediction=int(r.nearest_prediction),chars=int(r.target_chars+r.neighbor_chars),judges=len(v),opposing_flips=int((allr.category!=cat).sum()),scores={rr.model:dict(control=float(rr.control_score*100),nearest=float(rr.nn_score*100)) for rr in v.itertuples()}))
f=pd.DataFrame(out);f['readable']=f.chars.le(1800)
f=f.sort_values(['flag_type','category','readable','judges','opposing_flips','chars'],ascending=[True,True,False,False,True,True]);top=f.groupby(['flag_type','category']).head(8);want=set(zip(top.backbone,top.sample_id));texts={}
for line in gzip.open(M/'features.jsonl.gz','rt'):
 r=json.loads(line);k=(r['backbone'],r['sample_id'])
 if k in want:texts[k]=r
result=[]
for r in top.to_dict('records'):result.append(dict(texts[r['backbone'],r['sample_id']],**{k:v for k,v in r.items() if k not in ['backbone','sample_id','dataset','true_label','nearest_label','nearest_prediction']}))
(R/'candidates.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for key,g in top.groupby(['flag_type','category']):
 print('\n',key)
 for i,r in enumerate(g.to_dict('records')):
  t=texts[r['backbone'],r['sample_id']];print(i,r['dataset'],r['backbone'],'cos',r['cosine'],'gold/NNgold/NNpred',r['true_label'],r['nearest_label'],r['nearest_prediction'],'judges',r['judges'],'opposite',r['opposing_flips']);print('TARGET',t['target_messages']);print('NN',t['neighbor_messages']);print('SCORES',r['scores'])
