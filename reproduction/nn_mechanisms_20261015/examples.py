from pathlib import Path
import json,gzip,pandas as pd,numpy as np
R=Path(__file__).resolve().parent;O=R/'results'
d=pd.read_csv(O/'case_mechanisms.csv.gz');d=d[(d.scope=='common')&d.model.isin(['qwen','gemma','qwen122','claude4.8-opus','claude4.5-sonnet','gpt-astra','gpt-4o'])]
a=d.groupby(['backbone','sample_id']).agg(mean_gain=('correctness_gain','mean'),nhelp=('correctness_gain',lambda x:int((x>0).sum())),nhurt=('correctness_gain',lambda x:int((x<0).sum())),mean_shift=('score_delta','mean'))
lookup={(r['backbone'],r['sample_id']):r for r in (json.loads(line) for line in gzip.open(R/'features.jsonl.gz','rt'))}
selected=[]
for ds in ['toxicchat','beavertails','high_stakes','ragtruth','toxigen','wildguard']:
 for mode in ['help','hurt']:
  candidates=[(key,row,lookup[key]) for key,row in a.iterrows() if lookup[key]['dataset']==ds and lookup[key]['target_chars']<=2000 and lookup[key]['neighbor_chars']<=2000 and (row.mean_gain>0 if mode=='help' else row.mean_gain<0)]
  candidates.sort(key=lambda x:((-x[1].mean_gain if mode=='help' else x[1].mean_gain),-(x[1].nhelp if mode=='help' else x[1].nhurt),x[0]))
  for key,row,r in candidates[:1]:
   s=d[(d.backbone==key[0])&(d.sample_id==key[1])][['model','base_score','nn_score','correctness_gain']].to_dict('records')
   selected.append(dict(category=mode,**r,mean_correctness_gain=row.mean_gain,judge_scores=s))
(O/'matched_examples.json').write_text(json.dumps(selected,indent=2))
for r in selected:
 print('\n',r['dataset'],r['category'],r['backbone'],'gold',r['true_label'],'NNlabel',r['nearest_label'],'cos',round(r['nearest_cosine'],3))
 print('TARGET',r['target_messages']);print('NEIGHBOR',r['neighbor_messages']);print('scores',[(x['model'],x['base_score'],x['nn_score']) for x in r['judge_scores']])
