import json,gzip,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parent.parent;R=Path(__file__).resolve().parent
A=np.load(P/'mechanisms_all12_20261023/annotations.npz');S=np.load(P/'main_performance_all12_20261023/scores.npz')
S={k:S[k] for k in S.files};A={k:A[k] for k in A.files}
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];lookup={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)}
mp=[json.loads(l) for l in open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl')];mi={(r['backbone'],r['sample_id']):i for i,r in enumerate(mp)}
api=['claude-opus-5.5','gpt-5.5','claude4.8-opus','gpt-astra','claude4.5-sonnet','gpt-4o'];out=[]
for l in gzip.open(P/'nn_mechanisms_20261015/features.jsonl.gz','rt'):
 r=json.loads(l)
 if r['split']!='test' or r['dataset'] not in ['toxicchat','toxigen'] or r['true_label']!=0:continue
 if r['target_chars']>350 or r['neighbor_chars']>350:continue
 i=lookup[r['backbone'],r['sample_id']];u=mi[r['backbone'],r['sample_id']];fix=[]
 for j in api:
  x,y=int(S[j][i,0]),int(S[j][i,2]);ji=list(A['models']).index(j)
  if x>=50 and 0<=y<50:
   fix.append(dict(judge=j,control=x,nn=y,labels=A['leave_one_out'][ji,u].tolist()))
 if fix:out.append(dict(**r,fixes=fix,majority=A['majority'][u].tolist()))
out.sort(key=lambda r:(-int(r['probe_probability']>=.5),r['target_chars']+r['neighbor_chars']))
(R/'api_contrast_reverse_candidates.json').write_text(json.dumps(out,indent=2))
q=[r for r in out if any(f['labels']==[0,1] and f['control']>=60 and f['nn']<=40 and f['judge']!='gpt-4o' for f in r['fixes'])]
q.sort(key=lambda r:(r['nearest_label']!=1,-int(r['probe_probability']>=.5),r['target_chars']+r['neighbor_chars']))
seen=set()
for r in q:
 key=str(r['target_messages'])+str(r['neighbor_messages'])
 if key in seen:continue
 seen.add(key)
 print(json.dumps(dict(dataset=r['dataset'],backbone=r['backbone'],sample_id=r['sample_id'],probe=round(r['probe_probability'],3),nn_label=r['nearest_label'],nn_prediction=r['nearest_prediction'],target=r['target_messages'],neighbor=r['neighbor_messages'],fixes=r['fixes'])))
print('all_candidates',len(out))
