from pathlib import Path
import json,gzip,hashlib,html
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';O.mkdir(exist_ok=True)
F=[json.loads(l) for l in gzip.open(P/'nn_mechanisms_20261015/features.jsonl.gz','rt')]
lookup={};maxerr=0
for bb,ds in sorted({(r['backbone'],r['dataset']) for r in F}):
 z=np.load(R/'inputs'/f'{bb}__{ds}.npz');z={k:z[k] for k in z.files};m=json.load(open(R/'inputs'/f'{bb}__{ds}.json'))
 for sid,row in m.items():
  i=row['index'];assert z['labels'][i]==row['label'] and z['splits'][i]==row['split'];v=float(z['probabilities'][i]);assert np.isfinite(v) and 0<=v<=1
  lookup[bb,sid]=(v,row['label'],row['split'])
for r in F:
 tp,ty,ts=lookup[r['backbone'],r['sample_id']];np_,ny,ns=lookup[r['backbone'],r['nearest_id']]
 assert ty==r['true_label'] and ts==r['split'] and ny==r['nearest_label'] and ns=='train'
 maxerr=max(maxerr,abs(tp-r['probe_probability']));r['neighbor_probe_probability']=np_
# Equality here checks matching saved score files, not regenerated replay agreement.
assert maxerr<1e-12,maxerr
A=np.load(P/'mechanisms_all12_20261023/annotations.npz');A={k:A[k] for k in A.files}
S=np.load(P/'main_performance_all12_20261023/scores.npz');S={k:S[k] for k in S.files}
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];ix={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)}
mp=[json.loads(l) for l in open(P/'nn_mechanism_agreement_20261019/inputs/mapping_PRIVATE.jsonl')];mi={(r['backbone'],r['sample_id']):i for i,r in enumerate(mp)}
F=[r for r in F if r['split']=='test'];idx=np.array([ix[r['backbone'],r['sample_id']] for r in F]);ai=np.array([mi[r['backbone'],r['sample_id']] for r in F]);
base=pd.DataFrame([{k:r[k] for k in ['backbone','dataset','sample_id','true_label','probe_probability','nearest_label','nearest_prediction','nearest_id','neighbor_probe_probability','inclusion_probability']} for r in F]);base['w']=1/base.inclusion_probability
frames=[];examples=[]
for ji,j in enumerate(A['models']):
 g=base.copy();g['judge']=j;g['control']=S[j][idx,0];g['nn']=S[j][idx,2];g['similarity']=A['leave_one_out'][ji,ai,0];g['contrast']=A['leave_one_out'][ji,ai,1]
 g=g[(g.control>=0)&(g.nn>=0)].copy();g['helped']=((g.control>=50)!=g.true_label)&((g.nn>=50)==g.true_label);g['harmed']=((g.control>=50)==g.true_label)&((g.nn>=50)!=g.true_label);g['unchanged']=(g.control>=50)==(g.nn>=50)
 g['anchor']=np.where(g.neighbor_probe_probability>=.8,'Positive NN (80–100)',np.where(g.neighbor_probe_probability<=.2,'Negative NN (0–20)','Middle NN (20–80)'));frames.append(g)
 if j not in ['qwen','gemma','mistral','olmo','glm','qwen122']:
  h=g[(g.control>=50)&(g.control<=65)&(g.nn<50)&(g.true_label==0)&(g.neighbor_probe_probability>=.8)&(g.contrast==1)]
  for n,r in h.iterrows():
   src=F[n];examples.append(dict(**src,judge=str(j),control=int(r.control),nn=int(r.nn),similarity=int(r.similarity),contrast=int(r.contrast)))
G=pd.concat(frames,ignore_index=True)
G.to_pickle(O/'joined.pkl')
summary=[];cells=[]
for scope,ds in [('All ten datasets',None),('Four toxicity datasets',['aegis','beavertails','toxicchat','toxigen']),('ToxicChat and ToxiGen',['toxicchat','toxigen'])]:
 for side in ['Positive control','Positive control 50–65','Negative control']:
  cond=(G.control.between(50,65) if side=='Positive control 50–65' else ((G.control>=50) if side=='Positive control' else (G.control<50)))
  z=G[(G.contrast==1)&cond]
  if ds:z=z[z.dataset.isin(ds)]
  for (j,b,d_,a),v in z.groupby(['judge','backbone','dataset','anchor']):
   w=v.w.to_numpy();cells.append(dict(scope=scope,side=side,judge=j,backbone=b,dataset=d_,anchor=a,n=len(v),helped=100*np.average(v.helped,weights=w),harmed=100*np.average(v.harmed,weights=w),unchanged=100*np.average(v.unchanged,weights=w)))
  cc=pd.DataFrame([x for x in cells if x['scope']==scope and x['side']==side])
  # Backbones, then datasets, then judges equally weighted; only populated cells contribute.
  m=cc.groupby(['judge','dataset','anchor'])[['helped','harmed','unchanged']].mean().groupby(['judge','anchor']).mean().groupby('anchor').mean()
  for a,row in m.iterrows():
   zz=z[z.anchor==a];summary.append(dict(scope=scope,side=side,anchor=a,rows=len(zz),contexts=zz[['backbone','sample_id']].drop_duplicates().shape[0],cells=len(cc[cc.anchor==a]),helped=row.helped,harmed=row.harmed,unchanged=row.unchanged,net=row.helped-row.harmed))
summary=pd.DataFrame(summary);summary.to_csv(O/'summary.csv',index=False);pd.DataFrame(cells).to_csv(O/'cell_rates.csv',index=False)
examples.sort(key=lambda r:(r['dataset'] not in ['toxicchat','toxigen'],r['similarity']!=0,r['probe_probability']<.5,r['target_chars']+r['neighbor_chars']))
(O/'extreme_positive_corrections.json').write_text(json.dumps(examples,indent=2))
with gzip.open(O/'context_probe_scores.jsonl.gz','wt') as f:
 for r in F:f.write(json.dumps({k:r[k] for k in ['backbone','dataset','sample_id','nearest_id','probe_probability','neighbor_probe_probability','nearest_label','true_label']})+'\n')
receipt=dict(contexts=len(F),paired_valid_judge_outcomes=len(G),contrast_labelled=int(G.contrast.eq(1).sum()),saved_target_score_max_difference=maxerr,api_positive_anchor_corrections_50_to_65=len(examples),new_inference_calls=0,new_probe_training=0,methods='Original saved probe scores, cutoff 50 for judge outcomes. High anchor >=80, low <=20 selected for this exploratory check, not optimised. Independent contrast consensus excludes outcome judge. Weighted within each backbone/dataset/judge group; average backbones, datasets, judges equally within anchor group. Empty groups omitted. Different anchor groups may have different case and cell composition; descriptive, not controlled causal effect.')
(O/'VERIFIED.json').write_text(json.dumps(receipt,indent=2))
e=html.escape
body='<h1>Do strongly positive and negative neighbours differ?</h1><p>Saved original probe probabilities joined to the existing 1-NN test judgments. No new inference or retraining. Only pairs independently labelled Contrasting NN are included in the tables.</p><p>80–100 and 0–20 are exploratory score bands, not fitted cutoffs. Probe probability measures confidence in a class, not semantic severity. Judges saw the neighbour text, class label, and binary probe prediction—not its numeric probability. Different bands have different cases and group composition. The tables describe associations.</p><p>Rates use design weights within backbone/dataset/judge, then equally average backbones, datasets and judges. Missing groups omitted. Correct/incorrect refer to benchmark labels. Unchanged includes both unchanged-correct and unchanged-wrong. A context can appear for multiple judges.</p>'
for scope in summary.scope.unique():
 body+=f'<h2>{scope}</h2>'
 for side in ['Positive control','Positive control 50–65','Negative control']:
  v=summary[(summary.scope==scope)&(summary.side==side)][['anchor','rows','contexts','cells','helped','harmed','unchanged','net']];body+=f'<h3>{side}</h3>'+v.to_html(index=False,float_format=lambda x:f'{x:.2f}')
body+='<h2>Saved API corrections with a strongly positive neighbour</h2><p>Control probability 50–65, NN condition below 50, target gold negative, neighbour probe at least 80. First 30 candidates in the saved audit ordering. These are not all publication-friendly; full text is retained for inspection.</p>'
for i,r in enumerate(examples[:30],1):
 body+=f'<details><summary>{i}. {r["judge"]} / {r["dataset"]} / {r["backbone"]}: {r["control"]} → {r["nn"]}, neighbour probe {100*r["neighbor_probe_probability"]:.2f}%</summary><p>Target probe {100*r["probe_probability"]:.2f}%; target gold 0; NN label {r["nearest_label"]}; NN frozen prediction {r["nearest_prediction"]}; Similar NN {r["similarity"]}; Contrasting NN {r["contrast"]}.</p><b>Target</b><pre>'+e('\n'.join(x['content'] for x in r['target_messages']))+'</pre><b>Neighbour</b><pre>'+e('\n'.join(x['content'] for x in r['neighbor_messages']))+'</pre></details>'
(O/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>NN extremity check</title><style>body{max-width:1300px;margin:35px auto;font:16px/1.5 system-ui;color:#243141}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:#f3f6f8;padding:14px}details{margin:15px 0}</style>'+body)
print(summary[summary.side=='Positive control'].to_string(index=False));print(json.dumps(receipt,indent=2))
for i,r in enumerate(examples[:20]):print('EXAMPLE',i,r['judge'],r['dataset'],r['backbone'],r['control'],r['nn'],round(100*r['neighbor_probe_probability'],2),round(100*r['probe_probability'],2),r['target_messages'],r['neighbor_messages'])
