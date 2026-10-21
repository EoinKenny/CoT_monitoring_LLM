import json,hashlib,collections
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];I=P/'analysis_inputs';O=P/'results';O.mkdir(exist_ok=True)
def jl(p):
 with p.open() as f:return [json.loads(l) for l in f]
targets=jl(I/'private/targets.jsonl'); cells=collections.defaultdict(list)
for t in targets:cells[t['arm'],t['backbone'],t['dataset'],t['split']].append(t)
TOX={'aegis','beavertails','toxicchat','toxigen'}
def metric(y,p,w):
 tp=np.sum(w*(p&y));fp=np.sum(w*(p&~y));fn=np.sum(w*(~p&y));return dict(f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,precision=float(tp/(tp+fp)) if tp+fp else 0.,recall=float(tp/(tp+fn)) if tp+fn else 0.,accuracy=float(np.sum(w*(y==p))/sum(w)))
results=[];thresholds=[];inventory=[];controlled=[]
for j in ['qwen122','gemma','glm']:
 raw=I/'runs/natural'/j/'responses.jsonl';done=json.loads(raw.with_name('COMPLETED.json').read_text());assert hashlib.sha256(raw.read_bytes()).hexdigest()==done['responses_sha256'];rows=jl(raw);lookup={(r['arm'],r['backbone'],r['sample_id'],r['condition']):r for r in rows};assert len(lookup)==len(rows)==12032 and all(r['status']=='valid' for r in rows)
 for arm,b,d,sp in cells:
  if sp!='test':continue
  conds=['activation_reference','semantic_reference'];conds=(['baseline']+conds) if arm=='fresh' else conds
  arrays={}
  for split in ['val','test']:
   ts=cells[arm,b,d,split];y=np.array([t['true_label'] for t in ts],bool);w=np.array([t['weight'] for t in ts]);probe=np.array([t['probe_probability']*100 for t in ts]);scores={c:np.array([lookup[arm,b,t['sample_id'],c]['positive_probability'] for t in ts]) for c in conds};arrays[split]=(y,w,probe,scores)
  y,w,probe,scores=arrays['test'];inventory.append(dict(judge=j,arm=arm,backbone=b,dataset=d,n_test=len(y),n_positive=int(y.sum()),n_val=len(arrays['val'][0])))
  for c in conds:
   for policy,pred in [('fixed50',(probe>=50)&(scores[c]>=50)),('direct50',scores[c]>=50)]:results.append(dict(judge=j,arm=arm,backbone=b,dataset=d,condition=c,policy=policy,anchor='',**metric(y,pred,w)))
  vy,vw,vp,vs=arrays['val']
  if len(vy)==0:continue
  for anchor in conds:
   best=(-1,-1,-1)
   for pt in range(102):
    preds=(vp[:,None]>=pt)&(vs[anchor][:,None]>=np.arange(102));tp=np.sum(vw[:,None]*(preds&vy[:,None]),axis=0);fp=np.sum(vw[:,None]*(preds&~vy[:,None]),axis=0);fn=np.sum(vw[:,None]*(~preds&vy[:,None]),axis=0);den=2*tp+fp+fn;f=np.divide(2*tp,den,out=np.zeros(102),where=den>0);qt=int(np.flatnonzero(f==f.max())[-1]);cand=(float(f[qt]),pt,qt)
    if cand>best:best=cand
   _,pt,qt=best;thresholds.append(dict(judge=j,arm=arm,backbone=b,dataset=d,anchor=anchor,probe=pt,judge_threshold=qt,val_f1=best[0]))
   for c in conds:results.append(dict(judge=j,arm=arm,backbone=b,dataset=d,condition=c,policy='shared_tuned',anchor=anchor,**metric(y,(probe>=pt)&(scores[c]>=qt),w)))
 labels={r['prompt_id']:r['target_label_under_explicit_rule'] for r in jl(I/'private/controlled_labels.jsonl')};cr=jl(I/'runs/controlled'/j/'responses.jsonl')
 for c in sorted({r['condition'] for r in cr}):
  rr=[r for r in cr if r['condition']==c];controlled.append(dict(judge=j,condition=c,n=len(rr),correct=sum((r['positive_probability']>=50)==labels[r['prompt_id']] for r in rr)))
summary=[]
for arm in ['fresh','retrieval']:
 for policy in ['fixed50','shared_tuned','direct50']:
  for subset in ['all','toxicity','hallucination']:
   for judge in ['all','qwen122','gemma','glm']:
    rr=[r for r in results if r['arm']==arm and r['policy']==policy and (judge=='all' or r['judge']==judge) and (subset=='all' or (r['dataset'] in TOX)==(subset=='toxicity'))]
    for c in sorted({r['condition'] for r in rr}):
     a=[r for r in rr if r['condition']==c];summary.append(dict(arm=arm,policy=policy,subset=subset,judge=judge,condition=c,n_cells=len({(r['judge'],r['backbone'],r['dataset']) for r in a}),**{m:100*float(np.mean([r[m] for r in a])) for m in ['f1','precision','recall','accuracy']}))
for name,data in [('cell_results',results),('summary',summary),('thresholds',thresholds),('inventory',inventory),('controlled',controlled)]: (O/(name+'.json')).write_text(json.dumps(data,indent=2))
for r in summary:
 if r['policy'] in ['fixed50','shared_tuned'] and r['judge']=='all':print(r)
print('CONTROLLED',controlled)
print('SHORT_CELLS',[r for r in inventory if r['judge']=='qwen122' and r['arm']=='fresh' and (r['n_test']<100 or r['n_val']<100)])
