"""Weighted validation tuning; independent test estimates and paired intervals."""
import os,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009')
CS=['baseline','saliency','nearest'];BBS=['llama','qwen','gemma','llama70','qwen122']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(tp,fp,n,pos):
 return dict(tp=float(tp),fp=float(fp),fn=float(pos-tp),tn=float(n-pos-fp),precision=float(tp/(tp+fp)) if tp+fp else 0.,recall=float(tp/pos) if pos else 0.,f1=float(2*tp/(pos+tp+fp)) if pos+tp+fp else 0.,accuracy=float((tp+n-pos-fp)/n))
def curves(rows,pred,ts,pop):
 score=np.array([r['probe_probability'] for r in rows]);y=np.array([r['true_label'] for r in rows]);w=np.array([1/r['inclusion_probability'] for r in rows]);out=[]
 for gold in [1,0]:
  ix=np.argsort(score,kind='stable');weights=w*(y==gold)*pred;cumulative=np.r_[0,np.cumsum(weights[ix])];out.append(cumulative[-1]-cumulative[np.searchsorted(score[ix],ts,side='left')])
 tp,fp=out;return dict(tp=tp,fp=fp,f1=np.divide(2*tp,pop['positive']+tp+fp),recall=tp/pop['positive'],precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0))
def run():
 allresults=[]
 for b in BBS:
  R=ROOT/b;rows=read(R/'selected_flags.json');assert sha(R/'selected_flags.json')==read(R/'SAMPLE_FROZEN.json')['selection_sha256'];spec=read(R/'THRESHOLDS.json')
  for judge in ['qwen','gemma']:
   pred={}
   for c in CS:
    D=R/'judges'/f'{judge}_{c}'/'run';assert sha(D/'judgments.jsonl')==read(D/'COMPLETED')['judgments_sha256'];rr=[json.loads(l) for l in (D/'judgments.jsonl').open()];p={r['sample_id']:r['prediction'] for r in rr};assert len(p)==len(rows)==len(rr) and set(p)=={r['sample_id'] for r in rows};pred[c]=p
   for ds,s in spec.items():
    ts=np.array(s['thresholds']);vr=[r for r in rows if r['dataset']==ds and r['split']=='val'];tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];saved={};best={};selected={}
    for c in CS:
     vc=curves(vr,np.array([pred[c][r['sample_id']] for r in vr]),ts,s['populations']['val']);tc=curves(tr,np.array([pred[c][r['sample_id']] for r in tr]),ts,s['populations']['test']);k=np.flatnonzero(np.isclose(vc['f1'],vc['f1'].max(),rtol=0,atol=1e-12))[-1];best[c]=k
     for split,data in [('val',vc),('test',tc)]:
      for name,v in data.items():saved[f'{c}_{split}_{name}']=v
     selected[c]=metrics(tc['tp'][k],tc['fp'][k],s['populations']['test']['n'],s['populations']['test']['positive'])
     selected[c].update(threshold=float(ts[k]),validation_f1=float(vc['f1'][k]),validation_sample=len(vr),test_sample=len(tr),test_estimated=any(r['inclusion_probability']<1 for r in tr))
    y=np.array([r['true_label'] for r in tr]);w=np.array([1/r['inclusion_probability'] for r in tr]);score=np.array([r['probe_probability'] for r in tr]);rng=np.random.default_rng(int(hashlib.sha256(f'20261009:{b}:{ds}:{judge}'.encode()).hexdigest()[:16],16));mult=np.zeros((2000,len(tr)))
    for st in sorted({r['stratum'] for r in tr}):
     ix=np.array([i for i,r in enumerate(tr) if r['stratum']==st]);mult[:,ix]=1 if tr[ix[0]]['inclusion_probability']==1 else rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=2000)
    boots={}
    for c in CS:
     yes=np.array([pred[c][r['sample_id']] for r in tr])*(score>=ts[best[c]]);tp=mult@(w*y*yes);fp=mult@(w*(1-y)*yes);boots[c]=2*tp/(s['populations']['test']['positive']+tp+fp)
     selected[c]['test_f1_ci95']=np.quantile(boots[c],[.025,.975]).tolist()
    for c in CS[1:]:selected[c]['delta_f1_vs_baseline']=selected[c]['f1']-selected['baseline']['f1'];selected[c]['paired_delta_ci95']=np.quantile(boots[c]-boots['baseline'],[.025,.975]).tolist()
    # Probe-only threshold chosen using complete validation, evaluated on complete test.
    z=np.load(R/f'{ds}_predictions.npz');p=z['probabilities'];yy=z['labels'];sp=z['splits'];fullv=[dict(probe_probability=float(x),true_label=int(y0),inclusion_probability=1) for x,y0 in zip(p[sp=='val'],yy[sp=='val'])];pc=curves(fullv,np.ones(len(fullv)),ts,s['populations']['val']);pk=np.flatnonzero(np.isclose(pc['f1'],pc['f1'].max(),rtol=0,atol=1e-12))[-1];test=sp=='test';yes=p[test]>=ts[pk];gold=yy[test];probe=metrics(sum(yes&(gold==1)),sum(yes&(gold==0)),len(gold),int(gold.sum()));probe['threshold']=float(ts[pk])
    D=R/'evaluation';D.mkdir(exist_ok=True);np.savez_compressed(D/f'{judge}_{ds}_curves.npz',thresholds=ts,**saved)
    allresults.append(dict(backbone=b,judge=judge,dataset=ds,conditions=selected,probe_only=probe,bootstrap='2000 paired stratified test resamples; census fixed; conditional on validation-selected thresholds; not simultaneous frontier confidence'))
 (ROOT/'RESULTS.json').write_text(json.dumps(allresults,indent=2));(ROOT/'EVALUATION_COMPLETED.json').write_text(json.dumps(dict(pairs=len(allresults),condition_results=len(allresults)*3,sha256=sha(ROOT/'RESULTS.json'))))
if __name__=='__main__':run()
