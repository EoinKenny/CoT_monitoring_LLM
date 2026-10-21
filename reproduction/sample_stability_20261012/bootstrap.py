import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import json,hashlib,time,sys,collections,concurrent.futures
from pathlib import Path
import numpy as np
P=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');R=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009');O=Path(os.environ.get('OUT','/home/ubuntu/experiments/sample-stability-20261012'));O.mkdir(exist_ok=True)
BBS=['llama','qwen','gemma','llama70','qwen122'];JS=['qwen','gemma'];CS=['baseline','saliency','nearest'];MODES=['hard','joint','fixed50'];NAMES=[c+'_'+m for c in CS for m in MODES];REPS=int(os.environ.get('REPS','10000'));SEED=20261012

def read(p): return json.loads(p.read_text())
def jl(p): return [json.loads(l) for l in p.open()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def f1(tp,fp,pos): return np.divide(2*tp,pos+tp+fp,out=np.zeros_like(np.asarray(tp),dtype=float),where=pos+tp+fp>0)
def quotas(rows,n):
 groups=[np.array([i for i,r in enumerate(rows) if r['stratum']==s]) for s in sorted({r['stratum'] for r in rows})];q=np.zeros(len(groups),dtype=int)
 while q.sum()<n:
  for k,g in enumerate(groups):
   if q[k]<len(g): q[k]+=1
   if q.sum()==n:break
 weights=np.concatenate([np.full(k,len(g)/rows[g[0]]['inclusion_probability']/k) for g,k in zip(groups,q)])
 return groups,q,weights

def draw(rng,groups,q): return np.concatenate([rng.choice(g,k,replace=True) for g,k in zip(groups,q)])
def grid(scores,ts):
 order=np.argsort(scores,kind='stable');starts=np.searchsorted(scores[order],ts);keep=np.r_[np.flatnonzero(starts[:-1]!=starts[1:]),len(ts)-1];return order,starts[keep],ts[keep]
def tune(s,y,w,p,h,ts,pos):
 order,starts,ct=grid(s,ts);n=len(s);hs=[]
 # Exact integer cutoffs: cut=0 plus one above each observed value, and 101.
 cuts=np.unique(np.r_[0,p+1,101]);cuts=cuts[cuts<=101]
 for g in [1,0]:
  a=np.zeros((n+1,102));a[np.arange(n),p[order]]=w[order]*(y[order]==g);a=np.cumsum(np.cumsum(a[::-1,::-1],axis=0),axis=1)[::-1,::-1];hs.append(a)
 allf=f1(hs[0][starts],hs[1][starts],pos)
 # F1 flat between observed probability values; choose highest tied cutoff.
 choices=[]
 hc=[]
 for g in [1,0]:
  v=np.r_[0,np.cumsum((w*(y==g)*h)[order])];hc.append(v[-1]-v[starts])
 for a in [f1(*hc,pos)[:,None],allf,allf[:,50:51]]:
  # Rows ascending probe threshold, columns ascending probability cutoff.
  ix=np.flatnonzero(a.ravel()>=a.max()-1e-12)[-1];k,cut=np.unravel_index(ix,a.shape);choices.append((ct[k],int(cut)))
 choices[0]=(choices[0][0],-1);choices[2]=(choices[2][0],50)
 return choices

def metrics(s,y,w,p,h,t,pos):
 yes=(s>=t[0]) & ((h==1) if t[1]<0 else p>=t[1]);return float(f1(np.sum(w*y*yes),np.sum(w*(1-y)*yes),pos))
def summary(a): return dict(mean=float(np.mean(a)),sd=float(np.std(a,ddof=1)),p025=float(np.quantile(a,.025)),median=float(np.median(a)),p975=float(np.quantile(a,.975)))
def job(item):
 b,ds,rows,sp,pred=item;vr=[r for r in rows if r['dataset']==ds and r['split']=='val'];tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];assert len(vr)==len(tr)==500
 assert not {r['group'] for r in vr}&{r['group'] for r in tr}
 vs=np.array([r['probe_probability'] for r in vr]);xs=np.array([r['probe_probability'] for r in tr]);vy=np.array([r['true_label'] for r in vr]);ty=np.array([r['true_label'] for r in tr]);vw=np.array([1/r['inclusion_probability'] for r in vr]);tw=np.array([1/r['inclusion_probability'] for r in tr]);vp=sp['populations']['val']['positive'];tp=sp['populations']['test']['positive'];_,_,ts=grid(vs,np.array(sp['thresholds']));vg,vq,vbw=quotas(vr,100);tg,tq,tbw=quotas(tr,100)
 values={j:{c:tuple(np.array([pred[j][c][r['sample_id']][k] for r in rr]) for rr,k in [(vr,0),(tr,0),(vr,1),(tr,1)]) for c in CS} for j in JS}
 ref=np.zeros((2,9));refcuts=np.zeros((2,9,2));full=np.zeros((REPS,2,9));sample=full.copy();thresholds=np.zeros((REPS,2,9,2))
 for ji,j in enumerate(JS):
  for ci,c in enumerate(CS):
   pv,pt,hv,ht=values[j][c]
   for mi,t in enumerate(tune(vs,vy,vw,pv,hv,ts,vp)):
    ref[ji,ci*3+mi]=metrics(xs,ty,tw,pt,ht,t,tp);refcuts[ji,ci*3+mi]=t
    original=next(r for r in read(R/'RESULTS.json') if r['backbone']==b and r['dataset']==ds and r['judge']==j)['variants'][c+'_'+['parent_hard','joint','fixed50'][mi]]
    assert abs(ref[ji,ci*3+mi]-original['f1'])<1e-10,(b,ds,j,c,mi)
    assert t[0]==original['probe_threshold']
    if mi!=0: assert t[1]==original['probability_threshold']
 rng=np.random.default_rng(int(hashlib.sha256(f'{SEED}:{b}:{ds}'.encode()).hexdigest()[:16],16))
 for rep in range(REPS):
  vi=draw(rng,vg,vq);ti=draw(rng,tg,tq)
  for ji,j in enumerate(JS):
   for ci,c in enumerate(CS):
    pv,pt,hv,ht=values[j][c]
    for mi,t in enumerate(tune(vs[vi],vy[vi],vbw,pv[vi],hv[vi],ts,vp)):
     k=ci*3+mi;thresholds[rep,ji,k]=t;full[rep,ji,k]=metrics(xs,ty,tw,pt,ht,t,tp);sample[rep,ji,k]=metrics(xs[ti],ty[ti],tbw,pt[ti],ht[ti],t,tp)
  if rep%1000==0: save(O/f'{b}_{ds}_status.json',dict(replicates=rep,total=REPS,time=time.time()))
 np.savez_compressed(O/f'{b}_{ds}.npz',reference=ref,reference_thresholds=refcuts,full_test=full,sampled_test=sample,thresholds=thresholds)
 save(O/f'{b}_{ds}_status.json',dict(replicates=REPS,total=REPS,time=time.time(),complete=True))
 return b,ds

def main():
 inputs=[];hashes={}
 for b in BBS:
  rows=read(R/b/'selected_flags.json');assert rows==read(P/b/'selected_flags.json');sp=read(R/b/'THRESHOLDS.json');pred={}
  for f in [R/b/'selected_flags.json',R/b/'THRESHOLDS.json']:hashes[str(f)]=sha(f)
  for j in JS:
   pred[j]={}
   for c in CS:
    files=[R/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl',P/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl'];rr,hh=[jl(f) for f in files];assert [r['sample_id'] for r in rr]==[r['sample_id'] for r in hh]==[r['sample_id'] for r in rows]
    for f in files:
     digest=sha(f);assert digest==read(f.parent/'COMPLETED')['judgments_sha256'];hashes[str(f)]=digest
    pred[j][c]={r['sample_id']:(r['positive_probability'],h['prediction']) for r,h in zip(rr,hh)}
  for ds,s in sp.items():inputs.append((b,ds,rows,s,pred))
 save(O/'PROTOCOL.json',dict(seed=SEED,replicates=REPS,n_validation=100,n_test=100,parent_n_each=500,judges=JS,variants=NAMES,method='Paired stratified m-out-of-n bootstrap with replacement within original score-band by gold-label strata. Quotas identical to reduced study. Weights stratum population/quota. Independent val/test draws; shared draws across conditions and judges. Retune every replicate; original frozen grid and highest probe then probability tie break. Evaluate on both independent 100 test draw and unchanged 500 test reference.',limits='Conditional empirical sensitivity, not new population confidence intervals. Original 500 pool held fixed. Backbone samples resampled independently, so pooled spread does not model cross-backbone source overlap.',inputs=hashes,code_sha256=sha(Path(__file__))))
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
  for k,result in enumerate(ex.map(job,inputs)):
   save(O/'STATUS.json',dict(done=k+1,total=len(inputs),last=result,time=time.time()))
 aggregate(inputs)

def aggregate(inputs):
 full=[];sample=[];ref=[];details=[]
 for b,ds,*_ in inputs:
  z=np.load(O/f'{b}_{ds}.npz');full.append(z['full_test']);sample.append(z['sampled_test']);ref.append(z['reference'])
  for ji,j in enumerate(JS):
   for mode in ['hard','joint','fixed50']:
    k=MODES.index(mode);a=6+k;bb=k;d=z['sampled_test'][:,ji,a]-z['sampled_test'][:,ji,bb];df=z['full_test'][:,ji,a]-z['full_test'][:,ji,bb]
    details.append(dict(backbone=b,dataset=ds,judge=j,mode=mode,reference_delta=float(z['reference'][ji,a]-z['reference'][ji,bb]),sampled_delta=summary(d),validation_only_delta=summary(df),positive_fraction=float(np.mean(d>1e-12))))
 full=np.array(full);sample=np.array(sample);ref=np.array(ref);out={}
 for ji,j in enumerate(JS):
  out[j]={}
  for mode in MODES:
   k=MODES.index(mode);a=6+k;bb=k;d=(sample[:,:,ji,a]-sample[:,:,ji,bb]).mean(axis=0);df=(full[:,:,ji,a]-full[:,:,ji,bb]).mean(axis=0)
   out[j][mode]=dict(reference_baseline=float(ref[:,ji,bb].mean()),reference_nn=float(ref[:,ji,a].mean()),nn_minus_baseline=summary(d),validation_only_nn_minus_baseline=summary(df),nn_positive_fraction=float(np.mean(d>1e-12)),baseline_f1=summary(sample[:,:,ji,bb].mean(axis=0)),nn_f1=summary(sample[:,:,ji,a].mean(axis=0)),validation_only_baseline_f1=summary(full[:,:,ji,bb].mean(axis=0)),validation_only_nn_f1=summary(full[:,:,ji,a].mean(axis=0)))
 save(O/'SUMMARY.json',out);save(O/'PER_PAIR.json',details);save(O/'COMPLETED.json',dict(time=time.time(),replicates=REPS,pairs=100,outputs={p.name:sha(p) for p in O.glob('*.npz')}))
 print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
