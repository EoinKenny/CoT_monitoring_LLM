"""Validation-only joint tuning and paired, stratified weighted test evaluation."""
from common import *
import numpy as np

def metrics(tp,fp,n,pos):
 tp=float(tp);fp=float(fp)
 return dict(tp=tp,fp=fp,fn=float(pos-tp),tn=float(n-pos-fp),precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else 0.,f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0.,accuracy=(tp+n-pos-fp)/n)
def curve(rows,pred,ts,pop):
 score=np.array([r['probe_probability'] for r in rows]);y=np.array([r['true_label'] for r in rows]);w=np.array([1/r['inclusion_probability'] for r in rows]);ix=np.argsort(score,kind='stable');at=np.searchsorted(score[ix],ts,side='left');out=[]
 for gold in [1,0]:
  cumulative=np.r_[0.,np.cumsum((w*(y==gold)*pred)[ix])];out.append(cumulative[-1]-cumulative[at])
 tp,fp=out;den=pop['positive']+tp+fp
 return dict(tp=tp,fp=fp,f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0))
def grid(rows,prob,ts,pop):
 cc=[curve(rows,np.asarray(prob)>=cut,ts,pop) for cut in range(102)]
 return {name:np.stack([c[name] for c in cc]) for name in ['tp','fp','f1']}
def choose(f1,ts,fixed=None):
 if fixed is not None:
  k=int(np.flatnonzero(np.isclose(f1[fixed],f1[fixed].max(),rtol=0,atol=1e-12))[-1]);return fixed,k
 positions=np.argwhere(np.isclose(f1,f1.max(),rtol=0,atol=1e-12));cut,k=max(positions,key=lambda a:(ts[a[1]],a[0]));return int(cut),int(k)
def bootstrap(rows,rng,reps=2000):
 mult=np.zeros((reps,len(rows)))
 for st in sorted({r['stratum'] for r in rows}):
  ix=np.array([i for i,r in enumerate(rows) if r['stratum']==st]);mult[:,ix]=1 if rows[ix[0]]['inclusion_probability']==1 else rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=reps)
 return mult
def boot_metrics(mult,rows,yes,pop):
 w=np.array([1/r['inclusion_probability'] for r in rows]);y=np.array([r['true_label'] for r in rows]);tp=mult@(w*y*yes);fp=mult@(w*(1-y)*yes);den=tp+fp
 return dict(f1=np.divide(2*tp,pop['positive']+den,out=np.zeros_like(tp),where=pop['positive']+den>0),precision=np.divide(tp,den,out=np.zeros_like(tp),where=den>0),recall=tp/pop['positive'])
def main():
 reserve()
 for p,h in read(ROOT/'PARENT_VERIFIED.json')['sources'].items():assert sha(p)==h,p
 parent={(r['backbone'],r['judge'],r['dataset']):r for r in read(PARENT/'RESULTS.json')};allresults=[]
 for b in BBS:
  B=ROOT/b;rows=read(B/'selected_flags.json');ids=[r['sample_id'] for r in rows];spec=read(B/'THRESHOLDS.json')
  for j in JS:
   probs={};hard={}
   for c in CS:
    D=B/'judges'/f'{j}_{c}';done=read(D/'run/COMPLETED');assert sha(D/'run/judgments.jsonl')==done['judgments_sha256'];rr=list(jsonlines(D/'run/judgments.jsonl'));assert [r['sample_id'] for r in rr]==ids;probs[c]={r['sample_id']:r['positive_probability'] for r in rr}
    for batch in (D/'run/batches').glob('*.json'):
     if batch.name.endswith(('.attempt.json','.raw.json')):continue
     doc=read(batch);assert sha(batch.with_suffix('.raw.json'))==doc['raw_sha256'];assert doc['config_sha256']==done['config_sha256'];assert digest(json.dumps(doc['records'],sort_keys=True))==doc['records_sha256']
    hard[c]={r['sample_id']:r['prediction'] for r in jsonlines(PARENT/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl')}
   for ds,s in spec.items():
    ts=np.array(s['thresholds']);assert np.all(np.diff(ts)>=0)
    vr=[r for r in rows if r['dataset']==ds and r['split']=='val'];tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];pop=s['populations']['test'];vp=s['populations']['val'];z=np.load(B/f'{ds}_predictions.npz');fulltest=z['probabilities'][z['splits']=='test'];vscore=np.array([r['probe_probability'] for r in vr]);tscore=np.array([r['probe_probability'] for r in tr])
    rng=np.random.default_rng(int(digest(f'20261009:{b}:{ds}:{j}')[:16],16));mult=bootstrap(tr,rng);results={};boots={};arrays={};frozen={};calibration={}
    # Select every threshold using validation before inspecting test performance.
    for c in CS:
     pv=np.array([probs[c][r['sample_id']] for r in vr]);vg=grid(vr,pv,ts,vp);arrays[c+'_validation_f1']=vg['f1']
     for policy in ['fixed50','joint']:
      cut,k=choose(vg['f1'],ts,50 if policy=='fixed50' else None);frozen[c+'_'+policy]=dict(probe_threshold=float(ts[k]),probability_threshold=cut,k=k,validation_f1=float(vg['f1'][cut,k]))
    D=B/'evaluation';D.mkdir(exist_ok=True);write(D/f'{j}_{ds}_VALIDATION_SELECTED.json',frozen)
    for c in CS:
     pt=np.array([probs[c][r['sample_id']] for r in tr]);tg=grid(tr,pt,ts,pop)
     w=np.array([1/r['inclusion_probability'] for r in tr]);y=np.array([r['true_label'] for r in tr]);bins=[]
     for low in range(0,100,10):
      ix=(pt>=low)&(pt<(low+10) if low<90 else pt<=100)
      if ix.any():bins.append(dict(low=low,high=low+10,sample_n=int(ix.sum()),weighted_n=float(w[ix].sum()),mean_probability=float(np.average(pt[ix]/100,weights=w[ix])),observed_positive_rate=float(np.average(y[ix],weights=w[ix]))))
     calibration[c]=dict(weighted_brier=float(np.average((pt/100-y)**2,weights=w)),bins=bins,description='Descriptive design-weighted test calibration; not used to select cutoffs')
     for name,val in tg.items():arrays[c+'_test_'+name]=val
     for policy in ['fixed50','joint']:
      key=c+'_'+policy;f=frozen[key];cut=f['probability_threshold'];k=f['k'];yes=(tscore>=ts[k])&(pt>=cut);r=metrics(tg['tp'][cut,k],tg['fp'][cut,k],pop['n'],pop['positive']);r.update({k0:v for k0,v in f.items() if k0!='k'});r.update(routing_rate=float(np.mean(fulltest>=ts[k])),validation_sample=len(vr),test_sample=len(tr),test_estimated=any(x['inclusion_probability']<1 for x in tr));results[key]=r;boots[key]=boot_metrics(mult,tr,yes,pop)
     key=c+'_parent_hard';old=parent[b,j,ds]['conditions'][c];threshold=old['threshold'];yes=(tscore>=threshold)*np.array([hard[c][r['sample_id']] for r in tr]);tc=curve(tr,np.array([hard[c][r['sample_id']] for r in tr]),np.array([threshold]),pop);r=metrics(tc['tp'][0],tc['fp'][0],pop['n'],pop['positive']);assert abs(r['f1']-old['f1'])<1e-10;r.update(probe_threshold=threshold,validation_f1=old['validation_f1'],routing_rate=float(np.mean(fulltest>=threshold)),validation_sample=len(vr),test_sample=len(tr),test_estimated=any(x['inclusion_probability']<1 for x in tr));results[key]=r;boots[key]=boot_metrics(mult,tr,yes,pop)
    for key in results:
     results[key]['ci95']={m:np.quantile(v,[.025,.975]).tolist() for m,v in boots[key].items()}
    comparisons={}
    for c in CS:
     for policy in ['fixed50','joint']:
      key=c+'_'+policy
      refs=[c+'_parent_hard']+([c+'_fixed50'] if policy=='joint' else [])+(['baseline_'+policy] if c!='baseline' else [])
      for ref in refs:
       comparisons[key+'_minus_'+ref]={m:dict(delta=results[key][m]-results[ref][m],ci95=np.quantile(boots[key][m]-boots[ref][m],[.025,.975]).tolist()) for m in ['f1','precision','recall']}
    np.savez_compressed(D/f'{j}_{ds}_curves.npz',probe_thresholds=ts,probability_thresholds=np.arange(102),**arrays)
    allresults.append(dict(backbone=b,judge=j,dataset=ds,variants=results,comparisons=comparisons,score_diagnostics=calibration,probe_only=parent[b,j,ds]['probe_only'],bootstrap='2000 matched stratified test resamples, census fixed; conditional on validation-selected thresholds; pointwise, not simultaneous frontier intervals',interpretation='Exploratory study; probability scores not assumed calibrated'))
    write(ROOT/'EVALUATION_STATUS.json',dict(pairs=len(allresults),total=100,time=time.time()))
 assert len(allresults)==100;write(ROOT/'RESULTS.json',allresults);write(ROOT/'EVALUATION_COMPLETED.json',dict(pairs=100,variants_per_pair=9,judgments=300000,sha256=sha(ROOT/'RESULTS.json')))
if __name__=='__main__':main()
