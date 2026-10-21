import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from pathlib import Path
import json,hashlib,time,collections,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');R=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009');OUT=Path('/home/ubuntu/experiments/probability-comprehensive-analysis-20261011');OUT.mkdir(exist_ok=True)
BBS=['llama','qwen','gemma','llama70','qwen122'];JS=['qwen','gemma'];CS=['baseline','saliency','nearest']
CORE=['probe','baseline_parent_hard','baseline_joint','saliency_parent_hard','saliency_joint','nearest_parent_hard','nearest_joint'];BUDGETS=[.1,.25,.5,1.]
REPS=200

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,v):
 p=OUT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.replace(p)
def jl(p):return [json.loads(l) for l in Path(p).open()]
def cm(tp,fp,n,pos):
 tp=float(tp);fp=float(fp);return dict(tp=tp,fp=fp,fn=pos-tp,tn=n-pos-fp,f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/pos if pos else 0.,accuracy=(tp+n-pos-fp)/n)
def f1(tp,fp,pos):return np.divide(2*tp,pos+tp+fp,out=np.zeros_like(tp,dtype=float),where=pos+tp+fp>0)
def resamples(rows,rng,n):
 M=np.zeros((n,len(rows)))
 for st in sorted({r['stratum'] for r in rows}):
  ix=np.array([i for i,r in enumerate(rows) if r['stratum']==st]);M[:,ix]=1 if rows[ix[0]]['inclusion_probability']==1 else rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=n)
 return M

def prepare_grid(vscore,ts):
 order=np.argsort(vscore,kind='stable');starts=np.searchsorted(vscore[order],ts,'left');keep=np.r_[np.flatnonzero(starts[:-1]!=starts[1:]),len(ts)-1];return order,starts[keep],keep

def surface(order,starts,prob,y,w,pos):
 # Histogram score by sorted probe rank and elicited probability, then suffix sums.
 n=len(order);hist=[]
 for gold in [1,0]:
  h=np.zeros((n+1,102));h[np.arange(n),prob[order]]=w[order]*(y[order]==gold);h=np.cumsum(np.cumsum(h[::-1,::-1],axis=0),axis=1)[::-1,::-1];hist.append(h[starts].T)
 return f1(hist[0],hist[1],pos)
def hardcurve(order,starts,pred,y,w,pos):
 out=[]
 for g in [1,0]:
  c=np.r_[0.,np.cumsum((w*(y==g)*pred)[order])];out.append(c[-1]-c[starts])
 return f1(*out,pos)
def select(grid,ts,valid=None,fixed=None):
 g=grid if fixed is None else grid[fixed:fixed+1]
 if valid is not None:g=np.where(valid[None,:],g,-np.inf)
 maxv=g.max();pairs=np.argwhere(np.isclose(g,maxv,atol=1e-12,rtol=0));cut,k=max(pairs,key=lambda a:(ts[a[1]],a[0]));return int(cut if fixed is None else fixed),int(k),float(maxv)
def winner(candidates):
 best=max(v['validation_f1'] for v in candidates.values());tied=[k for k,v in candidates.items() if abs(v['validation_f1']-best)<=1e-12];return min(tied,key=lambda k:(candidates[k]['validation_routing'],CORE.index(k)))
def metrics_yes(rows,yes,pop):
 y=np.array([r['true_label'] for r in rows]);w=np.array([1/r['inclusion_probability'] for r in rows]);return cm(np.sum(w*y*yes),np.sum(w*(1-y)*yes),pop['n'],pop['positive'])
def boot_f1(rows,yes,M,pos):
 y=np.array([r['true_label'] for r in rows]);w=np.array([1/r['inclusion_probability'] for r in rows]);return f1(M@(w*y*yes),M@(w*(1-y)*yes),pos)
def verify():
 audit=[]
 for root in [P,R]:
  assert sha(root/'RESULTS.json')==read(root/'COMPLETED.json')['results_sha256'];n=0
  for b in BBS:
   ids=[r['sample_id'] for r in read(root/b/'selected_flags.json')];assert ids==[r['sample_id'] for r in read(P/b/'selected_flags.json')]
   for j in JS:
    for c in CS:
     D=root/b/'judges'/f'{j}_{c}';done=read(D/'run/COMPLETED');config=read(D/'run/config.json');assert sha(D/'run/judgments.jsonl')==done['judgments_sha256'];assert sha(D/'run/config.json')==done['config_sha256'];assert sha(D/'prompts.jsonl')==config['prompt_sha256'];rr=jl(D/'run/judgments.jsonl');assert [r['sample_id'] for r in rr]==ids
     committed=[]
     for batch in sorted((D/'run/batches').glob('*.json')):
      if batch.name.endswith(('.raw.json','.attempt.json')):continue
      d=read(batch);assert sha(batch.with_suffix('.raw.json'))==d['raw_sha256'];assert hashlib.sha256(json.dumps(d['records'],sort_keys=True).encode()).hexdigest()==d['records_sha256'];assert d['config_sha256']==done['config_sha256'];committed.extend(d['records'])
     assert committed==rr and len(rr)==10000;n+=len(rr)
  assert n==300000;audit.append(dict(root=str(root),judgments=n,complete=True))
 write('VERIFIED.json',dict(time=time.time(),runs=audit))

def run():
 verify();results={(x['backbone'],x['judge'],x['dataset']):x for x in read(R/'RESULTS.json')};pairs=[];budget_rows=[];endpoints=[];effects=[];support=[];cases=[];calibration=[];stability=[];curves=[];neighbour_groups=[];counts=[]
 for b in BBS:
  rows=read(R/b/'selected_flags.json');spec=read(R/b/'THRESHOLDS.json');nns={r['sample_id']:r for r in jl(P/b/'judge_inputs/explanations.jsonl')};valgroups={(r['dataset'],r['group']) for r in rows if r['split']=='val'};testgroups={(r['dataset'],r['group']) for r in rows if r['split']=='test'};assert not valgroups&testgroups
  for j in JS:
   hard={c:{r['sample_id']:r['prediction'] for r in jl(P/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl')} for c in CS};probs={c:{r['sample_id']:r['positive_probability'] for r in jl(R/b/'judges'/f'{j}_{c}'/'run/judgments.jsonl')} for c in CS}
   timing={}
   for c in CS:
    for mode,root in [('hard',P),('probability',R)]:
     D=root/b/'judges'/f'{j}_{c}';rr=jl(D/'run/judgments.jsonl');seconds=sum({r['batch_number']:r['batch_seconds'] for r in rr}.values());timing[c+'_'+mode]=dict(input_tokens=sum(r['input_tokens'] for r in rr),measured_inference_seconds=seconds,n=len(rr));counts.append(dict(backbone=b,judge=j,condition=c,mode=mode,**timing[c+'_'+mode]))
   for ds,sp in spec.items():
    start=time.time();key=dict(backbone=b,judge=j,dataset=ds);base=results[b,j,ds];vr=[r for r in rows if r['dataset']==ds and r['split']=='val'];tr=[r for r in rows if r['dataset']==ds and r['split']=='test'];vs=np.array([r['probe_probability'] for r in vr]);xs=np.array([r['probe_probability'] for r in tr]);vy=np.array([r['true_label'] for r in vr]);ty=np.array([r['true_label'] for r in tr]);vw=np.array([1/r['inclusion_probability'] for r in vr]);tw=np.array([1/r['inclusion_probability'] for r in tr]);vp=sp['populations']['val'];tp=sp['populations']['test'];ts=np.array(sp['thresholds']);order,starts,keep=prepare_grid(vs,ts);ct=ts[keep];z=np.load(R/b/f'{ds}_predictions.npz');fv=z['probabilities'][z['splits']=='val'];fy=z['labels'][z['splits']=='val'];ft=z['probabilities'][z['splits']=='test'];fty=z['labels'][z['splits']=='test'];vrouting=(len(fv)-np.searchsorted(np.sort(fv),ct,'left'))/len(fv);test_routing=lambda t:float(np.mean(ft>=t));val_routing=lambda t:float(np.mean(fv>=t));cand={};yesmap={};thresholds={};values={};grids={};hgrids={}
    for c in CS:
     pv=np.array([probs[c][r['sample_id']] for r in vr]);pt=np.array([probs[c][r['sample_id']] for r in tr]);hv=np.array([hard[c][r['sample_id']] for r in vr]);ht=np.array([hard[c][r['sample_id']] for r in tr]);values[c]=(pv,pt,hv,ht);grids[c]=surface(order,starts,pv,vy,vw,vp['positive']);hgrids[c]=hardcurve(order,starts,hv,vy,vw,vp['positive'])
     for suffix in ['parent_hard','joint','fixed50']:
      name=c+'_'+suffix;r=dict(base['variants'][name]);r['validation_routing']=val_routing(r['probe_threshold']);cand[name]=r;yesmap[name]=(xs>=r['probe_threshold'])*(ht if suffix=='parent_hard' else pt>=r['probability_threshold']);thresholds[name]=(r['probe_threshold'],r.get('probability_threshold',None))
      calc=metrics_yes(tr,yesmap[name],tp);assert abs(calc['f1']-r['f1'])<1e-10
      # Verify compressed tuning exactly reproduces stored choices.
      cut,k,v=select(hgrids[c][None,:] if suffix=='parent_hard' else grids[c],ct,fixed=50 if suffix=='fixed50' else None)
      assert abs(v-r['validation_f1'])<1e-10 and ct[k]==r['probe_threshold'],(key,name,'retuning mismatch',ct[k],r['probe_threshold'])
      if suffix=='joint':assert cut==r['probability_threshold']
      sel=xs>=r['probe_threshold'];ww=tw[sel];support.append(dict(**key,variant=name,routed_test_sample=int(sel.sum()),routed_effective_n=float(ww.sum()**2/(ww@ww)) if len(ww) else 0.,test_sample=len(tr),validation_sample=len(vr),min_inclusion_probability=float(min(x['inclusion_probability'] for x in tr))))
     # LLM-only endpoint; choose score cutoff on validation, with probe threshold zero.
     for mode in ['hard','fixed50','joint']:
      if mode=='hard':cut=None;yes=ht
      elif mode=='fixed50':cut=50;yes=pt>=cut
      else:
       endpoint=np.array([metrics_yes(vr,pv>=q,vp)['f1'] for q in range(102)]);cut=int(np.flatnonzero(np.isclose(endpoint,endpoint.max(),atol=1e-12,rtol=0))[-1]);yes=pt>=cut
      endpoints.append(dict(**key,condition=c,mode=mode,probability_threshold=cut,**metrics_yes(tr,yes,tp)))
     # Score distribution diagnostics; provided calibration bins/Brier in original evaluation.
     calibration.append(dict(**key,condition=c,weighted_brier=base['score_diagnostics'][c]['weighted_brier'],zero_mass=float(np.average(pt==0,weights=tw)),hundred_mass=float(np.average(pt==100,weights=tw)),multiple_of_10_mass=float(np.average(pt%10==0,weights=tw)),positive_mean=float(np.average(pt[ty==1],weights=tw[ty==1])),negative_mean=float(np.average(pt[ty==0],weights=tw[ty==0])),bins=base['score_diagnostics'][c]['bins']))
     for budget in BUDGETS:
      valid=vrouting<=budget+1e-12
      for mode in ['hard','joint','fixed50']:
       gg=hgrids[c][None,:] if mode=='hard' else grids[c];cut,k,v=select(gg,ct,valid,fixed=50 if mode=='fixed50' else None);yes=(xs>=ct[k])*(ht if mode=='hard' else pt>=cut);budget_rows.append(dict(**key,condition=c,mode=mode,budget=budget,validation_f1=v,probe_threshold=float(ct[k]),probability_threshold=None if mode=='hard' else cut,validation_routing=float(vrouting[k]),test_routing=test_routing(ct[k]),**metrics_yes(tr,yes,tp)))
    pr=dict(base['probe_only']);pr['probe_threshold']=pr.pop('threshold');pp=pr['probe_threshold'];pr.update(validation_f1=cm(np.sum((fv>=pp)&(fy==1)),np.sum((fv>=pp)&(fy==0)),len(fy),int(fy.sum()))['f1'],validation_routing=0.,routing_rate=0.);cand['probe']=pr;yesmap['probe']=(xs>=pp)
    chosen=winner({k:cand[k] for k in CORE});pairs.append(dict(**key,variants=cand,selected=chosen,selected_test=cand[chosen],test_sample=len(tr),validation_sample=len(vr),populations=sp['populations']))
    contrasts=[('baseline_joint','nearest_parent_hard'),('nearest_joint','baseline_joint'),('nearest_parent_hard','baseline_parent_hard')]+[(c+'_joint',c+'_parent_hard') for c in CS]+[(c+'_joint','probe') for c in CS]
    rng=np.random.default_rng(int(hashlib.sha256(f'analysis:20261011:{b}:{j}:{ds}'.encode()).hexdigest()[:16],16));TM=resamples(tr,rng,2000);boots={k:boot_f1(tr,yesmap[k],TM,tp['positive']) for k in CORE if k!='probe'};boots['probe']=np.repeat(pr['f1'],2000)
    for a,bb in contrasts:
     aa=yesmap[a].astype(bool);zz=yesmap[bb].astype(bool);effects.append(dict(**key,comparison=a+'_minus_'+bb,delta_f1=cand[a]['f1']-cand[bb]['f1'],ci95=np.quantile(boots[a]-boots[bb],[.025,.975]).tolist(),delta_precision=cand[a]['precision']-cand[bb]['precision'],delta_recall=cand[a]['recall']-cand[bb]['recall'],weighted_recovered_positives=float(np.sum(tw*(ty==1)*aa*~zz)),weighted_lost_positives=float(np.sum(tw*(ty==1)*~aa*zz)),weighted_removed_false_positives=float(np.sum(tw*(ty==0)*~aa*zz)),weighted_added_false_positives=float(np.sum(tw*(ty==0)*aa*~zz))))
    # Inspect NN advantage with both tuned and hard-label alternatives; retain all transition IDs.
    for mode in ['joint','parent_hard']:
     a=yesmap['nearest_'+mode].astype(bool);bb=yesmap['baseline_'+mode].astype(bool);ss=yesmap['saliency_'+mode].astype(bool)
     for i,r in enumerate(tr):
      ref=nns[r['sample_id']]['nearest'];ag=int(ref['label'])==int(ref['prediction']);labelmatch=int(ref['label'])==int(r['true_label']);improved=bool(a[i]==ty[i] and bb[i]!=ty[i]);worse=bool(a[i]!=ty[i] and bb[i]==ty[i]);neighbour_groups.append(dict(**key,mode=mode,agreement=ag,label_matches_target=labelmatch,score_band=r['score_band'],weight=float(tw[i]),improved=improved,worse=worse))
      if (improved and ss[i]!=ty[i]) or worse:cases.append(dict(**key,mode=mode,sample_id=r['sample_id'],gold=int(ty[i]),probe_probability=float(xs[i]),nn_decision=int(a[i]),baseline_decision=int(bb[i]),saliency_decision=int(ss[i]),nn_probability=probs['nearest'][r['sample_id']],baseline_probability=probs['baseline'][r['sample_id']],saliency_probability=probs['saliency'][r['sample_id']],weight=float(tw[i]),kind='nn_only_correct' if improved else 'nn_hurt',neighbour_label=int(ref['label']),neighbour_prediction=int(ref['prediction'])))
    # Selection uncertainty: resample the original judge-validation sample and paired test sample.
    VM=resamples(vr,rng,REPS);SM=resamples(tr,rng,REPS);freq=collections.Counter();rep_f1=[];cuts=collections.defaultdict(list);deltas=[]
    for rep in range(REPS):
     rc={'probe':pr};rtest={'probe':pr['f1']}
     for c in CS:
      pv,pt,hv,ht=values[c];gg=surface(order,starts,pv,vy,vw*VM[rep],vp['positive']);hh=hardcurve(order,starts,hv,vy,vw*VM[rep],vp['positive'])
      for mode,grid0 in [('joint',gg),('parent_hard',hh[None,:])]:
       cut,k,v=select(grid0,ct);name=c+'_'+mode;rc[name]=dict(validation_f1=v,validation_routing=float(vrouting[k]));yes=(xs>=ct[k])*(ht if mode=='parent_hard' else pt>=cut);ww=tw*SM[rep];rtest[name]=cm(np.sum(ww*ty*yes),np.sum(ww*(1-ty)*yes),tp['n'],tp['positive'])['f1'];cuts[name].append([float(ct[k]),None if mode=='parent_hard' else cut])
     win=winner(rc);freq[win]+=1;rep_f1.append(rtest[win]);deltas.append([rtest['baseline_joint']-rtest['nearest_parent_hard'],rtest['nearest_joint']-rtest['baseline_joint']])
    stability.append(dict(**key,replicates=REPS,configuration_frequencies=dict(freq),original_choice=chosen,original_choice_frequency=freq[chosen]/REPS,selected_f1_sensitivity_ci95=np.quantile(rep_f1,[.025,.975]).tolist(),primary_delta_sensitivity_ci95=np.quantile(deltas,[.025,.975],axis=0).tolist(),threshold_quantiles={k:dict(probe=np.quantile([x[0] for x in v],[.1,.5,.9]).tolist(),probability=np.quantile([x[1] for x in v],[.1,.5,.9]).tolist() if v[0][1] is not None else None) for k,v in cuts.items()},note='200 original-stratum validation and test resamples; complete-validation probe-only baseline fixed; exploratory sensitivity, not a new external validation set'))
    write('STATUS.json',dict(state='running',pairs=len(pairs),total=100,last=key,last_seconds=time.time()-start))
    if len(pairs)%10==0:write('PARTIAL_PAIRS.json',pairs)
 for name,obj in [('pairs',pairs),('budgets',budget_rows),('endpoints',endpoints),('effects',effects),('support',support),('cases',cases),('calibration',calibration),('stability',stability),('neighbour_groups',neighbour_groups),('runtime',counts)]:write(name+'.json',obj)
 write('ANALYSIS_COMPLETED.json',dict(time=time.time(),pairs=len(pairs),replicates_selection_sensitivity=REPS));write('STATUS.json',dict(state='completed',pairs=len(pairs),total=100))
if __name__=='__main__':run()
