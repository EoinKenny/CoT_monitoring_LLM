"""Read-only analysis of frozen study; writes only to a separate output directory."""
import json, hashlib, csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import precision_recall_curve

ROOT=Path('/home/ubuntu/experiments/grid-completion-20261008')
OUT=Path(sys.argv[1]); OUT.mkdir(parents=True,exist_ok=True)
BBS=['llama','qwen','gemma','llama70','qwen122']
JS=['qwen','gemma','glm','olmo','qwen122']
CS=['baseline','saliency','nearest']
BN=['Llama 1B','Qwen 4B','Gemma 31B','Llama 70B','Qwen 122B']
JN=['Qwen 27B','Gemma 31B','GLM Flash','OLMo 32B','Qwen 122B']
COL=['#546478','#D68B20','#008D83']; LABEL=['No explanation','Saliency','Nearest neighbour']
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source={}; grid=np.r_[0.,np.linspace(.05,.95,19)]
expected={(r['backbone'],r['dataset'],r['judge'],r['condition']):r for r in read(ROOT/'RESULTS.json')['judge_results']}
allrows=[]; cache={}; max_error=0.
for b in BBS:
 R=ROOT/b; selected=read(R/'selected_flags.json'); frozen=read(R/'SAMPLE_FROZEN.json')
 assert sha(R/'selected_flags.json')==frozen['selection_sha256']
 source[str(R/'selected_flags.json')]=sha(R/'selected_flags.json')
 results=read(R/'results.json'); decisions={}
 for j in JS:
  for c in CS:
   p=R/'judges'/f'{j}_{c}'/'run/judgments.jsonl'
   assert sha(p)==read(p.parent/'COMPLETED')['judgments_sha256']
   source[str(p)]=sha(p); rr=[json.loads(l) for l in p.open()]
   by={r['sample_id']:r['prediction'] for r in rr}
   assert len(by)==len(selected) and set(by)=={r['sample_id'] for r in selected}
   assert set(by.values()) <= {0,1}
   decisions[j,c]=by
 for ds,result in results.items():
  path=R/f'{ds}_predictions.npz'; source[str(path)]=sha(path)
  if ds in frozen.get('inputs',{}):assert source[str(path)]==frozen['inputs'][ds]['predictions_sha256']
  z=np.load(path); y=z['labels']; p=z['probabilities']; split=z['splits']; t0=result['threshold']
  yt=y[split=='test']; pt=p[split=='test']; N=len(yt); pos=int(yt.sum())
  vp=p[(split=='val')&(p>=t0)]; assert len(vp)
  ts=np.r_[t0,np.quantile(vp,grid[1:])]; assert np.all(ts>=t0)
  rows=[r for r in selected if r['dataset']==ds]
  sy=np.array([r['true_label'] for r in rows]); sp=np.array([r['probe_probability'] for r in rows]); w=np.array([1/r['inclusion_probability'] for r in rows])
  for r in rows:
   assert y[r['index']]==r['true_label'] and split[r['index']]=='test'
   assert abs(p[r['index']]-r['probe_probability'])<1e-12
  masks=sp[None,:]>=ts[:,None]; routed=np.array([(pt>=t).mean() for t in ts])
  pp,pr,_=precision_recall_curve(yt,pt)
  # Paired bootstrap within original gold strata. Census strata are fixed.
  rng=np.random.default_rng(int(hashlib.sha256((b+ds).encode()).hexdigest()[:8],16))
  mult=np.zeros((500,len(rows)))
  for gold in [0,1]:
   ix=np.flatnonzero(sy==gold)
   if not len(ix):continue
   assert np.allclose(w[ix],w[ix[0]])
   mult[:,ix]=1 if w[ix[0]]==1 else rng.multinomial(len(ix),np.full(len(ix),1/len(ix)),size=500)
  cache[b,ds]={'probe':(pr,pp),'curves':{},'t0':t0}
  for j in JS:
   for c in CS:
    pred=np.array([decisions[j,c][r['sample_id']] for r in rows]); yes=masks*pred
    tp=yes@(w*sy); fp=yes@(w*(1-sy)); recall=tp/pos
    precision=np.divide(tp,tp+fp,out=np.full(len(ts),np.nan),where=tp+fp>0)
    f1=2*tp/(pos+tp+fp)
    ref=expected[b,ds,j,c]['pipeline_metrics']
    err=max(abs(tp[0]-ref['tp']),abs(fp[0]-ref['fp']),abs(f1[0]-ref['f1']))
    max_error=max(max_error,err); assert err<1e-7
    bt=(mult*(w*sy))@yes.T; bf=(mult*(w*(1-sy)))@yes.T
    brec=bt/pos; bprec=np.divide(bt,bt+bf,out=np.full_like(bt,np.nan),where=bt+bf>0)
    bf1=2*bt/(pos+bt+bf)
    ci={k:np.nanquantile(v,[.025,.975],axis=0) for k,v in [('precision',bprec),('recall',brec),('f1',bf1)]}
    support=masks.sum(1); accepted=yes.sum(1); ok=(support>=30)&(accepted>=10)&np.isfinite(precision)
    cache[b,ds]['curves'][j,c]=(recall,precision,f1,routed,ok,ci)
    for k,t in enumerate(ts):
     row=dict(backbone=b,dataset=ds,judge=j,condition=c,validation_flag_quantile=float(grid[k]),threshold=float(t),original_threshold=t0,precision=float(precision[k]),recall=float(recall[k]),f1=float(f1[k]),routed_fraction=float(routed[k]),sample_remaining=int(support[k]),sample_accepted=int(accepted[k]),plot_supported=bool(ok[k]))
     for m in ci:row[m+'_lo']=float(ci[m][0,k]);row[m+'_hi']=float(ci[m][1,k])
     allrows.append(row)
 print('analysed',b,flush=True)

with (OUT/'threshold_metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,5,figsize=(19,4.6))
for j,name,ax in zip(JS,JN,axes):
 for c,col,label in zip(CS,COL,LABEL):
  vals=[v['curves'][j,c] for v in cache.values()]
  r=np.mean([v[0] for v in vals],axis=0);p=np.mean([v[1] for v in vals],axis=0)
  ok=np.all([v[4] for v in vals],axis=0)
  ax.plot(100*r[ok],100*p[ok],color=col,lw=2,label=label)
  ax.scatter(100*r[0],100*p[0],color=col,s=55,zorder=4)
 ax.set(title=name,xlabel='Mean recall (%)',ylabel='Mean precision (%)')
 ax.grid(alpha=.2)
axes[0].legend(fontsize=8,loc='best')
fig.suptitle('Existing-data threshold sweep: precision–recall trade-off',fontsize=17)
fig.text(.5,.01,'Equal means across 5 backbones × 10 datasets. Dots: original thresholds. Curves: stricter thresholds only.\nShared validation-score quantile policy; descriptive macro trajectories, not established Pareto frontiers.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.12,1,.91]);fig.savefig(OUT/'overview.png',dpi=180);fig.savefig(OUT/'overview.pdf');plt.close(fig)

fig,axes=plt.subplots(1,5,figsize=(19,4.6))
for j,name,ax in zip(JS,JN,axes):
 for c,col,label in zip(CS,COL,LABEL):
  vals=[v['curves'][j,c] for v in cache.values()];f=np.mean([v[2] for v in vals],axis=0);route=np.mean([v[3] for v in vals],axis=0);ok=np.all([v[4] for v in vals],axis=0)
  ax.plot(100*route[ok],100*f[ok],color=col,label=label,lw=2);ax.scatter(100*route[0],100*f[0],color=col,s=55)
 ax.set(title=name,xlabel='Mean test cases routed (%)',ylabel='Mean F1 (%)');ax.grid(alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('F1 versus judge routing rate — existing-data sweep',fontsize=17)
fig.text(.5,.02,'Dots: original thresholds. Lines use stricter thresholds only; equal averages across 50 backbone–dataset pairs.\nRouting rate does not include differences in prompt length. No thresholds were selected on test.',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.12,1,.91]);fig.savefig(OUT/'f1_cost.png',dpi=180);plt.close(fig)

with PdfPages(OUT/'all_pairs.pdf') as pdf:
 for b,bname in zip(BBS,BN):
  datasets=[d for bb,d in cache if bb==b]
  for j,jname in zip(JS,JN):
   fig,axes=plt.subplots(2,5,figsize=(19,8))
   for ds,ax in zip(datasets,axes.flat):
    v=cache[b,ds];r,p=v['probe'];ax.plot(r,p,color='#999999',lw=1,alpha=.8,label='Probe only (full range)')
    for f in [.4,.6,.8,.9]:
     x=np.linspace(f/(2-f)+.0001,1,150);z=f*x/(2*x-f);ax.plot(x,z,':',color='#dddddd',lw=.6)
    for c,col,label in zip(CS,COL,LABEL):
     r,p,f,route,ok,ci=v['curves'][j,c]
     ax.plot(r[ok],p[ok],color=col,lw=1.8,label=label)
     ax.errorbar(r[0],p[0],xerr=[[max(0,r[0]-ci['recall'][0,0])],[max(0,ci['recall'][1,0]-r[0])]],yerr=[[max(0,p[0]-ci['precision'][0,0])],[max(0,ci['precision'][1,0]-p[0])]],color=col,fmt='o',ms=4,capsize=2,lw=.8)
     ax.fill_between(r[ok],ci['precision'][0,ok],ci['precision'][1,ok],color=col,alpha=.08)
    ax.set(title=ds,xlabel='Recall',ylabel='Precision',xlim=(0,1.02),ylim=(0,1.02));ax.grid(alpha=.12)
   axes.flat[0].legend(fontsize=7,loc='lower left')
   fig.suptitle(f'{bname} backbone · {jname} judge',fontsize=17)
   fig.text(.5,.015,'Original-threshold dots with paired-stratified 95% marginal bootstrap intervals (500 draws; census strata fixed).\nShading: pointwise precision intervals, not simultaneous frontier confidence bands. Dashed grey: equal F1. Unsupported low-sample points omitted.',ha='center',fontsize=9)
   fig.tight_layout(rect=[0,.08,1,.94]);pdf.savefig(fig);plt.close(fig)
manifest=dict(source_sha256=source,script_sha256=sha(Path(__file__)),seed='SHA256(backbone+dataset) first 8 hex',bootstrap_replicates=500,max_original_metric_error=max_error,rows=len(allrows),judges=JS,backbones=BBS,threshold_quantiles=grid.tolist(),minimum_remaining_sample=30,minimum_accepted_sample=10)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.txt').write_text('Existing-data exploratory threshold sweep. No inference or probe training performed.\nThresholds are the original validation-selected threshold, then quantiles .05 through .95 of validation scores already above that threshold. This covers stricter routing only.\nTP and FP are Horvitz-Thompson estimates using ORIGINAL stratum inclusion probabilities, never renormalized after threshold filtering. Recall divides by exact full-test positives. F1=2TP/(positives+TP+FP).\nAll conditions share cases and bootstrap draws. Bootstrap is conditional on fixed probes and finite held-out populations; census strata fixed, no finite-population correction. Intervals are pointwise, not simultaneous, and do not represent uncertainty over datasets or training.\nOriginal threshold metrics reproduce the saved study. Figures omit points with fewer than 30 remaining sampled cases or 10 predicted positives. Macro figures retain a point only if all 50 cells pass; no changing dataset composition.\nFull probe-only precision-recall curves are in the 25-page detailed PDF. Macro trajectories average precision/recall per cell at a shared validation-quantile policy: they are NOT optimized frontiers. Equal-F1 contours apply only to individual-cell figures.\nLower thresholds need new judgments; validation judgments are needed for independent cascade threshold selection. No test-selected optimal F1 is claimed. Llama judge excluded as requested; Llama backbones retained. No aggregate bootstrap CI is constructed by assuming independence across shared datasets/backbones.\n')
print(json.dumps({'rows':len(allrows),'max_original_error':max_error,'output':str(OUT)}),flush=True)
