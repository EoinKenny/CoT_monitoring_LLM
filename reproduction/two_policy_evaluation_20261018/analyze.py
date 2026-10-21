"""Two requested policies and validation-selected PR trade-offs. No inference."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='2'
import ast, collections, csv, gzip, hashlib, json, time
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent; P=R.parent; S=P/'api_results_20261015'; O=R/'results'; O.mkdir(exist_ok=True)
C=['baseline','saliency','nearest']; LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122']; API=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra']; MODELS=LOCAL+API
BETAS=np.logspace(-1,1,41); BETAS[20]=1.
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,allow_nan=False,separators=(',',':')))
source=P/'two_nn_20261014/code/numerics_source.py'; ns={'np':np}
tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose']],type_ignores=[]),str(source),'exec'),ns)
protocol=dict(created='2026-10-18',requested_tests=['fixed_both50','joint_validation_f1'],conditions=C,models=MODELS,
 fixed='probe probability >=0.5 AND judge positive-class score >=50',
 joint='Full original probe candidate grid from validation scores and judge integer cutoffs 0..100 plus 101 (reject-all endpoint). Maximise weighted validation F1. Highest probe threshold then highest judge cutoff on ties within1e-12. Each judge and condition tunes independently. No active-judge or minimum-count constraints.',
 frontier='Enumerate validation nondominated precision/recall grid points for each cell, resolving duplicate validation metrics by the original highest-threshold rule; evaluate those frozen thresholds on test. For aggregate plots select validation F-beta maxima at41 betas0.1..10, beta1 exactly the main F1 policy. Macro-average test metrics at each common beta. Never prune or choose by test metrics. Aggregate curves are trade-off traces, not a test-optimised frontier or exhaustive macro Pareto envelope.',
 scope='Primary combines300 local full-test estimates with200 API condition-matched-valid estimates. This is a descriptive mixture, not one common population. Provide500-cell all-judge-common-valid sensitivity with identical retained cases. Paired condition coverage within every cell.',
 weighting='Original inverse selection probabilities; equal weight to each backbone-judge-dataset cell for macro summaries. Full-test local denominators exact; API valid-subset denominators are weighted retained cases. API missingness is not assumed random.',
 uncertainty='Existing pointwise paired bootstrap intervals available in parent audit; frontier displays point estimates only. Repeated cases across judges/backbones are not independent observations.',
 mechanism_next='Separate subsequent study: Qwen122 annotates dataset-convention/ambiguity clarification and decision-relevant contrast, blind to target gold, original judge scores and observed benefit/harm. No calls in this analysis.')
write(R/'PROTOCOL.json',protocol)
audit=P/'active_judge_audit_20261018/results';receipt=read(audit/'COMPLETED.json')
for name in ['cells.json']:
 assert sha(audit/name)==receipt['outputs'][name]
for f in [S/'results/scores.npz',S/'frozen/design.json.gz']:
 assert sha(f)==receipt['inputs'][str(f)]
d=read(S/'frozen/design.json.gz');cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];z=np.load(S/'results/scores.npz')
old={(r['scope'],r['model'],r['backbone'],r['dataset'],r['condition']):r for r in read(audit/'cells.json') if r['policy']=='unrestricted'}
fixedold={(r['scope'],r['model'],r['backbone'],r['dataset'],r['condition']):r for r in read(P/'both_fixed50_20261018/cells.json')}
valid={m:(z[m]>=0).all(axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));groups=collections.defaultdict(list)
for i,r in enumerate(cases):groups[r['backbone'],r['dataset'],r['split']].append(i)
rows=[];traces=[];frontiers=[];coverage=[];replayed=0
def population(rr):return dict(n=sum(1/r['inclusion_probability'] for r in rr),positive=sum(r['true_label']/r['inclusion_probability'] for r in rr))
def pareto(g,ts,pos):
 tp=g['tp'];fp=g['fp'];prec=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0);rec=tp/pos
 # Descending recall, then precision, then probe cutoff and judge cutoff.
 cuts,ks=np.indices(tp.shape);order=np.lexsort((-cuts.ravel(),-ts[ks].ravel(),-prec.ravel(),-rec.ravel()))
 best=-1.;out=[]
 for flat in order:
  pr=prec.ravel()[flat]
  if pr>best+1e-12:
   cut,k=np.unravel_index(flat,tp.shape);out.append((int(cut),int(k)));best=pr
 return out
for scope,models,masks in [('local_full',LOCAL,{m:np.ones(len(cases),bool) for m in LOCAL}),('api_matched',API,valid),('common_valid',MODELS,{m:common for m in MODELS})]:
 for b in d['samples']:
  for ds,spec in sorted(d['thresholds'][b].items()):
   for m in models:
    vi=[i for i in groups[b,ds,'val'] if masks[m][i]];ti=[i for i in groups[b,ds,'test'] if masks[m][i]];vr=[cases[i] for i in vi];tr=[cases[i] for i in ti]
    assert vi and ti
    base=dict(scope=scope,model=m,backbone=b,dataset=ds,validation_n=len(vi),test_n=len(ti));coverage.append(base)
    pv=np.array([r['probe_probability'] for r in vr]);pt=np.array([r['probe_probability'] for r in tr]);yv=np.array([r['true_label'] for r in vr]);yt=np.array([r['true_label'] for r in tr]);wv=np.array([1/r['inclusion_probability'] for r in vr]);wt=np.array([1/r['inclusion_probability'] for r in tr])
    vp=spec['populations']['val'] if scope=='local_full' else population(vr);pop=spec['populations']['test'] if scope=='local_full' else population(tr)
    fullts=np.array(spec['thresholds']);positions=np.searchsorted(np.sort(pv),fullts,side='left');ts=fullts[np.r_[positions[:-1]!=positions[1:],True]]
    for ci,c in enumerate(C):
     val=z[m][vi,ci];test=z[m][ti,ci];assert (val>=0).all() and (test>=0).all();g=ns['grid'](vr,val,ts,vp)
     def evaluate(th,cut):
      yes=(pt>=th)&(test>=cut);vyes=(pv>=th)&(val>=cut);ref=pt>=th;vref=pv>=th
      met=ns['metrics'](wt[yes&(yt==1)].sum(),wt[yes&(yt==0)].sum(),pop['n'],pop['positive'])
      vm=ns['metrics'](wv[vyes&(yv==1)].sum(),wv[vyes&(yv==0)].sum(),vp['n'],vp['positive'])
      return dict(probe_threshold=float(th),judge_threshold=int(cut),**met,validation_f1=vm['f1'],validation_precision=vm['precision'],validation_recall=vm['recall'],validation_referred=int(vref.sum()),validation_accepted=int(vyes.sum()),validation_rejected=int((vref&~vyes).sum()),test_referred=int(ref.sum()),test_accepted=int(yes.sum()),test_rejected=int((ref&~yes).sum()),referral_rate=float(wt[ref].sum()/pop['n']))
     cut,k=ns['choose'](g['f1'],ts);joint=evaluate(ts[k],cut);fixed=evaluate(.5,50);key=(scope,m,b,ds,c)
     assert cut==old[key]['judge_threshold'] and ts[k]==old[key]['probe_threshold']
     for metric in ['f1','precision','recall','accuracy']:
      assert abs(joint[metric]-old[key][metric])<1e-10,(key,metric)
      assert abs(fixed[metric]-fixedold[key][metric])<1e-10,(key,metric)
     replayed+=1
     for policy,met in [('fixed_both50',fixed),('joint_validation_f1',joint)]:rows.append(dict(**base,condition=c,policy=policy,**met))
     cache={}
     def at(cut,k):
      key=cut,k
      if key not in cache:cache[key]=evaluate(ts[k],cut)
      return cache[key]
     for bi,beta in enumerate(BETAS):
      den=beta**2*vp['positive']+g['tp']+g['fp'];objective=np.divide((1+beta**2)*g['tp'],den,out=np.zeros_like(den),where=den>0)
      bc,bk=ns['choose'](objective,ts);met=at(bc,bk)
      if bi==20:assert bc==cut and bk==k
      traces.append(dict(**base,condition=c,beta=float(beta),beta_index=bi,**met))
     for fc,fk in pareto(g,ts,vp['positive']):frontiers.append(dict(**base,condition=c,**at(fc,fk)))
   print(scope,b,ds,flush=True)
   write(O/'STATUS.json',dict(scope=scope,backbone=b,dataset=ds,replayed_cells=replayed))
assert replayed==3000 and len(rows)==6000 and len(traces)==123000
write(O/'cells.json',rows);write(O/'coverage.json',coverage)
for name,data in [('tradeoff_points.json.gz',traces),('validation_frontier_test_points.json.gz',frontiers)]:
 with gzip.open(O/name,'wt') as f:json.dump(data,f,separators=(',',':'),allow_nan=False)
def savecsv(path,data):
 with (gzip.open(path,'wt',newline='') if str(path).endswith('.gz') else open(path,'w',newline='')) as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
savecsv(O/'cell_results.csv',rows);savecsv(O/'tradeoff_points.csv.gz',traces);savecsv(O/'validation_frontier_test_points.csv.gz',frontiers)
summary=[]
for scope in ['combined','local_full','api_matched','common_valid']:
 rr=[r for r in rows if (scope=='combined' and r['scope'] in ['local_full','api_matched']) or r['scope']==scope]
 for model in ['ALL']+MODELS:
  for policy in ['fixed_both50','joint_validation_f1']:
   for c in C:
    sel=[r for r in rr if (model=='ALL' or r['model']==model) and r['condition']==c and r['policy']==policy]
    if not sel:continue
    summary.append(dict(scope=scope,model=model,policy=policy,condition=c,cells=len(sel),validation_case_evaluations=sum(r['validation_n'] for r in sel),test_case_evaluations=sum(r['test_n'] for r in sel),**{a:float(np.mean([r[a] for r in sel])) for a in ['f1','precision','recall','accuracy','referral_rate']},zero_cutoff_cells=sum(r['judge_threshold']==0 for r in sel),validation_bypass_cells=sum(r['validation_rejected']==0 for r in sel)))
write(O/'summary.json',summary);savecsv(O/'summary.csv',summary)
write(O/'COMPLETED.json',dict(completed=time.time(),replayed_condition_cells=replayed,policies_per_cell=2,validation_tradeoff_points=len(traces),validation_pareto_points=len(frontiers),new_model_calls=0,inputs={str(p):sha(p) for p in [S/'results/scores.npz',S/'frozen/design.json.gz',audit/'cells.json',R/'PROTOCOL.json',Path(__file__)]},outputs={p.name:sha(p) for p in O.iterdir() if p.name not in ['STATUS.json','COMPLETED.json']}))
for r in summary:
 if r['scope']=='combined' and r['model']=='ALL':print(r,flush=True)
