"""Transfer validation-selected threshold pairs across explanation conditions."""
import json,gzip,hashlib,collections,csv,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent;S=P/'api_results_20261015';OLD=P/'two_policy_evaluation_20261018/results';O=R/'results';O.mkdir(exist_ok=True)
C=['baseline','saliency','nearest'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];API=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra'];MODELS=LOCAL+API
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,separators=(',',':'),allow_nan=False))
def savecsv(p,rr):
 with gzip.open(p,'wt',newline='') if str(p).endswith('.gz') else open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def met(tp,fp,n,pos):return dict(tp=float(tp),fp=float(fp),tn=float(n-pos-fp),fn=float(pos-tp),precision=float(tp/(tp+fp)) if tp+fp else 0.,recall=float(tp/pos) if pos else 0.,f1=float(2*tp/(pos+tp+fp)) if pos+tp+fp else 0.,accuracy=float((tp+n-pos-fp)/n))
protocol=dict(date='2026-10-18',anchors=C,conditions=C,
 selection='For EACH judge-backbone-dataset and anchor condition, jointly maximise weighted validation F1 over original full probe candidate grid and judge0..101. Use the exact verified grid-search selection from two_policy_evaluation_20261018. Highest probe threshold then judge threshold for ties within1e-12. No new constraints, no test tuning.',
 transfer='Freeze BOTH cutoffs from the selected anchor, then apply the identical pair to all three conditions on identical retained test cases. Repeat control-anchor, NN-anchor and saliency-anchor. Distinct judges may use distinct pairs.',
 average='Compute NN-control, saliency-control and NN-saliency differences within each anchor and cell; average those differences equally across three anchors, then equally across500 model-backbone-dataset cells. This is a symmetric sensitivity summary, not an extra independent experiment or a deployable averaged threshold.',
 frontier='Use the same anchor-only validation F-beta grid selections (41 beta values0.1..10); transfer both thresholds unchanged to all conditions. Beta1 exactly matches main F1 policy. Never select/prune by test. Curves show transferred operating points, not independently optimised explanation frontiers.',
 scopes='Primary500-cell descriptive mean combines300 local full-test estimates and200 API condition-matched valid-case estimates. Also report500-cell common-valid-all-judges sensitivity. Original design weights retained. API missingness remains explicit.',
 fixed='Retain original fixed50/50 results unchanged as reference.',
 limits='Point estimates; no claim of statistical significance. Three anchors use the same observations and are not independent replications. Previously inspected test set; explanatory follow-up. Qwen mechanism annotation remains postponed; no inference calls.')
write(R/'PROTOCOL.json',protocol)
receipt=read(OLD/'COMPLETED.json')
for f in [S/'results/scores.npz',S/'frozen/design.json.gz']:assert sha(f)==receipt['inputs'][str(f)]
for name in ['cells.json','tradeoff_points.json.gz']:assert sha(OLD/name)==receipt['outputs'][name]
oldrows=read(OLD/'cells.json');oldtraces=read(OLD/'tradeoff_points.json.gz')
d=read(S/'frozen/design.json.gz');cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];z=np.load(S/'results/scores.npz')
valid={m:(z[m]>=0).all(axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));groups=collections.defaultdict(list)
for i,r in enumerate(cases):groups[r['backbone'],r['dataset'],r['split']].append(i)
cellkey=lambda r:(r['scope'],r['model'],r['backbone'],r['dataset'])
old={(cellkey(r),r['condition']):r for r in oldrows if r['policy']=='joint_validation_f1'}
source_traces=collections.defaultdict(list)
for r in oldtraces:source_traces[cellkey(r),r['condition']].append(r)
rows=[];traces=[];thresholds=[];diffs=[];diagonal=0
for key in sorted({cellkey(r) for r in oldrows}):
 scope,m,b,ds=key;mask=common if scope=='common_valid' else valid[m] if scope=='api_matched' else np.ones(len(cases),bool)
 ti=[i for i in groups[b,ds,'test'] if mask[i]];tr=[cases[i] for i in ti];pt=np.array([r['probe_probability'] for r in tr]);y=np.array([r['true_label'] for r in tr]);w=np.array([1/r['inclusion_probability'] for r in tr])
 pop=d['thresholds'][b][ds]['populations']['test'] if scope=='local_full' else {'n':float(w.sum()),'positive':float(w@y)}
 base=dict(scope=scope,model=m,backbone=b,dataset=ds,test_n=len(ti),validation_n=old[key,'baseline']['validation_n'])
 cache={}
 def evaluate(th,cut,ci):
  cachekey=float(th),int(cut),ci
  if cachekey not in cache:
   ref=pt>=th;yes=ref&(z[m][ti,ci]>=cut);v=met(w[yes&(y==1)].sum(),w[yes&(y==0)].sum(),pop['n'],pop['positive'])
   cache[cachekey]=dict(probe_threshold=float(th),judge_threshold=int(cut),test_referred=int(ref.sum()),test_accepted=int(yes.sum()),test_rejected=int((ref&~yes).sum()),referral_rate=float(w[ref].sum()/pop['n']),**v)
  return cache[cachekey]
 for anchor in C:
  sel=old[key,anchor];th=sel['probe_threshold'];cut=sel['judge_threshold'];thresholds.append(dict(**base,anchor=anchor,probe_threshold=th,judge_threshold=cut,anchor_validation_f1=sel['validation_f1']))
  current=[]
  for ci,c in enumerate(C):
   v=evaluate(th,cut,ci)
   if c==anchor:
    for a in ['f1','precision','recall','accuracy']:assert abs(v[a]-sel[a])<1e-10,(key,c,a)
    diagonal+=1
   row=dict(**base,anchor=anchor,condition=c,anchor_validation_f1=sel['validation_f1'],**v);rows.append(row);current.append(row)
   for point in source_traces[key,anchor]:
    tm=evaluate(point['probe_threshold'],point['judge_threshold'],ci)
    traces.append(dict(**base,anchor=anchor,condition=c,beta=point['beta'],beta_index=point['beta_index'],**tm))
    if point['beta_index']==20:
     assert tm['probe_threshold']==th and tm['judge_threshold']==cut
     assert abs(tm['f1']-v['f1'])<1e-10
  assert len({(r['probe_threshold'],r['judge_threshold'],r['test_referred']) for r in current})==1
  if cut==0:
   assert len({(r['tp'],r['fp'],r['tn'],r['fn']) for r in current})==1
  for c,ref in [('nearest','baseline'),('saliency','baseline'),('nearest','saliency')]:
   a=current[C.index(c)];bb=current[C.index(ref)];diffs.append(dict(**base,anchor=anchor,comparison=c+'-'+ref,**{k:a[k]-bb[k] for k in ['f1','precision','recall','accuracy']}))
 if len(thresholds)%150==0:print('cells',len(thresholds)//3,flush=True)
assert diagonal==3000 and len(rows)==9000 and len(traces)==369000
averages=[]
gg=collections.defaultdict(list)
for r in diffs:gg[cellkey(r),r['comparison']].append(r)
for (key,comp),rr in gg.items():
 assert len(rr)==3
 averages.append(dict(scope=key[0],model=key[1],backbone=key[2],dataset=key[3],comparison=comp,**{k:float(np.mean([r[k] for r in rr])) for k in ['f1','precision','recall','accuracy']}))
summary=[];delta_summary=[]
def scoped(data,scope):return [r for r in data if (scope=='combined' and r['scope']!='common_valid') or r['scope']==scope]
for scope in ['combined','local_full','api_matched','common_valid']:
 rr=scoped(rows,scope);dd=scoped(diffs,scope);aa=scoped(averages,scope)
 for model in ['ALL']+MODELS:
  filt=lambda data:[r for r in data if model=='ALL' or r['model']==model]
  if not filt(rr):continue
  for anchor in C+['average_of_three']:
   for c in C:
    selected=[r for r in filt(rr) if r['condition']==c and (anchor=='average_of_three' or r['anchor']==anchor)]
    summary.append(dict(scope=scope,model=model,anchor=anchor,condition=c,cells=len(selected)//(3 if anchor=='average_of_three' else 1),**{a:float(np.mean([r[a] for r in selected])) for a in ['precision','recall','f1','accuracy','referral_rate']},zero_cutoff_fraction=float(np.mean([r['judge_threshold']==0 for r in selected]))))
   for comp in ['nearest-baseline','saliency-baseline','nearest-saliency']:
    sel=[r for r in filt(aa if anchor=='average_of_three' else dd) if r['comparison']==comp and (anchor=='average_of_three' or r['anchor']==anchor)]
    delta_summary.append(dict(scope=scope,model=model,anchor=anchor,comparison=comp,cells=len(sel),**{a:float(np.mean([r[a] for r in sel])) for a in ['precision','recall','f1','accuracy']},positive_f1_cells=sum(r['f1']>1e-10 for r in sel),negative_f1_cells=sum(r['f1']< -1e-10 for r in sel),tied_f1_cells=sum(abs(r['f1'])<=1e-10 for r in sel)))
for name,data in [('cells',rows),('thresholds',thresholds),('cell_deltas',diffs),('cell_average_deltas',averages),('summary',summary),('delta_summary',delta_summary)]:write(O/(name+'.json'),data);savecsv(O/(name+'.csv'),data)
with gzip.open(O/'tradeoff_points.json.gz','wt') as f:json.dump(traces,f,separators=(',',':'),allow_nan=False)
savecsv(O/'tradeoff_points.csv.gz',traces)
write(O/'COMPLETED.json',dict(time=time.time(),diagonal_replays=diagonal,identical_threshold_and_referral_checks=True,zero_cutoff_equal_predictions_verified=True,rows=len(rows),new_model_calls=0,inputs={str(p):sha(p) for p in [OLD/'cells.json',OLD/'tradeoff_points.json.gz',S/'results/scores.npz',S/'frozen/design.json.gz',R/'PROTOCOL.json',Path(__file__)]},outputs={p.name:sha(p) for p in O.iterdir() if p.name!='COMPLETED.json'}))
for r in summary:
 if r['scope']=='combined' and r['model']=='ALL':print(r,flush=True)
for r in delta_summary:
 if r['scope']=='combined' and r['model']=='ALL':print(r,flush=True)
