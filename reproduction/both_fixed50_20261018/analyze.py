"""Both probe>=0.5 and judge>=50. Frozen cases, no threshold selection."""
import gzip,json,collections,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;S=R.parent/'api_results_20261015'
d=json.load(gzip.open(S/'frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rs in d['samples'].items() for r in rs];z=np.load(S/'results/scores.npz');C=['baseline','saliency','nearest'];local=['qwen','gemma','mistral','olmo','glm','qwen122'];api=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra'];valid={m:(z[m]>=0).all(axis=1) for m in api};common=np.logical_and.reduce(list(valid.values()));groups=collections.defaultdict(list)
for i,r in enumerate(cases):
 if r['split']=='test':groups[r['backbone'],r['dataset']].append(i)
def met(tp,fp,n,pos):return dict(f1=2*tp/(pos+tp+fp) if pos+tp+fp else 0,precision=tp/(tp+fp) if tp+fp else 0,recall=tp/pos if pos else 0,accuracy=(tp+n-pos-fp)/n)
rows=[]
for scope,models,masks in [('local_full',local,{m:np.ones(len(cases),bool) for m in local}),('api_matched',api,valid),('common_valid',local+api,{m:common for m in local+api})]:
 for m in models:
  for (b,ds),ii in groups.items():
   ix=np.array([i for i in ii if masks[m][i]]);rr=[cases[i] for i in ix];w=np.array([1/r['inclusion_probability'] for r in rr]);y=np.array([r['true_label'] for r in rr]);p=np.array([r['probe_probability'] for r in rr]);pop=d['thresholds'][b][ds]['populations']['test'] if scope=='local_full' else dict(n=float(w.sum()),positive=float(w@y));gate=p>=.5
   for ci,c in enumerate(C):
    yes=gate&(z[m][ix,ci]>=50);tp=float(w[yes&(y==1)].sum());fp=float(w[yes&(y==0)].sum());rows.append(dict(scope=scope,model=m,backbone=b,dataset=ds,condition=c,test_n=len(ix),referred_n=int(gate.sum()),vetoed_n=int((gate&~yes).sum()),probe_threshold=.5,judge_threshold=50,**met(tp,fp,pop['n'],pop['positive'])))
summary=[]
for scope in ['combined','local_full','api_matched','common_valid']:
 rr=[r for r in rows if (scope=='combined' and r['scope'] in ['local_full','api_matched']) or r['scope']==scope]
 for m in ['ALL']+sorted({r['model'] for r in rr}):
  for c in C:
   v=[r for r in rr if r['condition']==c and (m=='ALL' or r['model']==m)];summary.append(dict(scope=scope,model=m,condition=c,cells=len(v),test_n=sum(r['test_n'] for r in v),**{k:sum(r[k] for r in v)/len(v) for k in ['f1','precision','recall','accuracy']},no_veto_cells=sum(r['vetoed_n']==0 for r in v)))
json.dump(rows,open(R/'cells.json','w'),indent=2);json.dump(summary,open(R/'summary.json','w'),indent=2)
json.dump(dict(method='Fixed probe>=0.5 and judge>=50, no tuning or exclusion based on old thresholds. Means weight backbone-judge-dataset cells equally. Local full-test estimates use frozen design weights and exact population totals; API matched-valid results conditional on valid answers in all3conditions. Combined is descriptive mixture of these scopes, not a single common population. Common-valid scope evaluated separately. Existing responses only. Repeated targets are not independent observations.',inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'frozen/design.json.gz',S/'results/scores.npz',Path(__file__)]}),open(R/'VERIFIED.json','w'),indent=2)
for r in summary:
 if r['model']=='ALL' or r['scope']=='combined':print(r)
