"""Re-evaluate frozen scores using validation-only non-bypass policies."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
import json,gzip,hashlib,collections,ast,time,html
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent;S=P/'api_results_20261015';OUT=R/'results';OUT.mkdir(exist_ok=True)
def read(p):
 with gzip.open(p,'rt') if str(p).endswith('.gz') else open(p) as f:return json.load(f)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
CS=['baseline','saliency','nearest'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];API=['claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra'];ALL=LOCAL+API
POLICIES=['unrestricted','positive_cutoff','active','active20','fixed50']
protocol=dict(date='2026-10-18',conditions=CS,models=ALL,policies={'unrestricted':'Original 0..101 search retained for exact replay','positive_cutoff':'1..100; no functional veto requirement','active':'1..100 and at least one accepted and one rejected raw validation referral','active20':'Sensitivity: at least20 accepted and20 rejected raw validation referrals','fixed50':'50; original reference'},selection='Maximum weighted validation F1, ties within1e-12 choose highest probe threshold then highest judge cutoff. Test never selects. Infeasible policies retained, not replaced.',probes='Original full-validation probe-only retained separately, plus sample-validation-tuned probe-only reference using same candidates/weights as cascade.',data='Exact existing IDs, gold, weights, score artifacts; no inference or retraining. API scopes conditional on valid responses. Original scores.npz replay-verified against raw exports.',bootstrap='2000 paired stratified test resamples within each cell, census strata fixed; conditional on frozen validation choices. No aggregate independence claim across backbones/judges.',new_constraints='Post-hoc protocol amendment; original results retained. Functional activity enforced on validation only; test activity reported.')
write(R/'PROTOCOL.json',protocol)
source=P/'two_nn_20261014/code/numerics_source.py';ns={'np':np};tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose','bootstrap','boot_metrics']],type_ignores=[]),str(source),'exec'),ns)
design=read(S/'frozen/design.json.gz');cases=[dict(backbone=b,**r) for b,rr in design['samples'].items() for r in rr];idx={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)};z=np.load(S/'results/scores.npz');scores={m:z[m] for m in ALL}
manifest=read(S/'frozen/EXPORT_MANIFEST.json');assert sha(S/'frozen/design.json.gz')==manifest['files']['design.json.gz'];assert sha(S/'frozen/local_scores.jsonl.gz')==manifest['files']['local_scores.jsonl.gz']
seen=set()
with gzip.open(S/'frozen/local_scores.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);key=(r['model'],idx[r['backbone'],r['sample_id']],CS.index(r['condition']));assert key not in seen;seen.add(key);assert scores[key[0]][key[1],key[2]]==r['positive_probability']
assert len(seen)==900000
# API score replay from strict original response parsing and prompt mapping.
import csv,re
pid={}
with gzip.open(S/'frozen/prompt_index.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);pid[r['prompt_id']]=(idx[r['backbone'],r['sample_id']],CS.index(r['condition']))
napi=0
with open(S/'responses.csv') as f:
 for r in csv.DictReader(f):
  i,c=pid[r['prompt_id']];val=int(r['response'].strip()) if r['status']=='ok' and re.fullmatch(r'(?:0|[1-9][0-9]?|100)',r['response'].strip()) else -1
  assert scores[r['model']][i,c]==val;napi+=1
assert napi==600000
parents={(r['judge'],r['backbone'],r['dataset']):r for r in read(P/'probability_500_20261012/results/pairs.json')}
matched={(r['scope'],r['model'],r['backbone'],r['dataset'],r['condition'],r['policy']):r for r in read(S/'results/matched_results.json')}
valid={m:(scores[m]>=0).all(axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));groups=collections.defaultdict(list)
for i,r in enumerate(cases):groups[r['backbone'],r['dataset'],r['split']].append(i)
def population(rr):return dict(n=sum(1/r['inclusion_probability'] for r in rr),positive=sum(r['true_label']/r['inclusion_probability'] for r in rr))
def counts(ps,probs,ts):
 order=np.argsort(ps,kind='stable');at=np.searchsorted(ps[order],ts,side='left');a=np.zeros((102,len(ts)),dtype=int)
 for cut in range(102):
  cm=np.r_[0,np.cumsum(probs[order]>=cut)];a[cut]=cm[-1]-cm[at]
 return a,len(ps)-at
results=[];comparisons=[];audits=[];tests=0
for scope,models,masks in [('local_full',LOCAL,{m:np.ones(len(cases),bool) for m in LOCAL}),('common_valid',ALL,{m:common for m in ALL}),('api_matched',API,valid)]:
 for b in design['samples']:
  for ds,spec in sorted(design['thresholds'][b].items()):
   for m in models:
    vi=[i for i in groups[b,ds,'val'] if masks[m][i]];ti=[i for i in groups[b,ds,'test'] if masks[m][i]];vr=[cases[i] for i in vi];tr=[cases[i] for i in ti]
    pv=np.array([r['probe_probability'] for r in vr]);pt=np.array([r['probe_probability'] for r in tr]);yt=np.array([r['true_label'] for r in tr]);wt=np.array([1/r['inclusion_probability'] for r in tr]);vp=spec['populations']['val'] if scope=='local_full' else population(vr);tp=spec['populations']['test'] if scope=='local_full' else population(tr)
    # Retain the highest original threshold for each identical validation referral set.
    fullts=np.asarray(spec['thresholds']);positions=np.searchsorted(np.sort(pv),fullts,side='left');keep=np.r_[positions[:-1]!=positions[1:],True];ts=fullts[keep]
    assert len(ts)<=len(vr)+1
    seed=int(hashlib.sha256(f'active20261018:{scope}:{b}:{ds}'.encode()).hexdigest()[:16],16);mult=ns['bootstrap'](tr,np.random.default_rng(seed),2000)
    base=dict(scope=scope,model=m,backbone=b,dataset=ds,validation_n=len(vr),test_n=len(tr));boots={};cells={}
    for ci,c in enumerate(CS):
     val=scores[m][vi,ci];test=scores[m][ti,ci];vg=ns['grid'](vr,val,ts,vp);acc,ref=counts(pv,val,ts);rej=ref[None,:]-acc
     original_cut,original_k=ns['choose'](vg['f1'],ts)
     old=parents[m,b,ds]['variants'][c+'_joint'] if scope=='local_full' else matched['common_valid_all_judges' if scope=='common_valid' else 'api_condition_matched',m,b,ds,c,'joint']
     assert original_cut==old['probability_threshold'] and ts[original_k]==old['probe_threshold'];assert abs(vg['f1'][original_cut,original_k]-old['validation_f1'])<1e-10;tests+=1
     masks_policy={'unrestricted':np.ones_like(acc,bool),'positive_cutoff':np.broadcast_to(((np.arange(102)>0)&(np.arange(102)<=100))[:,None],acc.shape),'active':(acc>=1)&(rej>=1)&(np.arange(102)[:,None]>0)&(np.arange(102)[:,None]<=100),'active20':(acc>=20)&(rej>=20)&(np.arange(102)[:,None]>0)&(np.arange(102)[:,None]<=100),'fixed50':np.broadcast_to((np.arange(102)==50)[:,None],acc.shape)}
     bestpos=np.max(np.where(masks_policy['positive_cutoff'],vg['f1'],-np.inf));bestactive=np.max(np.where(masks_policy['active'],vg['f1'],-np.inf))
     audits.append(dict(**base,condition=c,original_cut=original_cut,original_accept=int(acc[original_cut,original_k]),original_reject=int(rej[original_cut,original_k]),original_validation_f1=float(vg['f1'].max()),nonzero_ties_optimum=bool(abs(bestpos-vg['f1'].max())<=1e-12),active_ties_optimum=bool(abs(bestactive-vg['f1'].max())<=1e-12),active_feasible=bool(masks_policy['active'].any())))
     for pol,mask in masks_policy.items():
      key=c,pol
      if not mask.any():results.append(dict(**base,condition=c,policy=pol,feasible=False));continue
      objective=np.where(mask,vg['f1'],-np.inf);cut,k=ns['choose'](objective,ts);yes=(pt>=ts[k])&(test>=cut);referred=pt>=ts[k]
      met=ns['metrics'](float(wt[yes&(yt==1)].sum()),float(wt[yes&(yt==0)].sum()),tp['n'],tp['positive'])
      if pol=='unrestricted':assert abs(met['f1']-old['f1'])<1e-10
      bt=ns['boot_metrics'](mult,tr,yes,tp);boots[key]=bt;cells[key]=met
      results.append(dict(**base,condition=c,policy=pol,feasible=True,probe_threshold=float(ts[k]),judge_threshold=cut,validation_f1=float(vg['f1'][cut,k]),validation_accepted=int(acc[cut,k]),validation_rejected=int(rej[cut,k]),test_referred=int(referred.sum()),test_accepted=int(yes.sum()),test_rejected=int((referred&~yes).sum()),weighted_referral_rate=float(wt[referred].sum()/tp['n']),**met,ci95={a:np.quantile(bt[a],[.025,.975]).tolist() for a in ['f1','precision','recall']}))
    # Probe-only using the same sampled validation objective, plus original full-validation frozen probe.
    pg=ns['curve'](vr,np.ones(len(vr)),ts,vp);k=int(np.flatnonzero(np.isclose(pg['f1'],pg['f1'].max(),rtol=0,atol=1e-12))[-1])
    for name,threshold in [('sample_tuned_probe',float(ts[k])),('original_probe',parents[LOCAL[0],b,ds]['variants']['probe']['probe_threshold'])]:
     yes=pt>=threshold;met=ns['metrics'](float(wt[yes&(yt==1)].sum()),float(wt[yes&(yt==0)].sum()),tp['n'],tp['positive'])
     if name=='original_probe' and scope=='local_full':met={k:parents[m,b,ds]['variants']['probe'][k] for k in ['tp','fp','tn','fn','f1','precision','recall','accuracy']}
     results.append(dict(**base,condition='probe_only',policy=name,feasible=True,probe_threshold=threshold,**met))
     if name=='sample_tuned_probe':boots['probe_only',name]=ns['boot_metrics'](mult,tr,yes,tp);cells['probe_only',name]=met
    for pol in POLICIES:
     for c in ['saliency','nearest']:
      for refkey in [('baseline',pol),('probe_only','sample_tuned_probe')]:
       key=c,pol
       if key not in boots or refkey not in boots:continue
       comparisons.append(dict(**base,condition=c,policy=pol,reference=':'.join(refkey),delta={a:cells[key][a]-cells[refkey][a] for a in ['f1','precision','recall']},paired_ci95={a:np.quantile(boots[key][a]-boots[refkey][a],[.025,.975]).tolist() for a in ['f1','precision','recall']}))
   write(OUT/'STATUS.json',dict(scope=scope,backbone=b,dataset=ds,cells=len(audits)//3,time=time.time()));print(scope,b,ds,flush=True)
write(OUT/'cells.json',results);write(OUT/'paired_comparisons.json',comparisons);write(OUT/'zero_audit.json',audits)
summary=[]
for scope in ['local_full','common_valid','api_matched']:
 for m in ['ALL']+sorted({r['model'] for r in results if r['scope']==scope}):
  for pol in POLICIES:
   selected=[r for r in results if r['scope']==scope and (m=='ALL' or r['model']==m) and r['policy']==pol]
   good=collections.defaultdict(set)
   for r in selected:
    if r['feasible']:good[r['model'],r['backbone'],r['dataset']].add(r['condition'])
   matchedkeys={k for k,v in good.items() if set(CS)<=v}
   for c in CS:
    rr=[r for r in selected if r['condition']==c and (r['model'],r['backbone'],r['dataset']) in matchedkeys]
    if rr:summary.append(dict(scope=scope,model=m,policy=pol,condition=c,matched_cells=len(rr),total_cells=len(selected)//3,**{a:float(np.mean([r[a] for r in rr])) for a in ['f1','precision','recall']},validation_bypass_cells=sum(r['validation_rejected']==0 for r in rr),test_bypass_cells=sum(r['test_rejected']==0 for r in rr)))
write(OUT/'summary.json',summary)
def table(rr,cols):
 def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
 return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k in cols)+'</tr>' for r in rr)+'</table>'
body='<h1>Core evaluation: explicitly active LLM policies</h1><p>Original results retained. Every new threshold is selected on validation only. Active requires both accepted and rejected validation referrals; active20 requires20 of each. Missing/infeasible cells remain reported. API scopes are conditional on response availability; local_full estimates full-test performance. Means below compare identical feasible cells across conditions within each policy. Different policies may have different feasible coverage.</p>'
body+=table([r for r in summary if r['model']=='ALL'],['scope','policy','condition','matched_cells','total_cells','f1','precision','recall','validation_bypass_cells','test_bypass_cells'])
body+='<h2>Per judge</h2>'+table([r for r in summary if r['model']!='ALL'],['scope','model','policy','condition','matched_cells','f1','precision','recall','validation_bypass_cells'])
body+='<p>cells.json retains all thresholds, probe-only references, support and pointwise intervals. paired_comparisons.json has2000 paired stratified resamples per cell. Global intervals are not constructed by treating repeated targets across backbones/judges as independent. Original-probe local metrics use exact full-test predictions; sampled-probe and cascades use frozen design weights.</p>'
(OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}th{background:#eef}</style>'+body)
write(OUT/'COMPLETED.json',dict(time=time.time(),replayed_original_cells=tests,raw_score_replay=dict(local=900000,api=600000),inputs={str(p):sha(p) for p in [S/'results/scores.npz',S/'frozen/design.json.gz',R/'PROTOCOL.json',Path(__file__)]},outputs={p.name:sha(p) for p in OUT.glob('*') if p.name not in ['COMPLETED.json','STATUS.json']},new_model_calls=0))
print('COMPLETE',flush=True)
for r in summary:
 if r['model']=='ALL':print(r)
