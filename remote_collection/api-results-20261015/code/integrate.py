import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
from pathlib import Path
import json,gzip,csv,hashlib,re,collections,ast,time,html
import numpy as np
R=Path(__file__).resolve().parent;F=R/'frozen';OUT=R/'results';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*2**20),b''):h.update(x)
 return h.hexdigest()
def jl(p):
 with gzip.open(p,'rt') as f:
  for line in f:yield json.loads(line)
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
for name,h in read(F/'EXPORT_MANIFEST.json')['files'].items():assert sha(F/name)==h
source=R.parent/'two_nn_20261014/code/numerics_source.py';ns={'np':np};tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose','bootstrap','boot_metrics']],type_ignores=[]),str(source),'exec'),ns)
with gzip.open(F/'design.json.gz','rt') as f:design=json.load(f)
BBS=list(design['samples']);CS=['baseline','saliency','nearest'];LOCAL=['qwen','gemma','mistral','olmo','glm','qwen122'];API=['claude4.8-opus','claude4.5-sonnet','gpt-astra','gpt-4o'];ALL=LOCAL+API
cases=[];lookup={}
for b in BBS:
 for r in design['samples'][b]:lookup[b,r['sample_id']]=len(cases);cases.append(dict(backbone=b,**r))
assert len(cases)==50000
pidmap={}
for r in jl(F/'prompt_index.jsonl.gz'):
 i=lookup[r['backbone'],r['sample_id']];c=CS.index(r['condition']);assert r['dataset']==cases[i]['dataset'] and r['split']==cases[i]['split'];pidmap[r['prompt_id']]=(i,c)
assert len(pidmap)==150000
scores={m:np.full((50000,3),-1,dtype=np.int16) for m in ALL};tokens={m:np.full((50000,3),np.nan) for m in ALL};latencies={m:np.full((50000,3),np.nan) for m in API}
counts=collections.Counter();badtypes=collections.Counter();coverage=collections.Counter();duplicates=0;seen=set()
fields=['source','model','prompt_id','backbone','dataset','sample_id','split','condition','true_label','probe_probability','inclusion_probability','response','status','positive_probability','input_tokens','output_tokens','latency_ms']
with gzip.open(OUT/'combined_judgments.csv.gz','wt',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for r in jl(F/'local_scores.jsonl.gz'):
  i,c=pidmap[r['prompt_id']];case=cases[i];m=r['model'];assert scores[m][i,c]==-1
  scores[m][i,c]=r['positive_probability'];tokens[m][i,c]=r['input_tokens']
  writer.writerow(dict(source='local',model=m,prompt_id=r['prompt_id'],**{k:case[k] for k in ['backbone','dataset','sample_id','split','true_label','probe_probability','inclusion_probability']},condition=CS[c],response=str(r['positive_probability']),status='ok',positive_probability=r['positive_probability'],input_tokens=r['input_tokens'],output_tokens=r['generated_tokens'],latency_ms=''))
 for r in csv.DictReader((R/'responses.csv').open()):
  assert r['prompt_id'] in pidmap;key=r['model'],r['prompt_id'];assert key not in seen;seen.add(key)
  i,c=pidmap[r['prompt_id']];m=r['model'];case=cases[i];assert m in API
  valid=r['status']=='ok' and re.fullmatch(r'(?:0|[1-9][0-9]?|100)',r['response'].strip()) is not None
  counts[m,r['status'],valid]+=1;coverage[m,case['backbone'],case['dataset'],case['split'],CS[c],valid]+=1
  score=int(r['response'].strip()) if valid else -1
  if valid:scores[m][i,c]=score
  else:
   text=r['response'].lower();reason='content_filter' if any(x in text for x in ['content_filter','content_filtered','content management policy']) else 'ambiguous_connection' if r['status']=='ambiguous' else 'provider_error' if text.startswith('error:') else 'extra_text_or_invalid_probability';badtypes[m,reason]+=1
  for field,dest in [('input_tokens',tokens),('latency_ms',latencies)]:
   if r[field].strip():
    value=float(r[field]);assert np.isfinite(value) and value>=0;dest[m][i,c]=value
  writer.writerow(dict(source='api',**{k:r[k] for k in ['model','prompt_id','response','status','input_tokens','output_tokens','latency_ms']},**{k:case[k] for k in ['backbone','dataset','sample_id','split','true_label','probe_probability','inclusion_probability']},condition=CS[c],positive_probability=score if valid else ''))
for m in LOCAL:assert (scores[m]>=0).all()
for m in API:assert sum(v for k,v in counts.items() if k[0]==m)==150000
np.savez_compressed(OUT/'scores.npz',**scores)
valid={m:np.all(scores[m]>=0,axis=1) for m in API};common=np.logical_and.reduce(list(valid.values()));assert common.any()
write(OUT/'coverage.json',dict(rows=1500000,api_rows=600000,local_rows=900000,all_prompt_ids_matched=True,duplicate_response_keys=0,models=[dict(model=m,valid_responses=int((scores[m]>=0).sum()),invalid_responses=int((scores[m]<0).sum()),matched_cases=int(valid[m].sum())) for m in API],all_api_common_cases=int(common.sum()),failure_categories=[dict(model=m,reason=c,n=n) for (m,c),n in badtypes.items()]))
write(OUT/'coverage_by_cell.json',[dict(model=m,backbone=b,dataset=d,split=s,condition=c,valid=v,n=n) for (m,b,d,s,c,v),n in coverage.items()])
write(OUT/'MATCHED_CASES.json',dict(common=[dict(backbone=cases[i]['backbone'],sample_id=cases[i]['sample_id']) for i in np.flatnonzero(common)],per_api={m:[dict(backbone=cases[i]['backbone'],sample_id=cases[i]['sample_id']) for i in np.flatnonzero(mask)] for m,mask in valid.items()}))
print('Coverage',read(OUT/'coverage.json'),flush=True)

# Replay the frozen numerical estimator before changing only the matched-case estimand.
parent={(r['backbone'],r['judge'],r['dataset']):r for r in read(R.parent/'probability_500_20261012/results/pairs.json')}
for b in BBS:
 ds=sorted(design['thresholds'][b])[0];spec=design['thresholds'][b][ds];ts=np.array(spec['thresholds']);ix=[i for i,r in enumerate(cases) if r['backbone']==b and r['dataset']==ds and r['split']=='val'];rows=[cases[i] for i in ix]
 for c in CS:
  g=ns['grid'](rows,scores['qwen'][ix,CS.index(c)],ts,spec['populations']['val']);cut,k=ns['choose'](g['f1'],ts);exp=parent[b,'qwen',ds]['variants'][c+'_joint'];assert abs(ts[k]-exp['probe_threshold'])<1e-15 and cut==exp['probability_threshold']
print('Frozen estimator replay passed',flush=True)

results=[];comparisons=[];bootdata={};cells=collections.defaultdict(list)
for i,r in enumerate(cases):cells[r['backbone'],r['dataset'],r['split']].append(i)
for scope,models,masks in [('common_valid_all_judges',ALL,{m:common for m in ALL}),('api_condition_matched',API,valid)]:
 for b in BBS:
  for ds,spec in sorted(design['thresholds'][b].items()):
   ts=np.array(spec['thresholds'])
   for m in models:
    mask=masks[m];vi=[i for i in cells[b,ds,'val'] if mask[i]];ti=[i for i in cells[b,ds,'test'] if mask[i]];vr=[cases[i] for i in vi];tr=[cases[i] for i in ti]
    assert len(vi)>0 and len(ti)>0
    def pop(rows):return dict(n=sum(1/r['inclusion_probability'] for r in rows),positive=sum(r['true_label']/r['inclusion_probability'] for r in rows))
    vp=pop(vr);tp=pop(tr);weights=np.array([1/r['inclusion_probability'] for r in tr]);ps=np.array([r['probe_probability'] for r in tr]);y=np.array([r['true_label'] for r in tr]);mult=ns['bootstrap'](tr,np.random.default_rng(int(hashlib.sha256(f'{scope}:{b}:{ds}'.encode()).hexdigest()[:16],16)),2000);boots={}
    for c in CS:
     ci=CS.index(c);vs=scores[m][vi,ci];ss=scores[m][ti,ci];vg=ns['grid'](vr,vs,ts,vp);tg=ns['grid'](tr,ss,ts,tp)
     for policy in ['fixed50','joint']:
      cut,k=ns['choose'](vg['f1'],ts,50 if policy=='fixed50' else None);yes=(ps>=ts[k])&(ss>=cut);metrics=ns['metrics'](tg['tp'][cut,k],tg['fp'][cut,k],tp['n'],tp['positive']);bt=ns['boot_metrics'](mult,tr,yes,tp);boots[c,policy]=bt
      referred=ps>=ts[k];it=tokens[m][ti,ci];input_cost=float(np.sum(weights[referred]*it[referred])/tp['n']) if np.isfinite(it[referred]).all() else None
      results.append(dict(scope=scope,source='api' if m in API else 'local',model=m,backbone=b,dataset=ds,condition=c,policy=policy,validation_n=len(vi),test_n=len(ti),probe_threshold=float(ts[k]),probability_threshold=cut,validation_f1=float(vg['f1'][cut,k]),**metrics,referral_rate=float(np.sum(weights*referred)/tp['n']),input_tokens_per_matched_example=input_cost,ci95={k:np.quantile(v,[.025,.975]).tolist() for k,v in bt.items()},estimand='Design-weighted matched valid cases; conditional on response availability, not full-test performance'))
    for c in ['saliency','nearest']:
     for policy in ['fixed50','joint']:
      rr=[next(x for x in reversed(results) if x['scope']==scope and x['model']==m and x['backbone']==b and x['dataset']==ds and x['condition']==cond and x['policy']==policy) for cond in ['baseline',c]]
      comparisons.append(dict(scope=scope,model=m,backbone=b,dataset=ds,condition=c,reference='baseline',policy=policy,delta={k:rr[1][k]-rr[0][k] for k in ['f1','precision','recall']},paired_ci95={k:np.quantile(boots[c,policy][k]-boots['baseline',policy][k],[.025,.975]).tolist() for k in ['f1','precision','recall']}))
   print('evaluated',scope,b,ds,flush=True)
write(OUT/'matched_results.json',results);write(OUT/'paired_comparisons.json',comparisons)
# Preserve full local results as a separate scope, never mix them into the matched comparison.
full=[]
for r in parent.values():
 for c in CS:
  for policy in ['fixed50','joint']:
   v=r['variants'][c+'_'+policy];full.append(dict(scope='local_full_test_estimate',source='local',model=r['judge'],backbone=r['backbone'],dataset=r['dataset'],condition=c,policy=policy,validation_n=500,test_n=500,**{k:v[k] for k in ['f1','precision','recall','probe_threshold','probability_threshold','validation_f1']}))
write(OUT/'local_full_results.json',full)
summary=[]
for scope in ['common_valid_all_judges','api_condition_matched','local_full_test_estimate']:
 data=full if scope=='local_full_test_estimate' else [r for r in results if r['scope']==scope]
 for m in sorted({r['model'] for r in data}):
  for c in CS:
   for policy in ['fixed50','joint']:
    rr=[r for r in data if r['model']==m and r['condition']==c and r['policy']==policy];assert len(rr)==50
    summary.append(dict(scope=scope,model=m,condition=c,policy=policy,cells=50,validation_n=sum(r['validation_n'] for r in rr),test_n=sum(r['test_n'] for r in rr),**{k:float(np.mean([r[k] for r in rr])) for k in ['f1','precision','recall']}))
write(OUT/'summary.json',summary)
with (OUT/'summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
# Descriptive average of paired differences; no independence assumption across backbones/datasets.
macro_effect=[]
for scope in ['common_valid_all_judges','api_condition_matched']:
 for m in sorted({r['model'] for r in summary if r['scope']==scope}):
  for policy in ['fixed50','joint']:
   for c in ['saliency','nearest']:
    rows=[r for r in comparisons if r['scope']==scope and r['model']==m and r['policy']==policy and r['condition']==c]
    macro_effect.append(dict(scope=scope,model=m,condition=c,policy=policy,delta={k:float(np.mean([r['delta'][k] for r in rows])) for k in ['f1','precision','recall']},positive_f1_cells=sum(r['delta']['f1']>1e-12 for r in rows),negative_f1_cells=sum(r['delta']['f1']< -1e-12 for r in rows),n=50))
write(OUT/'macro_effects.json',macro_effect)
write(OUT/'ANALYSIS_MANIFEST.json',dict(input_csv_sha256=sha(R/'responses.csv'),frozen_export_manifest_sha256=sha(F/'EXPORT_MANIFEST.json'),journal_audit_sha256=sha(R/'JOURNAL_AUDIT.json'),numerical_source_sha256=sha(source),script_sha256=sha(Path(__file__)),output_rows=1500000,method='Strict status ok plus complete integer regex. Match all three conditions before threshold selection, using validation only and frozen tie rule. Common view matches all four API judges and all conditions, then reevaluates all six local judges on those same cases. Design weights preserved; denominator is matched weighted cohort. No failure imputation, parsing of prose, or model calls. Per-cell paired bootstrap2000 conditional on observed matched cases. Exact provider configuration remains unavailable from the supplied journal.',time=time.time()))
print('ANALYSIS COMPLETE',flush=True)
