from pathlib import Path
import zipfile,json,collections,csv,gzip,ast,hashlib
import numpy as np
R=Path(__file__).resolve().parent;O=R/'results';F=R/'frozen'
counts=collections.Counter();events={}
with zipfile.ZipFile('/Users/eoinkenny/Downloads/request_journal.zip') as z:
 with z.open('request_journal.txt') as f:
  for line in f:
   r=json.loads(line)
   if r['event']=='dispatch':counts[r['model'],r['prompt_id']]+=1
   elif r['event']=='result':events[r['model'],r['prompt_id']]=(r['status'],r['latency_ms'])
repeated={k for k,n in counts.items() if n>1};statuses=collections.Counter()
for r in csv.DictReader((R/'responses.csv').open()):
 key=r['model'],r['prompt_id'];status,ms=events[key];assert status==r['status'];assert float(ms)==float(r['latency_ms'])
 if key in repeated:statuses[r['status']]+=1
with gzip.open(F/'design.json.gz','rt') as f:design=json.load(f)
cases=[];lookup={}
for b,rows in design['samples'].items():
 for r in rows:lookup[b,r['sample_id']]=len(cases);cases.append(dict(backbone=b,**r))
CS=['baseline','saliency','nearest'];pidmap={}
with gzip.open(F/'prompt_index.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);pidmap[r['prompt_id']]=lookup[r['backbone'],r['sample_id']]
excluded={pidmap[pid] for m,pid in repeated};z=np.load(O/'scores.npz');models=list(z.files);apis=['gpt-astra','gpt-4o','claude4.8-opus','claude4.5-sonnet'];valid={m:np.all(z[m]>=0,axis=1) for m in apis};common=np.logical_and.reduce(list(valid.values()))
source=R.parent/'two_nn_20261014/code/numerics_source.py';ns={'np':np};tree=ast.parse(source.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['metrics','curve','grid','choose']],type_ignores=[]),str(source),'exec'),ns)
old=json.load(open(O/'matched_results.json'));index={(r['scope'],r['model'],r['backbone'],r['dataset'],r['condition'],r['policy']):r for r in old};changes=[]
for scope,ms,mask in [('common_valid_all_judges',models,common),('api_condition_matched',['gpt-astra'],valid['gpt-astra'])]:
 affected={(cases[i]['backbone'],cases[i]['dataset']) for i in excluded if mask[i]}
 for b,ds in sorted(affected):
  vi=[i for i,r in enumerate(cases) if r['backbone']==b and r['dataset']==ds and r['split']=='val' and mask[i] and i not in excluded];ti=[i for i,r in enumerate(cases) if r['backbone']==b and r['dataset']==ds and r['split']=='test' and mask[i] and i not in excluded];vr=[cases[i] for i in vi];tr=[cases[i] for i in ti];ts=np.array(design['thresholds'][b][ds]['thresholds'])
  pop=lambda rows:dict(n=sum(1/r['inclusion_probability'] for r in rows),positive=sum(r['true_label']/r['inclusion_probability'] for r in rows))
  for m in ms:
   for ci,c in enumerate(CS):
    vg=ns['grid'](vr,z[m][vi,ci],ts,pop(vr));tg=ns['grid'](tr,z[m][ti,ci],ts,pop(tr))
    for policy in ['fixed50','joint']:
     cut,k=ns['choose'](vg['f1'],ts,50 if policy=='fixed50' else None);tp=pop(tr);actual=ns['metrics'](tg['tp'][cut,k],tg['fp'][cut,k],tp['n'],tp['positive']);prior=index[scope,m,b,ds,c,policy]
     changes.append(dict(scope=scope,model=m,backbone=b,dataset=ds,condition=c,policy=policy,f1_delta=actual['f1']-prior['f1'],probe_threshold_changed=float(ts[k])!=prior['probe_threshold'],judge_threshold_changed=cut!=prior['probability_threshold']))
macro=[]
for scope,m,c,policy in sorted({(r['scope'],r['model'],r['condition'],r['policy']) for r in changes}):
 delta=sum(r['f1_delta'] for r in changes if (r['scope'],r['model'],r['condition'],r['policy'])==(scope,m,c,policy))/50
 macro.append(dict(scope=scope,model=m,condition=c,policy=policy,macro_f1_change_pp=100*delta))
result=dict(repeated_dispatch_keys=len(repeated),unique_cases=len(excluded),final_statuses=dict(statuses),journal_response_status_and_latency_match=True,excluded_from_primary=False,sensitivity='Remove every case with a repeated dispatch from all conditions and retune on the remaining validation cases; compare with primary results',macro_changes=macro,max_absolute_macro_f1_change_pp=max(abs(x['macro_f1_change_pp']) for x in macro),case_changes=changes)
(O/'journal_sensitivity.json').write_text(json.dumps(result,indent=2));print('Repeated-dispatch sensitivity',result['final_statuses'],'max macro F1 change pp',result['max_absolute_macro_f1_change_pp'])
