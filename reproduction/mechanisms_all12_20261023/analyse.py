import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import numpy as np,json,gzip,csv,re,collections,itertools,hashlib
R=Path(__file__).resolve().parent;P=R.parent;O=R/'results';O.mkdir(exist_ok=True)
LAB=['labelled_similarity','important_contrast'];VALUES=['no','yes','uncertain','missing'];CODE={x:i for i,x in enumerate(VALUES)}
M=['qwen','gemma','mistral','olmo','glm','qwen122','claude4.5-sonnet','claude4.8-opus','gpt-4o','gpt-astra','claude-opus-5.5','gpt-5.5']
I=P/'nn_mechanism_agreement_20261019/inputs';mapping=[json.loads(l) for l in open(I/'mapping_PRIVATE.jsonl')];assert len(mapping)==25000
unit={r['unit_id']:i for i,r in enumerate(mapping)};a=np.full((12,25000,2),3,dtype=np.int8);counts=collections.Counter()
def put(m,u,v):
 assert set(v)==set(LAB) and all(v[k] in CODE and v[k]!='missing' for k in LAB)
 a[M.index(m),unit[u]]=[CODE[v[k]] for k in LAB]
for m in ['qwen','gemma','qwen122']:
 f=I/m/'annotations.jsonl';comp=json.load(open(I/m/'COMPLETED.json'));assert hashlib.sha256(f.read_bytes()).hexdigest()==comp['annotations_sha256']
 for r in map(json.loads,open(f)):
  if r['status']=='valid':put(m,r['unit_id'],{k:r[k] for k in LAB})
with gzip.open(R/'local_additional.jsonl.gz','rt') as f:
 seen=set()
 for r in map(json.loads,f):
  key=r['judge'],r['prompt_id'];assert key not in seen;seen.add(key)
  if r['status']=='valid':put(r['judge'],r['prompt_id'],r['value'])
 assert len(seen)==75000
normal={'stop','end_turn','completed','eos','stop_sequence'};seen=set()
def no_duplicate(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate')
  d[k]=v
 return d
with open('/Users/eoinkenny/Desktop/six_judges_responses_5col.csv',newline='') as f:
 for r in csv.DictReader(f):
  if r['prompt_id'] not in unit:continue
  key=r['model'],r['prompt_id'];assert key not in seen;seen.add(key);t=r['response'].strip();wrapped=False
  if r['status'] not in ['ok','invalid'] or r['finish_reason'] not in normal:counts[r['model'],'unavailable']+=1;continue
  match=re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```',t,re.S)
  if match:t=match.group(1);wrapped=True
  try:v=json.loads(t,object_pairs_hook=no_duplicate);put(r['model'],r['prompt_id'],v);counts[r['model'],'fenced_recovered' if wrapped else 'valid']+=1
  except (ValueError,AssertionError,TypeError):counts[r['model'],'unavailable']+=1
assert len(seen)==150000
# Strict majority of all 12 annotators; ties/weak agreement remain unresolved.
def consensus(v,minimum):
 n=np.stack([(v==k).sum(axis=0) for k in range(3)]);res=n.argmax(axis=0).astype(np.int8);res[n.max(axis=0)<minimum]=3;return res
major=consensus(a,7);loo=np.stack([consensus(np.delete(a,i,axis=0),6) for i in range(12)])
np.savez_compressed(R/'annotations.npz',annotations=a,majority=major,leave_one_out=loo,models=np.array(M),unit_ids=np.array(list(unit)))
cohorts={'all10':None,'safety6':{'aegis','beavertails','toxicchat','toxigen','wildguard','wildjailbreak'},'hallucination3':{'fava','halueval','ragtruth'},'high_stakes':{'high_stakes'},'toxicity4':{'aegis','beavertails','toxicchat','toxigen'}}
ds=np.array([r['dataset'] for r in mapping]);bb=np.array([r['backbone'] for r in mapping]);cell=bb.astype(object)+':'+ds;cm={co:np.ones(25000,bool) if ss is None else np.isin(ds,list(ss)) for co,ss in cohorts.items()}
frequency=[]
for an,v in list(zip(M,a))+[('majority12',major)]:
 for co,mask in cm.items():
  for li,lab in enumerate(LAB):
   for k,name in enumerate(VALUES):frequency.append(dict(annotator=an,cohort=co,label=lab,level=('unresolved' if an=='majority12' and k==3 else name),n=int((v[mask,li]==k).sum()),total=int(mask.sum()),fraction=float((v[mask,li]==k).mean())))
agreement=[]
for i,j in itertools.combinations(range(12),2):
 for li,lab in enumerate(LAB):
  ok=(a[i,:,li]!=3)&(a[j,:,li]!=3);x=a[i,ok,li];y=a[j,ok,li];obs=(x==y).mean();expected=sum((x==k).mean()*(y==k).mean() for k in range(3));agreement.append(dict(annotator_a=M[i],annotator_b=M[j],label=lab,n=len(x),agreement=obs,kappa=(obs-expected)/(1-expected) if expected<1 else float('nan')))
d=json.load(gzip.open(P/'api_results_20261015/frozen/design.json.gz','rt'));cases=[dict(backbone=b,**r) for b,rr in d['samples'].items() for r in rr];look={(r['backbone'],r['sample_id']):i for i,r in enumerate(cases)};ix=np.array([look[r['backbone'],r['sample_id']] for r in mapping]);y=np.array([cases[i]['true_label'] for i in ix]);probe=np.array([cases[i]['probe_probability'] for i in ix]);w0=np.array([1/cases[i]['inclusion_probability'] for i in ix]);z=np.load(P/'main_performance_all12_20261023/scores.npz')
# Equal mass per judge/backbone/dataset cell before conditioning on outcome/labels.
prev=[];conditional=[];events=[];balanced=[]
for ji,m in enumerate(M):
 s=z[m][ix];valid=(s[:,0]>=0)&(s[:,2]>=0);w=np.zeros(25000)
 for c in np.unique(cell):
  mask=valid&(cell==c);w[mask]=w0[mask]/w0[mask].sum()
 for policy in ['judge50_direct','pipeline50']:
  ref=np.ones(25000,bool) if policy=='judge50_direct' else probe>=.5
  bc=((s[:,0]>=50)&ref)==y;nc=((s[:,2]>=50)&ref)==y;e=np.where(~bc&nc,0,np.where(bc&~nc,1,2));assert np.all((e>=0)&(e<=2))
  for co,mask0 in cm.items():
   mask=mask0&valid
   for ei,event in enumerate(['helpful','harmful','unchanged']):
    mm=mask&(e==ei);events.append(dict(judge=m,policy=policy,cohort=co,outcome=event,n=int(mm.sum()),mass=float(w[mm].sum()),total_mass=float(w[mask].sum())))
  annsets=[('majority12',major),('independent_majority',loo[ji])]+list(zip(M,a))
  for an,vals in annsets:
   for li,lab in enumerate(LAB):
    known=valid&(vals[:,li]<2);yes=vals[:,li]==1
    # Counts and weighted masses allow exact pooled summaries without replicating cases.
    for co,mask0 in cm.items():
     mask=known&mask0
     for ei,event in enumerate(['helpful','harmful','unchanged']):
      mm=mask&(e==ei);prev.append(dict(annotator=an,judge=m,policy=policy,cohort=co,label=lab,outcome=event,n=int(mm.sum()),yes_n=int((mm&yes).sum()),mass=float(w[mm].sum()),yes_mass=float(w[mm&yes].sum()),excluded_n=int((valid&mask0&~known&(e==ei)).sum())))
    if an=='independent_majority':
     for c in np.unique(cell):
      mask=known&(cell==c);den=[w[mask&(e==ei)].sum() for ei in [0,1]]
      if min(den)>0:
       rates=[w[mask&(e==ei)&yes].sum()/den[ei] for ei in [0,1]];b,ds0=c.split(':');balanced.append(dict(judge=m,backbone=b,dataset=ds0,policy=policy,label=lab,helpful_yes=rates[0],harmful_yes=rates[1],delta=rates[0]-rates[1]))
     for dim,values in [('dataset',ds),('backbone',bb)]:
      for group in np.unique(values):
       mask=known&(values==group)
       for ei,event in enumerate(['helpful','harmful','unchanged']):
        mm=mask&(e==ei);conditional.append(dict(dimension=dim,group=group,judge=m,policy=policy,label=lab,outcome=event,n=int(mm.sum()),mass=float(w[mm].sum()),yes_mass=float(w[mm&yes].sum())))
 print('analysed',m,flush=True)
def save(name,rr):
 with open(O/(name+'.csv'),'w') as f:q=csv.DictWriter(f,fieldnames=list(rr[0]));q.writeheader();q.writerows(rr)
for name,rr in [('label_distribution',frequency),('agreement',agreement),('outcome_counts',events),('prevalence',prev),('breakdowns',conditional),('equal_cell_sensitivity',balanced)]:save(name,rr)
# Pooled across equally weighted original cells; report both conditional directions.
agg=collections.defaultdict(lambda:collections.Counter())
for r in prev:
 key=tuple(r[k] for k in ['annotator','policy','cohort','label','outcome'])
 for k in ['n','yes_n','mass','yes_mass','excluded_n']:agg[key][k]+=r[k]
summary=[]
for key,v in agg.items():summary.append(dict(zip(['annotator','policy','cohort','label','outcome'],key),**v,yes_fraction=v['yes_mass']/v['mass'] if v['mass'] else float('nan')))
save('summary',summary)
for r in summary:
 if r['annotator']=='independent_majority' and r['cohort']=='all10' and r['policy']=='judge50_direct':print(r)
(O/'VERIFIED.json').write_text(json.dumps(dict(annotation_contexts=25000,annotators=12,outcome_judges=12,annotation_records_expected=300000,new_calls=0,local_sha256=hashlib.sha256((R/'local_additional.jsonl.gz').read_bytes()).hexdigest(),parsing=[dict(model=m,outcome=k,n=n) for (m,k),n in counts.items()],main='Original 1NN vs control, judge cutoff50; pipeline50 sensitivity. Independent majority excludes outcome judge and requires6 of11 votes; pooled majority requires7 of12. Uncertain/missing/unresolved excluded from binary associations, retained in prevalence. Weights normalized to equal cell mass before conditioning. Descriptive association, not causal internal reasoning.',selection='Test cases only, valid control and 1NN pair within each outcome judge, no cross-judge exclusion'),indent=2))
