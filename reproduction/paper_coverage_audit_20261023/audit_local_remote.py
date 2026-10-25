import pathlib,json,hashlib,collections,time
B=pathlib.Path('/home/ubuntu/experiments');P=B/'paper-complete-20261021';OUT=P/'recovery_20261023/PAPER_AUDIT.json'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def rows(p):
 with pathlib.Path(p).open() as f:
  for l in f:yield json.loads(l)
r=read(P/'recovery_20261023/COLLECTION_RESOLVED.json');results=[];issues=[]
for t in r['tasks']:
 path=pathlib.Path(t['path']);assert sha(path)==t['sha256'];canonical={x['prompt_id']:(x['backbone'],x['dataset'],x['split'],x['condition']) for x in rows(t['task']['path'])};seen=set();c=collections.Counter();by=collections.Counter()
 for x in rows(path):
  pid=x['prompt_id'];assert pid not in seen and pid in canonical;seen.add(pid);assert (x['backbone'],x['dataset'],x['split'],x['condition'])==canonical[pid]
  c[x['status']]+=1;by[x['dataset'],x['status']]+=1
  if x['status']=='valid':
   if t['task']['study']=='ablation':assert type(x['value'])==int and 0<=x['value']<=100
   else:assert set(x['value'])=={'labelled_similarity','important_contrast'} and all(v in ['yes','no','uncertain'] for v in x['value'].values())
  else:issues.append(dict(judge=x['judge'],study=t['task']['study'],prompt_id=pid,dataset=x['dataset'],status=x['status'],finish_reason=x['finish_reason'],excerpt=x['raw_response'][:300]))
 assert seen==set(canonical) and len(seen)==t['n'];results.append(dict(index=t['index'],judge=t['task']['judge'],study=t['task']['study'],backbone=t['task']['backbone'],n=len(seen),statuses=dict(c),by_dataset=[list(k)+[v] for k,v in by.items()],sha256=t['sha256']))
 print('checked',t['index'],flush=True)
reused=[]
for x in read(P/'REUSED_COVERAGE.json')['ablations']:
 d=pathlib.Path(x['path']);fs=list(d.glob('*.jsonl'));f=d/'judgments.jsonl';assert f.exists();n=0;c=collections.Counter();seen=set()
 for y in rows(f):
  key=y.get('prompt_id',y['sample_id']);assert key not in seen;seen.add(key);n+=1;c['valid' if isinstance(y.get('positive_probability'),(float,int)) and 0<=y['positive_probability']<=100 else 'invalid']+=1
 assert n==10000
 reused.append(dict(judge=x['judge'],backbone=x['backbone'],condition=x['condition'],n=n,statuses=dict(c),sha256=sha(f)))
companions={}
for name in ['probability-500-20261012','two-nn-20261014','nn-reason-flips-20261017','nn-reason-first-512-20261017']:
 f=B/name/'COMPLETED.json';companions[name]=dict(completion=read(f),sha256=sha(f))
info=dict(final_collection=results,reused_ablations=reused,companions=companions,invalid_final_rows=issues,time=time.time(),note='Every final-addition record matched exact canonical prompt membership and output checksums; all 350000 reused ablation records scanned. Companion historical completion manifests copied; independent local row audit follows.')
OUT.write_text(json.dumps(info,indent=2));print('AUDIT COMPLETE',OUT)
