"""Exact cosine retrieval from the complete training pool; stable immutable-ID ties."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from common import *
import numpy as np,joblib
from threadpoolctl import threadpool_limits
threadpool_limits(8);R=B/'judge_inputs';R.mkdir(exist_ok=True);selected=read(B/'selected_flags.json');results=read(B/'results.json');allrecords=[]
for ds in sorted(DATASETS):
 reserve();subset=[r for r in selected if r['dataset']==ds];done=R/f'{ds}_explanations.json';sources=dict(probe=sha(B/f'{ds}_probe.joblib'),selection=sha(B/'selected_flags.json'),features=read(B/f'{ds}_EXTRACTED.json')['sha256'])
 if done.exists():
  saved=read(done);assert saved['sources']==sources;allrecords+=saved['records'];continue
 if not subset:write(done,dict(sources=sources,records=[]));continue
 meta=read(B/f'{ds}_metadata.json');X=np.load(B/f'{ds}_features.npy',mmap_mode='r');fit=joblib.load(B/f'{ds}_probe.joblib');scaler,clf=fit['scaler'],fit['model'];trainix=np.array(sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']));T=scaler.transform(X[trainix]).astype(np.float64);T/=np.maximum(np.linalg.norm(T,axis=1)[:,None],1e-30);sal={}
 for path in sorted((B/'saliency'/ds).glob('*.json')):
  doc=read(path);assert doc['probe_sha256']==sources['probe']
  for row in doc['records']:assert row['sample_id'] not in sal;sal[row['sample_id']]=row
 records=[]
 for a in range(0,len(subset),64):
  group=subset[a:a+64];Z=scaler.transform(X[[r['index'] for r in group]]).astype(np.float64);Z/=np.maximum(np.linalg.norm(Z,axis=1)[:,None],1e-30);best=np.zeros(len(group),dtype=int);bestscore=np.full(len(group),-np.inf)
  for ts in range(0,len(T),8192):
   sim=Z@T[ts:ts+8192].T;local=sim.argmax(axis=1);values=sim[np.arange(len(group)),local];better=values>bestscore;best[better]=ts+local[better];bestscore[better]=values[better]
  for k,r in enumerate(group):
   ni=int(trainix[best[k]]);nr=meta[ni];assert nr['split']=='train';pred=int(clf.predict_proba(scaler.transform(X[ni:ni+1]))[0,1]>=fit['threshold']);s=sal[r['sample_id']]
   records.append(dict(sample_id=r['sample_id'],dataset=ds,messages=meta[r['index']]['messages'],saliency=s['saliency'],pooling_error=s['pooling_error'],nearest=dict(index=ni,id=nr['id'],immutable_id=nr['immutable_id'],source='train',messages=nr['messages'],label=int(nr['label']),prediction=pred,cosine=float(bestscore[k]))))
 write(done,dict(sources=sources,records=records));allrecords+=records;del T,X,meta
 print(json.dumps(dict(event='neighbours_complete',dataset=ds,n=len(records))),flush=True)
assert [r['sample_id'] for r in allrecords]==read(B/'SAMPLE_FROZEN.json')['sample_ids']
tmp=R/'explanations.jsonl.tmp'
with tmp.open('w') as f:
 for r in allrecords:f.write(json.dumps(r)+'\n')
 f.flush();os.fsync(f.fileno())
os.replace(tmp,R/'explanations.jsonl');write(R/'private.json',selected);write(R/'weighted_results.json',dict(results={ds:dict(test=v['metrics']['test']) for ds,v in results.items()}));write(R/'EXTRACTION_COMPLETED',dict(n=len(allrecords),max_pooling_error=max((r['pooling_error'] for r in allrecords),default=0),explanations_sha256=sha(R/'explanations.jsonl'),nearest='exact standardized-activation cosine; complete training pool; immutable-ID ties',sentence_splitter='pysbd English clean=False char_span=True'))
