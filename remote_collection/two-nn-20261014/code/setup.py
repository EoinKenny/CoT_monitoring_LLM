import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
from common import *
import numpy as np,joblib,sys
from threadpoolctl import threadpool_limits
threadpool_limits(8)
FOOTER='Positive-class probability (0-100):'
INTRO='You will also receive the most similar training example, retrieved using the activation representation used by the probe.'

def transform(messages,second,names):
 system,user=[m['content'] for m in messages]
 assert system.count(INTRO)==1 and user.endswith(FOOTER)
 system=system.replace(INTRO,'You will also receive the two most similar distinct training examples, retrieved using the activation representation used by the probe.',1)
 system=system.replace("The reference example includes its dataset label and the probe's prediction.","Each reference example includes its dataset label and the probe's prediction.")
 system=system.replace('the reference and the current conversation','either reference and the current conversation').replace('Use the reference as supporting context','Use the references as supporting context')
 original=user[:-len(FOOTER)]
 extra='The second most similar distinct training example was:\n'+json.dumps(second['messages'],ensure_ascii=True)+'\n\nReference example 2:\n'+f"Dataset label: {names[second['label']]} ({second['label']}) |\nProbe prediction: {names[second['prediction']]} ({second['prediction']})\n\n"
 result=[dict(role='system',content=system),dict(role='user',content=original+extra+FOOTER)]
 assert result[1]['content'].startswith(original)
 assert system.split('Return only one integer',1)[1]==messages[0]['content'].split('Return only one integer',1)[1]
 return result

def best_other(Z,T,excluded,chunk=8192):
 n=len(Z);best=np.zeros(n,dtype=int);scores=np.full(n,-np.inf);top=np.zeros(n,dtype=int);topscore=np.full(n,-np.inf)
 for start in range(0,len(T),chunk):
  sim=Z@T[start:start+chunk].T;assert np.isfinite(sim).all()
  loc=sim.argmax(axis=1);val=sim[np.arange(n),loc];better=val>topscore;top[better]=start+loc[better];topscore[better]=val[better]
  mask=(excluded>=start)&(excluded<start+sim.shape[1]);sim[np.flatnonzero(mask),excluded[mask]-start]=-np.inf
  loc=sim.argmax(axis=1);val=sim[np.arange(n),loc];better=val>scores;best[better]=start+loc[better];scores[better]=val[better]
 assert np.isfinite(scores).all() and np.all(best!=excluded)
 return best,scores,top,topscore

def retrieve(b,ds,rows,explanations):
 src=HARD/b;out=ROOT/'retrieval'/b/f'{ds}.json'
 files={suffix:src/f'{ds}_{suffix}' for suffix in ['features.npy','metadata.json','probe.joblib','predictions.npz']}
 proof=read(files['probe.joblib'].resolve().parent/f'{ds}_TRAIN_COMPLETED.json')
 hashes={str(p.resolve()):sha(p) for p in files.values()}
 for p in files.values():assert hashes[str(p.resolve())]==proof['sha256'][p.name],str(p)
 selected=[r for r in rows if r['dataset']==ds];assert len(selected)==1000
 sources=dict(files=hashes,selection_sha256=sha(PARENT/'score'/b/'selected_flags.json'),explanations_sha256=sha(src/'judge_inputs/explanations.jsonl'))
 if out.exists():
  saved=read(out);assert saved['sources']==sources;assert [r['sample_id'] for r in saved['records']]==[r['sample_id'] for r in selected];return
 meta=read(files['metadata.json']);X=np.load(files['features.npy'],mmap_mode='r');fit=joblib.load(files['probe.joblib']);scaler,model=fit['scaler'],fit['model'];threshold=proof['result']['threshold'];z=np.load(files['predictions.npz'])
 assert len(meta)==len(X)==len(z['probabilities']) and np.array_equal(z['labels'],[r['label'] for r in meta]) and np.array_equal(z['splits'],[r['split'] for r in meta])
 train=np.array(sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']))
 assert len(train)>1 and len({meta[i]['immutable_id'] for i in train})==len(train)
 assert not {meta[i]['group'] for i in train}&{r['group'] for r in selected}
 positions={int(ix):k for k,ix in enumerate(train)}
 T=scaler.transform(X[train]).astype(np.float64);assert np.isfinite(T).all();T/=np.maximum(np.linalg.norm(T,axis=1)[:,None],1e-30)
 records=[];started=time.time()
 for a in range(0,len(selected),64):
  reserve();group=selected[a:a+64];ix=[r['index'] for r in group]
  for r in group:assert meta[r['index']]['immutable_id']==r['sample_id'] and meta[r['index']]['split']==r['split']
  Z=scaler.transform(X[ix]).astype(np.float64);assert np.isfinite(Z).all();Z/=np.maximum(np.linalg.norm(Z,axis=1)[:,None],1e-30)
  first=[explanations[r['sample_id']]['nearest'] for r in group];excluded=np.array([positions[x['index']] for x in first]);chosen,cos,top,topcos=best_other(Z,T,excluded)
  indices=train[chosen];prob=model.predict_proba(scaler.transform(X[indices]))[:,1];assert np.isfinite(prob).all()
  for k,r in enumerate(group):
   n1=first[k];n1meta=meta[n1['index']];n2=meta[indices[k]]
   assert n1meta['split']=='train' and n1meta['messages']==n1['messages'] and n1meta['label']==n1['label']
   assert n2['immutable_id']!=n1meta['immutable_id'] and n2['split']=='train'
   second=dict(index=int(indices[k]),immutable_id=n2['immutable_id'],messages=n2['messages'],label=int(n2['label']),prediction=int(prob[k]>=threshold),probe_probability=float(prob[k]),cosine=float(cos[k]))
   records.append(dict(sample_id=r['sample_id'],dataset=ds,split=r['split'],first_id=n1meta['immutable_id'],first_label=n1['label'],first_prediction=n1['prediction'],second=second,labels_agree=n1['label']==second['label'],predictions_agree=n1['prediction']==second['prediction'],diagnostic_recomputed_top1_matches=bool(top[k]==excluded[k]),diagnostic_recomputed_top1_id=meta[train[top[k]]]['immutable_id']))
  write(ROOT/'RETRIEVAL_STATUS.json',dict(backbone=b,dataset=ds,completed=len(records),total=1000,time=time.time()))
 write(out,dict(backbone=b,dataset=ds,sources=sources,training_n=len(train),training_ids_sha256=digest(json.dumps([meta[i]['immutable_id'] for i in train],separators=(',',':'))),records=records,seconds=time.time()-started,diagnostic_top1_mismatches=sum(not r['diagnostic_recomputed_top1_matches'] for r in records)))
 print('retrieved',b,ds,len(records),'seconds',round(time.time()-started,1),flush=True)

def main():
 import re,collections
 reserve();assert read(PARENT/'COMPLETED.json')['n']==900000
 write(ROOT/'judge_specs.json',read(PARENT/'judge_specs.json'))
 for b in BBS:
  rows=read(PARENT/'score'/b/'selected_flags.json');assert len(rows)==10000 and len({r['sample_id'] for r in rows})==10000
  assert not {(r['dataset'],r['group']) for r in rows if r['split']=='val'}&{(r['dataset'],r['group']) for r in rows if r['split']=='test'}
  d=ROOT/'score'/b;d.mkdir(parents=True,exist_ok=True)
  for name in ['selected_flags.json','SAMPLE_FROZEN.json','THRESHOLDS.json']:
   raw=(PARENT/'score'/b/name).read_bytes()
   if (d/name).exists():assert (d/name).read_bytes()==raw
   else:(d/name).write_bytes(raw)
  ef=HARD/b/'judge_inputs/explanations.jsonl';assert sha(ef)==read(ef.parent/'EXTRACTION_COMPLETED')['explanations_sha256'];expl={r['sample_id']:r for r in jl(ef)}
  for ds in sorted({r['dataset'] for r in rows}):
   dest=d/f'{ds}_predictions.npz';target=(HARD/b/f'{ds}_predictions.npz').resolve()
   if dest.exists():assert dest.resolve()==target
   else:dest.symlink_to(target)
   retrieve(b,ds,rows,expl)
 # Canonical prompts are judge independent, ordered exactly as their original cells.
 parent= PARENT/'canonical/cases.jsonl';assert sha(parent)==read(PARENT/'canonical/PREPARED.json')['cases_sha256']
 second={};retrieval_hashes={}
 for b in BBS:
  for p in sorted((ROOT/'retrieval'/b).glob('*.json')):
   for r in read(p)['records']:second[b,r['sample_id']]=r['second']
   retrieval_hashes[str(p.relative_to(ROOT))]=sha(p)
 assert len(second)==TOTAL
 dest=ROOT/'canonical/cases.jsonl';dest.parent.mkdir(exist_ok=True);assert not dest.exists(),'Canonical file already exists; inspect before repeating setup'
 ids=[];by=collections.defaultdict(list);examples={}
 # Resolve names across all parent reference labels without any target gold.
 names_by={}
 for r in jl(parent):
  if r['condition']!='nearest':continue
  names=names_by.setdefault((r['backbone'],r['dataset']),{})
  for kind in ['Dataset label','Probe prediction']:
   hit=re.search(r'\n'+kind+r': (.+) \(([01])\)',r['messages'][1]['content']);assert hit;names[int(hit[2])]=hit[1]
 assert all(set(x)=={0,1} for x in names_by.values())
 examples={}
 with dest.with_suffix('.tmp').open('w') as f:
  for r in jl(parent):
   if r['condition']!='nearest':continue
   b=r['backbone'];sid=r['sample_id'];msg=transform(r['messages'],second[b,sid],names_by[b,r['dataset']]);mh=digest(json.dumps(msg,ensure_ascii=False,separators=(',',':')));pid='nn2_'+digest(json.dumps([b,sid,mh],separators=(',',':')))
   out=dict(backbone=b,dataset=r['dataset'],sample_id=sid,split=r['split'],condition='nearest2',mode='score',messages=msg,messages_sha256=mh,prompt_id=pid,parent_prompt_id=r['prompt_id']);f.write(json.dumps(out,ensure_ascii=False)+'\n');ids.append(pid);by[b].append(sid);examples.setdefault(b,msg)
  f.flush();os.fsync(f.fileno())
 assert len(ids)==len(set(ids))==TOTAL
 for b in BBS:assert by[b]==[r['sample_id'] for r in read(ROOT/'score'/b/'selected_flags.json')]
 dest.with_suffix('.tmp').replace(dest);h=sha(dest)
 write(ROOT/'canonical/PREPARED.json',dict(n=TOTAL,cases_sha256=h,parent_sha256=sha(parent),retrieval_hashes=retrieval_hashes));write(ROOT/'PROMPT_EXAMPLES.json',examples)
 write(ROOT/'SETUP_COMPLETED.json',dict(n=TOTAL,calls=300000,canonical_sha256=h,retrieval_hashes=retrieval_hashes,time=time.time()))
if __name__=='__main__':main()
