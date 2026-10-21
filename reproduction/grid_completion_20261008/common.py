import os,json,hashlib,shutil,time
from pathlib import Path
ROOT=Path(os.environ.get('STUDY_ROOT','/home/ubuntu/experiments/grid-completion-20261008'))
E=ROOT.parent
OLD={b:ROOT/b for b in ['llama','qwen','gemma','llama70','qwen122']}
B=ROOT/'qwen122'
PRIOR=E/'reproducible-sampled-20261004'
PILOT=E/'llama70b-bf16-pilot-20261005'
MODEL=Path('/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b')
DATASETS=['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']
JUDGES=['qwen','gemma','llama','glm','olmo','qwen122'];CONDITIONS=['baseline','saliency','nearest']
SEED=20261004
CPU='/home/ubuntu/misalignment-indicators-repro/.venv-uv/bin/python'
GPU=str(E/'judge-pilot-20261001/.venv-vllm/bin/python')
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def read(p):return json.loads(Path(p).read_text())
def reserve():
 if shutil.disk_usage(ROOT).free<20*2**30:raise RuntimeError('Disk reserve below 20 GiB; stopped safely')
def identity(ds,r):return ds+':'+digest(json.dumps([str(r['id']),r['input_hash']],separators=(',',':')))
def sample_flags(rows,backbone,dataset):
 assert len({r['sample_id'] for r in rows})==len(rows)
 bins={s:[r for r in rows if r['true_label']==y] for s,y in [('TP',1),('FP',0)]}
 quota={s:min(250,len(v)) for s,v in bins.items()};remaining=min(500,len(rows))-sum(quota.values())
 for s in ['TP','FP']:
  n=min(remaining,len(bins[s])-quota[s]);quota[s]+=n;remaining-=n
 assert remaining==0
 chosen=[];counts={}
 for s in ['TP','FP']:
  def rank(r):return digest('\0'.join([str(SEED),backbone,dataset,s,r['sample_id']]))
  pool=sorted(bins[s],key=lambda r:(rank(r),r['sample_id']));n=len(pool);k=quota[s]
  counts[s]=dict(population=n,selected=k,inclusion_probability=k/n if n else None,census=n==k)
  chosen += [dict(r,stratum=s,selection_hash=rank(r),inclusion_probability=k/n,population_weight=n/k) for r in pool[:k]]
 return chosen,dict(population_flags=len(rows),selected_flags=len(chosen),balanced_250_possible=all(len(v)>=250 for v in bins.values()),strata=counts)
def confusion_metrics(tp,fp,fn,tn):
 n=tp+fp+fn+tn
 return dict(tp=float(tp),fp=float(fp),fn=float(fn),tn=float(tn),f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,accuracy=(tp+tn)/n if n else 0.,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.)
def pipeline_counts(base,selected,predictions):
 tp=fp=0.
 for r,p in zip(selected,predictions):
  assert p in [0,1]
  if r['true_label']:tp+=p*r['population_weight']
  else:fp+=p*r['population_weight']
 return confusion_metrics(tp,fp,base['fn']+base['tp']-tp,base['tn']+base['fp']-fp)
