from pathlib import Path
import os,json,hashlib,time,shutil
import numpy as np,joblib
R=Path('/home/ubuntu/experiments/nn-embedding-gate-20261019');R.mkdir(exist_ok=True)
assert shutil.disk_usage(R).free>20*1024**3
H=R.parent/'full-threshold-sweep-20261009';P=R.parent/'probability-500-20261012';DS=['aegis','beavertails','toxicchat','toxigen'];BB=['llama','qwen','gemma','llama70','qwen122']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
for b in BB:
 out=R/b;out.mkdir(exist_ok=True)
 if (out/'COMPLETED.json').exists():
  rec=json.loads((out/'COMPLETED.json').read_text())
  for name,digest in rec['outputs'].items():assert sha(out/name)==digest
  print('already exported',b,flush=True);continue
 selected=[r for r in json.loads((P/'score'/b/'selected_flags.json').read_text()) if r['dataset'] in DS];need={r['sample_id'] for r in selected};refs={}
 with open(H/b/'judge_inputs/explanations.jsonl') as f:
  for line in f:
   ex=json.loads(line)
   if ex['sample_id'] in need:refs[ex['sample_id']]=ex
 assert len(selected)==len(refs)==4000
 target=[];neighbor=[];mapping=[];means=[];scales=[];sources=[];diagnostics=[]
 for ds in DS:
  root=H/b;mp=root/(ds+'_metadata.json');fp=root/(ds+'_features.npy');pp=root/(ds+'_probe.joblib');meta=json.loads(mp.read_text());X=np.load(fp,mmap_mode='r');probe=joblib.load(pp);sc=probe['scaler'];means.append(sc.mean_);scales.append(sc.scale_)
  subset=[r for r in selected if r['dataset']==ds];ti=np.array([r['index'] for r in subset]);ni=np.array([refs[r['sample_id']]['nearest']['index'] for r in subset]);T=np.array(X[ti],dtype=np.float32);N=np.array(X[ni],dtype=np.float32);assert np.isfinite(T).all() and np.isfinite(N).all()
  for i,r in enumerate(subset):
   ex=refs[r['sample_id']];nm=meta[ni[i]];tm=meta[ti[i]];assert tm['immutable_id']==r['sample_id'] and tm['split']==r['split'] and int(tm['label'])==r['true_label'];assert nm['split']=='train' and int(nm['label'])==ex['nearest']['label'];assert tm['messages']==ex['messages'] and nm['messages']==ex['nearest']['messages'];assert str(tm['group'])!=str(nm['group'])
   mapping.append(dict(backbone=b,**r,nearest_index=int(ni[i]),nearest_id=nm.get('immutable_id',ds+':'+str(nm.get('input_hash',nm['id']))),nearest_label=int(nm['label']),nearest_cosine=ex['nearest']['cosine'],array_row=len(mapping),scaler_row=DS.index(ds)))
  a=sc.transform(T).astype(np.float64);c=sc.transform(N).astype(np.float64);cos=np.sum(a*c,axis=1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(c,axis=1));old=np.array([refs[r['sample_id']]['nearest']['cosine'] for r in subset]);diagnostics.append(dict(dataset=ds,cosine_max_abs_difference=float(np.max(abs(cos-old))),cosine_mean_abs_difference=float(np.mean(abs(cos-old)))))
  target.append(T);neighbor.append(N);sources.append(dict(dataset=ds,feature_path=str(fp.resolve()),shape=list(X.shape),dtype=str(X.dtype),feature_size=fp.stat().st_size,metadata_sha256=sha(mp),probe_sha256=sha(pp)))
 np.save(out/'target.npy',np.concatenate(target));np.save(out/'neighbor.npy',np.concatenate(neighbor));np.savez(out/'scalers.npz',mean=np.stack(means),scale=np.stack(scales));(out/'mapping.json').write_text(json.dumps(mapping));
 files=['target.npy','neighbor.npy','scalers.npz','mapping.json'];receipt=dict(backbone=b,n=4000,datasets=DS,sources=sources,numerical_agreement_diagnostic_only=diagnostics,outputs={name:sha(out/name) for name in files},time=time.time());(out/'COMPLETED.json').write_text(json.dumps(receipt,indent=2));print('exported',b,np.concatenate(target).shape,flush=True)
(R/'EXPORT_COMPLETED.json').write_text(json.dumps(dict(backbones=BB,rows=20000,files={b:sha(R/b/'COMPLETED.json') for b in BB}),indent=2))
