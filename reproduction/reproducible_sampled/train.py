import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='8'
os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,importlib.util,ast,warnings,time,gc,random
from common import *
import numpy as np,torch,joblib
from transformers import AutoModel,AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,roc_auc_score,accuracy_score,confusion_matrix,roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
b=sys.argv[1];R=ROOT/b;O=OLD[b];sys.path.insert(0,str(R));start=time.time()
threadpool_limits(8);torch.set_num_threads(8);torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED);torch.use_deterministic_algorithms(True);warnings.filterwarnings('error',category=ConvergenceWarning)
bb=read(R/'backbone.json');tok=AutoTokenizer.from_pretrained(bb['path'],local_files_only=True);model=None;dim={'llama':2048,'qwen':2560,'gemma':5376}[b]
if b!='llama':from tokenize_rows import tokenize_record
results={};inventory={};captured={}
def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
def status(event,**kw):write(R/'TRAIN_STATUS.json',dict(event=event,seconds=time.time()-start,**kw));print(json.dumps(dict(event=event,backbone=b,**kw)),flush=True)
def full_rows(ds):
 if ds not in DATASETS[:3]:return read(ROOT/'prepared'/f'{ds}_selected.json')
 B=E/'logistic-probes-20261001';meta=read(B/f'{ds}_metadata.json');splits=read(B/f'{ds}_splits.json');out=[]
 for i,(m,s) in enumerate(zip(meta,splits)):
  assert m['input_hash']==s['input_hash'] and m['label']==s['label']
  if s['split'] not in ['train','val','test']:continue
  out.append(dict(m,split=s['split'],group=s['group'],origin_index=i,legacy_dataset=ds,n_tokens=m['all_tokens'],role='assistant' if ds=='wildguard' else ('user' if ds=='toxicchat' else 'all'),cache_file=str(E/'activation-full-20261001/activations'/m['cache_shard'])))
 return out
for ds in DATASETS:
 reserve();done=R/f'{ds}_TRAIN_COMPLETED.json'
 if done.exists():
  mark=read(done)
  for name,h in mark['sha256'].items():assert sha(R/name)==h,(ds,name,'changed committed artifact')
  results[ds]=mark['result'];continue
 status('preparing',dataset=ds)
 old=read(O/f'{ds}_metadata.json');source=full_rows(ds);by={identity(ds,r):r for r in source};assert len(by)==len(source)
 for split in ['val','test']:
  sig=lambda rows:sorted((identity(ds,r),r['label'],r['group'],r['input_hash']) for r in rows if r['split']==split)
  assert sig(old)==sig(source),(b,ds,split,'heldout membership changed')
 for a in ['train','val','test']:
  for c in ['train','val','test']:
   if a!=c:
    assert not ({r['group'] for r in source if r['split']==a}&{r['group'] for r in source if r['split']==c})
    assert not ({r['input_hash'] for r in source if r['split']==a}&{r['input_hash'] for r in source if r['split']==c})
 # Retain old order and cache indices; append only restored training rows, deterministically.
 rows=[];known=set()
 for r in old:
  uid=identity(ds,r);s=by[uid];assert (r['label'],r['split'],r['group'],r['messages'])==(s['label'],s['split'],s['group'],s['messages']);rows.append(dict(r,immutable_id=uid));known.add(uid)
 added=[by[u] for u in sorted(set(by)-known)];assert all(r['split']=='train' for r in added)
 for k,r in enumerate(added):
  r=dict(r,immutable_id=identity(ds,r))
  if b!='llama':
   r={key:r[key] for key in ['id','messages','label','split','group','input_hash','role','immutable_id']};r.update(tokenize_record(tok,r))
  rows.append(r)
  if k%2000==0:status('tokenizing_missing_training',dataset=ds,completed=k,total=len(added))
 oldx=np.load(O/f'{ds}_features.npy',mmap_mode='r');x=np.empty((len(rows),dim),np.float32);x[:len(old)]=oldx
 reuse_sources=[dict(path=str(O/f'{ds}_features.npy'),sha256=sha(O/f'{ds}_features.npy'),rows=len(old))]
 missing=list(range(len(old),len(rows)))
 if b=='llama' and ds in DATASETS[:3]:
  path=E/'logistic-probes-20261001'/f'{ds}_features.npy';extra=np.load(path,mmap_mode='r')
  for i in missing:x[i]=extra[rows[i]['origin_index']]
  reuse_sources.append(dict(path=str(path),sha256=sha(path),rows=len(missing)));missing=[];del extra
 if b=='llama' and ds=='fava':
  root=E/'training-balance-comparison-20261004';extra_rows=read(root/'fava_selected.json');ix={identity(ds,r):i for i,r in enumerate(extra_rows)};extra=np.load(root/'fava_features.npy',mmap_mode='r')
  for i in missing:
   j=ix[rows[i]['immutable_id']];assert rows[i]['token_ids']==extra_rows[j]['token_ids'] and rows[i]['mask']==extra_rows[j]['mask'];x[i]=extra[j]
  reuse_sources.append(dict(path=str(root/'fava_features.npy'),sha256=sha(root/'fava_features.npy'),rows=len(missing)));missing=[];del extra
 manifest=dict(backbone=bb,old_metadata_sha256=sha(O/f'{ds}_metadata.json'),cache_sources=reuse_sources,missing_vectors=len(missing),reused_vectors=len(rows)-len(missing),split_records=[dict(immutable_id=r['immutable_id'],split=r['split'],label=r['label'],group=r['group'],input_hash=r['input_hash']) for r in rows])
 if (R/f'{ds}_SPLITS.json').exists():assert read(R/f'{ds}_SPLITS.json')==manifest,'Inputs changed across resume'
 else:write(R/f'{ds}_SPLITS.json',manifest)
 if missing:
  if model is None:
   status('loading_backbone',dataset=ds)
   if b=='gemma':
    from backbone_loader import load_model
    model=load_model()
   else:
    model=AutoModel.from_pretrained(bb['path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa');model.layers=torch.nn.ModuleList(list(model.layers[:bb['block_index']+1]));model.norm=torch.nn.Identity();model=model.to('cuda').eval()
   model.layers[bb['block_index']].register_forward_hook(hook)
  order=sorted(missing,key=lambda i:(rows[i]['n_tokens'],rows[i]['immutable_id']));chunks=[];batch=[]
  for i in order:
   if batch and (len(batch)>=4 or (len(batch)+1)*rows[i]['n_tokens']>2048):chunks.append(batch);batch=[]
   batch.append(i)
  if batch:chunks.append(batch)
  cache=R/'added_features'/ds;cache.mkdir(parents=True,exist_ok=True);completed=0
  # Persist groups of <=128 sequences; uncommitted deterministic extraction can be replayed.
  for block in range(0,len(chunks),32):
   reserve();group=chunks[block:block+32];indices=[i for chunk in group for i in chunk];dest=cache/f'{block:07d}.npz';proof=dest.with_suffix('.json')
   if proof.exists():
    assert sha(dest)==read(proof)['sha256'];assert read(proof)['ids']==[rows[i]['immutable_id'] for i in indices];z=np.load(dest);assert z['indices'].tolist()==indices;x[indices]=z['features'];completed+=len(indices);continue
   assert not dest.exists(),'Uncommitted extraction file requires inspection'
   vals=[]
   for idx in group:
    width=max(rows[i]['n_tokens'] for i in idx);pad=tok.pad_token_id if tok.pad_token_id is not None else 128009
    ids=torch.full((len(idx),width),pad,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
    for j,i in enumerate(idx):n=rows[i]['n_tokens'];ids[j,:n]=torch.tensor(rows[i]['token_ids'],device='cuda');att[j,:n]=1
    with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
    h=captured.pop('h');assert torch.isfinite(h).all()
    for j,i in enumerate(idx):v=h[j,rows[i]['mask']].float().mean(0).cpu().numpy();x[i]=v;vals.append(v)
    del h,ids,att
   with dest.with_suffix('.tmp').open('wb') as f:np.savez(f,indices=np.array(indices),features=np.array(vals));f.flush();os.fsync(f.fileno())
   os.replace(dest.with_suffix('.tmp'),dest);write(proof,dict(sha256=sha(dest),ids=[rows[i]['immutable_id'] for i in indices]));completed+=len(indices);status('extracting_missing_training',dataset=ds,completed=completed,total=len(missing))
 assert np.isfinite(x).all();status('fitting_probe',dataset=ds,train=sum(r['split']=='train' for r in rows))
 y=np.array([r['label'] for r in rows]);sp=np.array([r['split'] for r in rows]);tr=sp=='train';va=sp=='val';scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x)
 clf=LogisticRegression(C=.001 if ds=='high_stakes' else .01,class_weight='balanced',max_iter=2000,tol=1e-6,solver='lbfgs',random_state=SEED).fit(z[tr],y[tr]);p=clf.predict_proba(z)[:,1]
 precision,recall,ts=precision_recall_curve(y[va],p[va]);f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15);threshold=float(ts[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])
 scores={}
 for s in ['train','val','test']:
  sel=sp==s;tn,fp,fn,tp=confusion_matrix(y[sel],p[sel]>=threshold,labels=[0,1]).ravel();scores[s]=dict(confusion_metrics(tp,fp,fn,tn),n=int(sum(sel)),positive=int(sum(y[sel])),auc=float(roc_auc_score(y[sel],p[sel])))
 result=dict(threshold=threshold,metrics=scores,counts={s+':'+str(c):int(sum((sp==s)&(y==c))) for s in ['train','val','test'] for c in [0,1]},flagged_test=int(sum((sp=='test')&(p>=threshold))),training='all eligible, class_weight balanced',C=clf.C,missing_vectors=len(missing))
 np.save(R/f'{ds}_features.npy',x);write(R/f'{ds}_metadata.json',rows);joblib.dump(dict(model=clf,scaler=scaler,threshold=threshold,config=dict(backbone=bb,seed=SEED,C=clf.C,class_weight='balanced')),R/f'{ds}_probe.joblib');np.savez(R/f'{ds}_predictions.npz',labels=y,probabilities=p,splits=sp)
 fpr,tpr,t=roc_curve(y[sp=='test'],p[sp=='test']);np.savez(R/f'{ds}_roc.npz',fpr=fpr,tpr=tpr,thresholds=t)
 names=[f'{ds}_{suffix}' for suffix in ['features.npy','metadata.json','probe.joblib','predictions.npz','SPLITS.json','roc.npz']]
 write(done,dict(result=result,sha256={n:sha(R/n) for n in names}));results[ds]=result;write(R/'partial_results.json',results);status('dataset_complete',dataset=ds,result=result);del x,z,oldx,rows,source,by;gc.collect()
write(R/'results.json',results);write(R/'COMPLETED',dict(datasets=10,seconds=time.time()-start));status('completed')
