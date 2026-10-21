import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,sys,traceback,fcntl,hashlib,warnings
from pathlib import Path
import numpy as np,torch,joblib
from transformers import AutoModel,AutoTokenizer
from safetensors.torch import save_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix,precision_recall_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
R=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');START=time.time();lock=(R/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'results.json').exists()
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def log(event,**kw):
 x=dict(event=event,seconds=time.time()-START,**kw);write(R/'status.json',x);print(json.dumps(x),flush=True)
def fail(typ,exc,tb):write(R/'FAILED.json',dict(error=str(exc),traceback=''.join(traceback.format_exception(typ,exc,tb))));sys.__excepthook__(typ,exc,tb)
sys.excepthook=fail
torch.manual_seed(20261002);np.random.seed(20261002);torch.set_num_threads(8);threadpool_limits(8);warnings.filterwarnings('error',category=ConvergenceWarning)
BACKBONE=json.loads((R/'backbone.json').read_text());MODEL=BACKBONE['path']
config=dict(model=MODEL,block_index=24,hidden_size=2560,dtype='bfloat16',seed=20261002,C=.01,class_weight='balanced',max_input_tokens=None,overlength='Preserve frozen memberships; no truncation, batch smaller for longer Qwen tokenizations',pool='mean content tokens only; response tasks use assistant tokens conditioned on evidence/request',threshold='maximum validation F1; highest threshold breaks ties',split='source groups disjoint; exact duplicate/conflict audit; same balanced training and natural validation/test memberships as frozen Llama run',notes='Full balanced training; full eligible natural-proportion validation and test; former pilot test cases are exploratory rather than fresh independent evidence.')
write(R/'config.json',config);log('loading')
TOKENIZER=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
model=None
captured={}
def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
results={}
def threshold(y,p):
 precision,recall,ts=precision_recall_curve(y,p);f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15)
 return float(ts[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])
def metrics(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return dict(n=len(y),positive=int(sum(y)),accuracy=float(accuracy_score(y,pred)),f1=float(f1_score(y,pred,zero_division=0)),auc=float(roc_auc_score(y,p)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
B=Path('/home/ubuntu/experiments/logistic-probes-20261001')
for ds in ['wildguard','high_stakes','toxicchat','beavertails','aegis','toxigen','halueval','ragtruth','fava','wildjailbreak']:
 start=time.time()
 rows=json.loads((R/f'{ds}_selected.json').read_text());n=len(rows);features=np.zeros((n,2560),dtype=np.float32);out=R/'test_activations'/ds;out.mkdir(parents=True,exist_ok=True);assert not list(out.glob('*.safetensors'))
 if model is None:
  model=AutoModel.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();assert model.config.hidden_size==2560;model.layers[24].register_forward_hook(hook)
 order=sorted(range(n),key=lambda i:rows[i]['n_tokens']);batches=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=4 or (len(chunk)+1)*rows[i]['n_tokens']>8192):batches.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:batches.append(chunk)
 torch.cuda.reset_peak_memory_stats();log('extracting',dataset=ds,total=n);extractstart=time.time()
 for batchno,idx in enumerate(batches):
  lens=[rows[i]['n_tokens'] for i in idx];width=max(lens);ids=torch.full((len(idx),width),TOKENIZER.pad_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
  for j,i in enumerate(idx):ids[j,:lens[j]]=torch.tensor(rows[i]['token_ids'],device='cuda');att[j,:lens[j]]=1
  with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
  h=captured.pop('h')
  for j,i in enumerate(idx):features[i]=h[j,rows[i]['mask']].float().mean(0).cpu().numpy()
  testjs=[j for j,i in enumerate(idx) if rows[i]['split']=='test']
  if testjs:
   testix=[idx[j] for j in testjs];testlens=[lens[j] for j in testjs];hcpu=torch.cat([h[j,:lens[j]] for j in testjs]).cpu().contiguous();offsets=torch.tensor([0]+np.cumsum(testlens).tolist(),dtype=torch.int64);path=out/f'{batchno:05d}.safetensors'
   save_file({'activations':hcpu,'offsets':offsets,'row_indices':torch.tensor(testix),'token_ids':torch.tensor([t for i in testix for t in rows[i]['token_ids']],dtype=torch.int64)},str(path))
   for j,i in enumerate(testix):rows[i].update(cache_file=str(path),cache_row=j)
   del hcpu
  del h,ids,att
  if batchno%100==0:log('extracting',dataset=ds,completed=sum(len(z) for z in batches[:batchno+1]),total=n)
 extraction=time.time()-extractstart;peak=torch.cuda.max_memory_allocated()/2**30;cache_gib=sum(z.stat().st_size for z in out.glob('*.safetensors'))/2**30
 np.save(R/f'{ds}_features.npy',features)
 (R/f'{ds}_metadata.json').write_text(json.dumps(rows))
 y=np.array([r['label'] for r in rows]);split=np.array([r['split'] for r in rows]);tr=split=='train';va=split=='val'
 scaler=StandardScaler().fit(features[tr]);x=scaler.transform(features);clf=LogisticRegression(C=.001 if ds=='high_stakes' else .01,class_weight='balanced',max_iter=2000,tol=1e-6,solver='lbfgs').fit(x[tr],y[tr]);p=clf.predict_proba(x)[:,1];t=threshold(y[va],p[va])
 scores={s:metrics(y[split==s],p[split==s],t) for s in ['train','val','test']}
 length=np.log1p(np.array([[(len(r['mask']) if 'mask' in r else r['selected_tokens']),r['n_tokens']] for r in rows]));ls=StandardScaler().fit(length[tr]);lc=LogisticRegression(C=.01,class_weight='balanced',max_iter=2000).fit(ls.transform(length[tr]),y[tr]);lp=lc.predict_proba(ls.transform(length))[:,1];lt=threshold(y[va],lp[va])
 result=dict(threshold=t,metrics=scores,length_only_test=metrics(y[split=='test'],lp[split=='test'],lt),extraction_seconds=extraction,peak_gpu_allocated_gib=peak,counts={s+':'+str(y):sum(r['split']==s and r['label']==y for r in rows) for s in ['train','val','test'] for y in [0,1]},mean_tokens=sum(r['n_tokens'] for r in rows)/len(rows),cache_gib=cache_gib,flagged_test=int(sum((split=='test') & (p>=t))),flagged_test_tokens=int(sum(r['n_tokens'] for i,r in enumerate(rows) if r['split']=='test' and p[i]>=t)))
 joblib.dump(dict(model=clf,scaler=scaler,threshold=t,config=dict(config,C=clf.C)),R/f'{ds}_probe.joblib');np.savez(R/f'{ds}_predictions.npz',labels=y,probabilities=p,splits=split,length_probabilities=lp)
 results[ds]=result;write(R/'partial_results.json',results);log('dataset_complete',dataset=ds,result=result)
write(R/'results.json',results);write(R/'COMPLETED',dict(datasets=len(results),examples=sum(sum(x['counts'].values()) for x in results.values()),seconds=time.time()-START));log('completed',datasets=len(results))
