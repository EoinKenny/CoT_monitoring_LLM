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
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
R=Path('/home/ubuntu/experiments/seven-dataset-pilot-20261002');START=time.time();lock=(R/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert (R/'PREPARED').exists();assert not (R/'results.json').exists()
def write(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def log(event,**kw):
 x=dict(event=event,seconds=time.time()-START,**kw);write(R/'status.json',x);print(json.dumps(x),flush=True)
def fail(typ,exc,tb):write(R/'FAILED.json',dict(error=str(exc),traceback=''.join(traceback.format_exception(typ,exc,tb))));sys.__excepthook__(typ,exc,tb)
sys.excepthook=fail
torch.set_num_threads(8);threadpool_limits(8);warnings.filterwarnings('error',category=ConvergenceWarning)
MODEL='/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
config=dict(model=MODEL,block_index=10,hidden_size=2048,dtype='bfloat16',seed=20261002,C=.01,class_weight='balanced',max_input_tokens=4096,overlength='exclude candidate; never truncate',pool='mean content tokens only; response tasks use assistant tokens conditioned on evidence/request',threshold='maximum validation F1; highest threshold breaks ties',split='source groups disjoint; exact duplicate/conflict audit; balanced undersampling within each split',notes='Tiny development pilot, not deployment-prevalence evaluation. No LLM judge invoked.')
write(R/'config.json',config);log('loading')
model=AutoModel.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();assert model.config.hidden_size==2048
captured={}
def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
handle=model.layers[10].register_forward_hook(hook)
results={};inventory=json.loads((R/'inventory.json').read_text())
def threshold(y,p):
 ts=np.unique(np.r_[0.,p,1.]);scores=np.array([f1_score(y,p>=t,zero_division=0) for t in ts]);return float(ts[np.flatnonzero(scores==scores.max())[-1]])
def metrics(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return dict(n=len(y),positive=int(sum(y)),accuracy=float(accuracy_score(y,pred)),f1=float(f1_score(y,pred,zero_division=0)),auc=float(roc_auc_score(y,p)),precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
for ds in inventory:
 start=time.time();rows=json.loads((R/f'{ds}_selected.json').read_text());n=len(rows);features=np.zeros((n,2048),dtype=np.float32);out=R/'activations'/ds;out.mkdir(parents=True,exist_ok=True);assert not list(out.glob('*.safetensors'))
 order=sorted(range(n),key=lambda i:rows[i]['n_tokens']);batches=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=8 or (len(chunk)+1)*rows[i]['n_tokens']>16384):batches.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:batches.append(chunk)
 torch.cuda.reset_peak_memory_stats();log('extracting',dataset=ds,total=n)
 for batchno,idx in enumerate(batches):
  lens=[rows[i]['n_tokens'] for i in idx];width=max(lens);ids=torch.full((len(idx),width),128009,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
  for j,i in enumerate(idx):ids[j,:lens[j]]=torch.tensor(rows[i]['token_ids'],device='cuda');att[j,:lens[j]]=1
  with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
  h=captured.pop('h')
  for j,i in enumerate(idx):features[i]=h[j,rows[i]['mask']].float().mean(0).cpu().numpy()
  hcpu=torch.cat([h[j,:ln] for j,ln in enumerate(lens)]).cpu().contiguous();offsets=torch.tensor([0]+np.cumsum(lens).tolist(),dtype=torch.int64)
  save_file({'activations':hcpu,'offsets':offsets,'row_indices':torch.tensor(idx),'token_ids':torch.tensor([t for i in idx for t in rows[i]['token_ids']],dtype=torch.int64)},str(out/f'{batchno:04d}.safetensors'))
  del h,hcpu,ids,att
  if batchno%20==0:log('extracting',dataset=ds,completed=sum(len(z) for z in batches[:batchno+1]),total=n)
 np.save(R/f'{ds}_features.npy',features);extraction=time.time()-start
 y=np.array([r['label'] for r in rows]);split=np.array([r['split'] for r in rows]);tr=split=='train';va=split=='val'
 scaler=StandardScaler().fit(features[tr]);x=scaler.transform(features);clf=LogisticRegression(C=.01,class_weight='balanced',max_iter=2000,tol=1e-6,solver='lbfgs').fit(x[tr],y[tr]);p=clf.predict_proba(x)[:,1];t=threshold(y[va],p[va])
 scores={s:metrics(y[split==s],p[split==s],t) for s in ['train','val','test']}
 length=np.log1p(np.array([[len(r['mask']),r['n_tokens']] for r in rows]));ls=StandardScaler().fit(length[tr]);lc=LogisticRegression(C=.01,class_weight='balanced',max_iter=2000).fit(ls.transform(length[tr]),y[tr]);lp=lc.predict_proba(ls.transform(length))[:,1];lt=threshold(y[va],lp[va])
 result=dict(threshold=t,metrics=scores,length_only_test=metrics(y[split=='test'],lp[split=='test'],lt),extraction_seconds=extraction,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,counts=inventory[ds]['counts'],mean_tokens=inventory[ds]['mean_tokens'],cache_gib=sum(z.stat().st_size for z in out.glob('*.safetensors'))/2**30)
 joblib.dump(dict(model=clf,scaler=scaler,threshold=t,config=config),R/f'{ds}_probe.joblib');np.savez(R/f'{ds}_predictions.npz',labels=y,probabilities=p,splits=split,length_probabilities=lp)
 results[ds]=result;write(R/'partial_results.json',results);log('dataset_complete',dataset=ds,result=result)
write(R/'results.json',results);write(R/'COMPLETED',dict(datasets=len(results),examples=sum(x['examples'] for x in inventory.values()),seconds=time.time()-START));log('completed',datasets=len(results))
