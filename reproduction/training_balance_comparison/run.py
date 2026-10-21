import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='4'
os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import json,time,hashlib,fcntl,warnings
import numpy as np,torch,joblib
from transformers import AutoModel,AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve,confusion_matrix,f1_score,roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
R=Path('/home/ubuntu/experiments/training-balance-comparison-20261004');B=R.parent/'ten-dataset-full-20261002'
lock=(R/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'RESULTS.json').exists();start=time.time();threadpool_limits(4);torch.set_num_threads(4)
warnings.filterwarnings('error',category=ConvergenceWarning)
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def status(event,**kw):
 x=dict(event=event,seconds=time.time()-start,**kw);write(R/'status.json',x);print(json.dumps(x),flush=True)
rows=json.loads((R/'fava_selected.json').read_text());old=json.loads((B/'fava_metadata.json').read_text());oldx=np.load(B/'fava_features.npy',mmap_mode='r')
for split in ['val','test']:
 assert [(x['id'],x['label'],x['group'],x['input_hash']) for x in rows if x['split']==split]==[(x['id'],x['label'],x['group'],x['input_hash']) for x in old if x['split']==split]
lookup={x['id']:i for i,x in enumerate(old)};missing=[i for i,x in enumerate(rows) if x['id'] not in lookup]
assert all(rows[i]['split']=='train' for i in missing)
x=np.empty((len(rows),2048),dtype=np.float32)
for i,r in enumerate(rows):
 if r['id'] in lookup:
  j=lookup[r['id']];assert (r['label'],r['split'],r['input_hash'])==(old[j]['label'],old[j]['split'],old[j]['input_hash']);x[i]=oldx[j]
MODEL='/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6'
write(R/'PROTOCOL.json',dict(dataset='FAVA',backbone=MODEL,block_index=10,comparison=['original balanced undersample + balanced class weights','all eligible training + balanced class weights'],C=.01,threshold='Maximum validation positive-class F1; highest-threshold tie break',selection='User-requested winner by test F1, exploratory and not an independent confirmatory test',train_counts={str(y):sum(r['split']=='train' and r['label']==y for r in rows) for y in [0,1]},reused_vectors=len(rows)-len(missing),new_training_vectors=len(missing),original_metadata_sha256=hashlib.sha256((B/'fava_metadata.json').read_bytes()).hexdigest()))
torch.manual_seed(20261002);torch.cuda.set_per_process_memory_fraction(.065)
status('loading_backbone',missing=len(missing))
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
model=AutoModel.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa')
model.layers=torch.nn.ModuleList(list(model.layers[:11]));model.norm=torch.nn.Identity();model=model.to('cuda').eval()
# Read the pre-final-normalization output of block 11, as in the original probe.
order=sorted(missing,key=lambda i:rows[i]['n_tokens']);batches=[];chunk=[]
for i in order:
 if chunk and (len(chunk)>=4 or (len(chunk)+1)*rows[i]['n_tokens']>2048):batches.append(chunk);chunk=[]
 chunk.append(i)
if chunk:batches.append(chunk)
for k,idx in enumerate(batches):
 width=max(rows[i]['n_tokens'] for i in idx);ids=torch.full((len(idx),width),128009,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
 for j,i in enumerate(idx):n=rows[i]['n_tokens'];ids[j,:n]=torch.tensor(rows[i]['token_ids'],device='cuda');mask[j,:n]=1
 with torch.inference_mode():h=model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
 assert torch.isfinite(h).all()
 for j,i in enumerate(idx):x[i]=h[j,rows[i]['mask']].float().mean(0).cpu().numpy()
 del h,ids,mask
 if k%100==0:status('extracting_added_training_examples',completed=sum(len(z) for z in batches[:k+1]),total=len(missing))
del model;torch.cuda.empty_cache();np.save(R/'fava_features.npy',x)
y=np.array([r['label'] for r in rows]);splits=np.array([r['split'] for r in rows]);val=splits=='val';test=splits=='test'
oldtrain={r['id'] for r in old if r['split']=='train'}
def threshold(y,p):
 precision,recall,t=precision_recall_curve(y,p);f=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-15);return float(t[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))[-1]])
def metrics(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();return dict(f1=float(f1_score(y,pred)),auc=float(roc_auc_score(y,p)),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn))
results={}
for name,tr in [('undersampled_weighted',np.array([r['id'] in oldtrain for r in rows])),('full_weighted',splits=='train')]:
 scaler=StandardScaler().fit(x[tr]);z=scaler.transform(x);clf=LogisticRegression(C=.01,class_weight='balanced',max_iter=2000,tol=1e-6,solver='lbfgs').fit(z[tr],y[tr]);p=clf.predict_proba(z)[:,1];t=threshold(y[val],p[val]);results[name]=dict(train_counts={str(c):int(sum(y[tr]==c)) for c in [0,1]},threshold=t,val=metrics(y[val],p[val],t),test=metrics(y[test],p[test],t))
 joblib.dump(dict(model=clf,scaler=scaler,threshold=t),R/(name+'.joblib'));np.save(R/(name+'_probabilities.npy'),p)
baseline=json.loads((B/'results.json').read_text())['fava']['metrics']['test']
assert all(results['undersampled_weighted']['test'][k]==baseline[k] for k in ['tp','fp','fn','tn']),'Balanced baseline failed exact confusion-matrix reproduction'
winner=max(results,key=lambda k:results[k]['test']['f1'])
out=dict(results=results,winner_by_test_f1=winner,all_positive_test_f1=float(f1_score(y[test],np.ones(sum(test)))),scope='One dataset, one backbone, fixed split and regularisation. Selecting by test F1 is exploratory; no claim of general superiority across backbones/datasets.',elapsed_seconds=time.time()-start)
write(R/'RESULTS.json',out);write(R/'COMPLETED',dict(seconds=time.time()-start));status('completed',result=out)
