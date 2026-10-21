"""Replay original extraction batches for validation rows lacking token caches.
Persist only token logits, never duplicate large activation tensors.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='8'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import sys,json,hashlib,time
from pathlib import Path
import numpy as np,torch,joblib
from transformers import AutoModel,AutoTokenizer
ROOT=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009');b=sys.argv[1];R=ROOT/b
OLD=Path('/home/ubuntu/experiments')/{'llama':'ten-dataset-full-20261002','qwen':'qwen4b-backbone-20261002','gemma':'gemma31b-backbone-20261003'}[b]
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(id):return hashlib.sha256(id.encode()).hexdigest()
torch.manual_seed(20261002);np.random.seed(20261002);torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
bb=read(R/'backbone.json');tok=AutoTokenizer.from_pretrained(bb['path'],local_files_only=True);selected=read(R/'selected_flags.json');model=None;captured={};count=0
def hook(m,i,o):captured['h']=(o[0] if isinstance(o,tuple) else o).detach()
for ds in sorted({r['dataset'] for r in selected}):
 meta=read(R/f'{ds}_metadata.json');targets={r['index']:r for r in selected if r['dataset']==ds and 'cache_file' not in meta[r['index']]}
 if not targets:continue
 original=read(OLD/f'{ds}_metadata.json');assert all(i<len(original) and meta[i]['token_ids']==original[i]['token_ids'] and meta[i]['messages']==original[i]['messages'] for i in targets)
 fit=joblib.load(R/f'{ds}_probe.joblib');clf,scaler=fit['model'],fit['scaler'];w=clf.coef_[0]/scaler.scale_;bias=float(clf.intercept_[0]-w@scaler.mean_);X=np.load(R/f'{ds}_features.npy',mmap_mode='r')
 order=sorted(range(len(original)),key=lambda i:original[i]['n_tokens']);bs,cap={'llama':(8,16384),'qwen':(4,8192),'gemma':(4,2048)}[b];chunks=[];chunk=[]
 for i in order:
  if chunk and (len(chunk)>=bs or (len(chunk)+1)*original[i]['n_tokens']>cap):chunks.append(chunk);chunk=[]
  chunk.append(i)
 if chunk:chunks.append(chunk)
 for batch,ix in enumerate(chunks):
  wanted=[i for i in ix if i in targets];paths={i:R/'replayed_scores'/ds/(key(meta[i]['immutable_id'])+'.npz') for i in wanted}
  if not wanted:continue
  for i,p in paths.items():
   if p.with_suffix('.json').exists():assert sha(p)==read(p.with_suffix('.json'))['sha256']
  pending=[i for i in wanted if not paths[i].with_suffix('.json').exists()]
  if not pending:continue
  if model is None:
   if b=='gemma':
    sys.path.insert(0,str(R));from backbone_loader import load_model
    model=load_model()
   else:
    model=AutoModel.from_pretrained(bb['path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa');model.layers=torch.nn.ModuleList(list(model.layers[:bb['block_index']+1]));model.norm=torch.nn.Identity();model=model.to('cuda').eval()
   model.layers[bb['block_index']].register_forward_hook(hook)
  width=max(original[i]['n_tokens'] for i in ix);pad=tok.pad_token_id if tok.pad_token_id is not None else 128009;ids=torch.full((len(ix),width),pad,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
  for j,i in enumerate(ix):n=original[i]['n_tokens'];ids[j,:n]=torch.tensor(original[i]['token_ids'],device='cuda');att[j,:n]=1
  with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
  h=captured.pop('h');attention_backend='default'
  for j,i in enumerate(ix):
   if i not in pending:continue
   row=meta[i];a=h[j,:row['n_tokens']].float().cpu().numpy();pooled=a[row['mask']].mean(0);score=a.astype(np.float64)@w+bias;expected=float(clf.decision_function(scaler.transform(X[i:i+1]))[0]);error=abs(float(score[row['mask']].mean())-expected);cos=float(pooled@X[i]/(np.linalg.norm(pooled)*np.linalg.norm(X[i])))
   original_p=float(1/(1+np.exp(-np.clip(expected,-700,700))));replayed_p=float(1/(1+np.exp(-np.clip(float(score[row['mask']].mean()),-700,700))));prob_error=abs(original_p-replayed_p)
   saturation_exception=bool(prob_error<=1e-4 and (min(original_p,replayed_p)>=.999 or max(original_p,replayed_p)<=.001))
   proof=dict(sample_id=row['immutable_id'],original_batch=batch,original_indices=ix,pooling_error=error,cosine=cos,bitwise_equal=bool(np.array_equal(pooled,X[i])),probe_sha256=sha(R/f'{ds}_probe.joblib'),original_probability=original_p,replayed_probability=replayed_p,probability_error=prob_error,saturation_exception=saturation_exception)
   path=paths[i];path.parent.mkdir(parents=True,exist_ok=True)
   # Cross-architecture BF16 reproduction is numerical, not exact artifact replay.
   # Routing uses original frozen scores; this replay supplies sentence ranking only.
   proof['attention_backend']=attention_backend
   proof['agreement_policy']='diagnostic_only_user_authorized_20261009'
   assert np.isfinite(a).all() and np.isfinite(score).all() and np.isfinite(pooled).all() and np.isfinite(error), 'Non-finite replay values'
   assert not path.exists(),'Uncommitted replay scores require inspection'
   with path.with_suffix('.tmp').open('wb') as f:np.savez(f,ids=np.array(row['token_ids']),score=score)
   path.with_suffix('.tmp').replace(path);proof['sha256']=sha(path);write(path.with_suffix('.json'),proof);count+=1
  del h
  write(ROOT/'REPLAY_STATUS.json',dict(backbone=b,dataset=ds,completed_this_session=count,time=time.time()))
 print(json.dumps(dict(dataset=ds,replayed=count)),flush=True)
write(R/'REPLAY_COMPLETED.json',dict(n=count,time=time.time()))
