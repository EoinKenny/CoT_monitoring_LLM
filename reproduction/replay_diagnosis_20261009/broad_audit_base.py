import os
os.environ['OMP_NUM_THREADS']='8';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,time,hashlib,contextlib
from pathlib import Path
import numpy as np,torch,joblib
from transformers import AutoModel,AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors import safe_open
S=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');R=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009/qwen');OUT=Path('/home/ubuntu/experiments/replay-diagnosis-20261009/sentence_check');OUT.mkdir(exist_ok=True,parents=True)
def read(p):return json.loads(p.read_text())
bb=read(R/'backbone.json');meta=read(R/'fava_metadata.json');original=read(S/'fava_metadata.json');X=np.load(R/'fava_features.npy',mmap_mode='r');oldX=np.load(S/'fava_features.npy',mmap_mode='r');fit=joblib.load(R/'fava_probe.joblib');clf,scaler=fit['model'],fit['scaler'];w=clf.coef_[0]/scaler.scale_;bias=float(clf.intercept_[0]-w@scaler.mean_)
order=sorted(range(len(original)),key=lambda i:original[i]['n_tokens']);chunks=[];chunk=[]
for i in order:
 if chunk and (len(chunk)>=4 or (len(chunk)+1)*original[i]['n_tokens']>8192):chunks.append(chunk);chunk=[]
 chunk.append(i)
if chunk:chunks.append(chunk)
chosen=[2339]+sorted([k for k,ix in enumerate(chunks) if any('cache_file' in original[i] for i in ix)],key=lambda k:abs(k-2339))[:3]
assert chunks[2339]==[2094,2109,2465,2784]
audit=dict(batch_indices={str(k):chunks[k] for k in chosen},rows={str(i):dict(split=meta[i]['split'],n_tokens=meta[i]['n_tokens'],ids_match=meta[i]['token_ids']==original[i]['token_ids'],mask_match=meta[i]['mask']==original[i]['mask'],features_match=bool(np.array_equal(X[i],oldX[i])),source_path=original[i].get('cache_file')) for k in chosen for i in chunks[k]},torch=torch.__version__,cuda=torch.version.cuda,backbone=bb,probe_threshold=fit['threshold'])
tok=AutoTokenizer.from_pretrained(bb['path'],local_files_only=True);torch.set_num_threads(8);torch.manual_seed(20261002)
model=AutoModel.from_pretrained(bb['path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();captured={};model.layers[24].register_forward_hook(lambda m,i,o:captured.update(h=(o[0] if isinstance(o,tuple) else o).detach()))
import pysbd
seg=pysbd.Segmenter(language='en',clean=False,char_span=True)
def top_sentence(row,scores):
 text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026',enable_thinking=False)
 enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==row['token_ids'];offs=np.array(enc['offset_mapping']);mask=np.zeros(len(scores),bool);mask[row['mask']]=True;out=[]
 for cs in row['content_spans']:
  assert text[cs['start']:cs['end']]==cs['text']
  for span in seg.segment(cs['text']):
   start=cs['start']+span.start;end=cs['start']+span.end;sm=mask&(offs[:,1]>start)&(offs[:,0]<end)
   if sm.any():out.append((float(scores[sm].mean()),cs['message_index'],start,end))
 out.sort(reverse=True);return dict(identity=list(out[0][1:]),score=out[0][0],margin=out[0][0]-out[1][0] if len(out)>1 else None)
results=[];reference={}
for mode in ['full_original_defaults','prefix_math']:
 if mode=='prefix_math':model.layers=torch.nn.ModuleList(list(model.layers[:25]));model.norm=torch.nn.Identity()
 torch.use_deterministic_algorithms(mode=='prefix_deterministic')
 ctx=sdpa_kernel(SDPBackend.MATH) if mode=='prefix_math' else sdpa_kernel(SDPBackend.FLASH_ATTENTION) if mode=='prefix_flash' else contextlib.nullcontext()
 with ctx:
  for k in chosen:
   ix=chunks[k];width=max(meta[i]['n_tokens'] for i in ix);ids=torch.full((len(ix),width),tok.pad_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
   for j,i in enumerate(ix):n=meta[i]['n_tokens'];ids[j,:n]=torch.tensor(meta[i]['token_ids'],device='cuda');att[j,:n]=1
   with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
   h=captured.pop('h');hh=h.float().cpu().numpy()
   for j,i in enumerate(ix):
    row=meta[i];a=hh[j,:row['n_tokens']];mask=row['mask'];pooled=h[j,mask].float().mean(0).cpu().numpy();expected=float(clf.decision_function(scaler.transform(X[i:i+1]))[0]);scores=a.astype(np.float64)@w+bias;actual=float(scores[mask].mean());record=dict(mode=mode,batch=k,index=i,original_logit=expected,replay_logit=actual,logit_error=actual-expected,original_probability=float(1/(1+np.exp(-expected))),replay_probability=float(1/(1+np.exp(-actual))),cosine=float(pooled@X[i]/(np.linalg.norm(pooled)*np.linalg.norm(X[i]))),relative_l2=float(np.linalg.norm(pooled-X[i])/np.linalg.norm(X[i])),bitwise_features=bool(np.array_equal(pooled,X[i])))
    if mode=='full_original_defaults':reference[i]=a.copy()
    else:record['bitwise_vs_full']=bool(np.array_equal(a,reference[i]));record['max_abs_vs_full']=float(np.abs(a-reference[i]).max())
    if 'cache_file' in row:
     with safe_open(row['cache_file'],framework='pt') as sf:
      offs=sf.get_tensor('offsets');z=row['cache_row'];aa,zz=int(offs[z]),int(offs[z+1]);old=sf.get_slice('activations')[aa:zz].float().numpy()
     record['cached_token_exact']=bool(np.array_equal(a,old));oldscore=old.astype(np.float64)@w+bias;record['old_sentence']=top_sentence(row,oldscore);record['new_sentence']=top_sentence(row,scores);record['same_top_sentence']=record['old_sentence']['identity']==record['new_sentence']['identity'];record['token_logit_correlation']=float(np.corrcoef(scores[mask],oldscore[mask])[0,1]);record['mean_abs_token_logit_diff']=float(np.abs(scores[mask]-oldscore[mask]).mean())
    results.append(record)
    if i==2465:print(json.dumps(record),flush=True)
   del h
 (OUT/'RESULTS.json').write_text(json.dumps(dict(audit=audit,results=results),indent=2))
print('COMPLETED',flush=True)
