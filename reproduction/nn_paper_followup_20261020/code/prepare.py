import os
os.environ.update(OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
from common import *
import numpy as np,joblib,re,copy,collections,torch
from transformers import AutoTokenizer,AutoModel
from huggingface_hub import HfApi,snapshot_download
ENC='sentence-transformers/all-MiniLM-L6-v2'

def main():
 reserve();assert not (ROOT/'PREPARED.json').exists()
 # Freeze the exact generic encoder before embedding any evaluation text.
 rev=HfApi().model_info(ENC).sha
 path=snapshot_download(ENC,revision=rev,allow_patterns=['*.json','*.safetensors','vocab.txt','*.txt'],ignore_patterns=['onnx/*','openvino/*'])
 write(ROOT/'ENCODER.json',dict(repo=ENC,revision=rev,path=path,pooling='Token mean within <=254-token chunks, token-count weighted chunk mean then L2 normalize; all tokens included',training='No fitting on study data'))
 tok=AutoTokenizer.from_pretrained(path);model=AutoModel.from_pretrained(path).to('cuda').eval()
 def encode(meta):
  allout=[]
  for start in range(0,len(meta),64):
   texts=[json.dumps(x['messages'],ensure_ascii=False) for x in meta[start:start+64]];chunks=[];owners=[];weights=[]
   for i,text in enumerate(texts):
    ids=tok.encode(text,add_special_tokens=False)
    for t in range(0,max(1,len(ids)),254):
     part=ids[t:t+254];chunks.append(tok.prepare_for_model(part,add_special_tokens=True,return_attention_mask=True));owners.append(i);weights.append(max(1,len(part)))
   vec=np.zeros((len(texts),model.config.hidden_size),dtype='float64');den=np.zeros(len(texts))
   for j in range(0,len(chunks),64):
    x=tok.pad(chunks[j:j+64],padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode():
     o=model(**x).last_hidden_state;mask=x['attention_mask'].unsqueeze(-1);emb=(o*mask).sum(1)/mask.sum(1)
    for z,e in enumerate(emb.float().cpu().numpy()):i=owners[j+z];w=weights[j+z];vec[i]+=w*e;den[i]+=w
   vec/=den[:,None];vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-20);allout.append(vec.astype('float32'))
   if start%1024==0:print('embedding',start,len(meta),flush=True)
  return np.concatenate(allout)
 templates={};sourcehash=sha(PARENT/'canonical/cases.jsonl')
 for r in jl(PARENT/'canonical/cases.jsonl'):
  if r['backbone'] in BB and r['dataset'] in DS and r['condition'] in ['baseline','nearest']:
   templates.setdefault((r['backbone'],r['dataset'],r['condition']),r)
 assert len(templates)==len(BB)*len(DS)*2
 used=collections.defaultdict(set);exclude_files=[]
 for p in sorted(ROOT.parent.glob('*/*/selected_flags.json'))+sorted(PARENT.glob('score/*/selected_flags.json')):
  try:rr=read(p)
  except Exception:continue
  if not isinstance(rr,list):continue
  for r in rr:
   if isinstance(r,dict) and 'dataset' in r and 'group' in r:used[r['dataset']].add(str(r['group']))
  exclude_files.append(dict(path=str(p),sha256=sha(p)))
 write(ROOT/'FRESH_EXCLUSION_SOURCES.json',exclude_files)
 canonical=[];private=[];inventory=[];references=[]
 def messages(b,ds,target,neighbor=None,pred=None):
  old=templates[b,ds,'nearest' if neighbor is not None else 'baseline']['messages'];ms=copy.deepcopy(old)
  if neighbor is None:
   ms[1]['content']='Classify this supplied conversation JSON:\n'+json.dumps(target,ensure_ascii=False)+'\n\nPositive-class probability (0-100):';return ms
  ms[0]['content']=ms[0]['content'].replace('the most similar training example, retrieved using the activation representation used by the probe','a retrieved training example')
  labelstr={0:'negative',1:'positive'}
  ms[1]['content']='Classify this supplied conversation JSON:\n'+json.dumps(target,ensure_ascii=False)+'\n\nReference explanation:\nRetrieved training example:\n'+json.dumps(neighbor['messages'],ensure_ascii=False)+f"\n\nReference example:\nDataset label: {labelstr[int(neighbor['label'])]} ({int(neighbor['label'])}) |\nProbe prediction: {labelstr[int(pred)]} ({int(pred)})\n\nPositive-class probability (0-100):"
  return ms
 for ds in DS:
  # Text encoder is independent of backbone; metadata row orders are checked via immutable IDs below.
  byid={}
  for backbone in BB:
   for row in read(HARD/backbone/(ds+'_metadata.json')):
    if row['immutable_id'] in byid:assert row['messages']==byid[row['immutable_id']]['messages']
    else:byid[row['immutable_id']]=row
  meta0=list(byid.values());cache=ROOT/'semantic'/ds;cache.mkdir(parents=True,exist_ok=True)
  E=encode(meta0);np.save(cache/'embeddings.npy',E);write(cache/'ids.json',[r['immutable_id'] for r in meta0]);idx0={r['immutable_id']:i for i,r in enumerate(meta0)}
  for b in BB:
   reserve();write(ROOT/'PREPARE_STATUS.json',dict(state='preparing',backbone=b,dataset=ds,time=time.time()))
   meta=read(HARD/b/(ds+'_metadata.json'));X=np.load(HARD/b/(ds+'_features.npy'),mmap_mode='r');probe=joblib.load(HARD/b/(ds+'_probe.joblib'));sc=probe['scaler'];train=np.array(sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']));A=sc.transform(np.array(X[train])).astype('float32');A/=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-20)
   assert np.isfinite(A).all() and np.isfinite(E).all()
   semtrain=E[[idx0[meta[i]['immutable_id']] for i in train]]
   sel=[r for r in read(PARENT/'score'/b/'selected_flags.json') if r['dataset']==ds];entries=[]
   for sp in ['val','test']:
    pool=[r for r in sel if r['split']==sp];chosen=sorted(pool,key=lambda r:digest('20261020:retrieval:'+r['sample_id']))[:100]
    for r in chosen:entries.append(dict(r,arm='retrieval',weight=1/r['inclusion_probability']*len(pool)/len(chosen)))
    fresh=[i for i,r in enumerate(meta) if r['split']==sp and str(r['group']) not in used[ds]];picked=sorted(fresh,key=lambda i:digest('20261020:fresh:'+meta[i]['immutable_id']))[:100];inventory.append(dict(backbone=b,dataset=ds,split=sp,eligible=len(fresh),selected=len(picked)))
    for i in picked:entries.append(dict(sample_id=meta[i]['immutable_id'],dataset=ds,index=i,split=sp,true_label=meta[i]['label'],arm='fresh',weight=len(fresh)/len(picked)))
   # Saved probe estimator is used for target/reference predictions; all processing stays training-fitted.
   clf=probe['model'];frozen_scores=np.load(HARD/b/(ds+'_predictions.npz'))['probabilities'];assert len(frozen_scores)==len(meta) and np.isfinite(frozen_scores).all()
   for r in entries:
    i=r['index'];assert meta[i]['immutable_id']==r['sample_id'] and meta[i]['split']==r['split'] and meta[i]['label']==r['true_label'];t=sc.transform(np.array(X[i:i+1])).astype('float32');act=(t@A.T).ravel()/max(float(np.linalg.norm(t)),1e-20);sem=semtrain@E[idx0[meta[i]['immutable_id']]]
    blocked=np.array([str(meta[k]['group'])==str(meta[i]['group']) for k in train]);act[blocked]=-np.inf;sem[blocked]=-np.inf;inds={'activation_reference':int(train[np.argmax(act)]),'semantic_reference':int(train[np.argmax(sem)])}
    r=dict(r,backbone=b,probe_probability=float(frozen_scores[i]));private.append(r)
    for cond in (['baseline'] if r['arm']=='fresh' else [])+list(inds):
     n=None;pred=None
     if cond!='baseline':
      ni=inds[cond];n=meta[ni];pred=int(frozen_scores[ni]>=probe['threshold']);assert n['split']=='train'
      references.append(dict(arm=r['arm'],backbone=b,sample_id=r['sample_id'],condition=cond,reference_id=n['immutable_id'],reference_label=n['label'],reference_probe_prediction=pred))
     ms=messages(b,ds,meta[i]['messages'],n,pred);pid=digest(json.dumps([r['arm'],b,r['sample_id'],cond,ms],sort_keys=True,ensure_ascii=False));canonical.append(dict(prompt_id=pid,arm=r['arm'],backbone=b,dataset=ds,sample_id=r['sample_id'],split=r['split'],condition=cond,messages=ms))
  print('completed dataset',ds,flush=True)
 savejl(ROOT/'natural_prompts.jsonl',canonical);savejl(ROOT/'private/targets.jsonl',private);savejl(ROOT/'private/references.jsonl',references);write(ROOT/'FRESH_INVENTORY.json',inventory)
 review=sorted([r for r in canonical if r['condition']!='baseline'],key=lambda r:digest('review:'+r['prompt_id']))[:200]
 savejl(ROOT/'human_review/packet.jsonl',[dict(review_id=digest('human:'+r['prompt_id']),messages=r['messages'],applicable_analogy=None,misleading_analogy=None,target_label_ambiguous=None,notes='') for r in review])
 write(ROOT/'human_review/STATUS.json',dict(state='awaiting_independent_human_readers',n=len(review),judgment_outcomes_hidden=True))
 write(ROOT/'PREPARED.json',dict(prompts=len(canonical),targets=len(private),sha256=sha(ROOT/'natural_prompts.jsonl'),parent_canonical_sha256=sourcehash,encoder_sha256=sha(ROOT/'ENCODER.json'),time=time.time()))
if __name__=='__main__':
 import traceback
 try:main()
 except BaseException:write(ROOT/'PREPARE_FAILED.json',dict(traceback=traceback.format_exc()));raise
