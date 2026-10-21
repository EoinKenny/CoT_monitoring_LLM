import os
os.environ['OMP_NUM_THREADS']='8';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,hashlib,time
from pathlib import Path
import numpy as np,torch,joblib,pysbd
from transformers import AutoModel,AutoTokenizer
from safetensors import safe_open
R=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009/qwen');S=Path('/home/ubuntu/experiments/qwen4b-backbone-20261002');O=Path('/home/ubuntu/experiments/replay-diagnosis-20261009/broad');O.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text());bb=read(R/'backbone.json');tok=AutoTokenizer.from_pretrained(bb['path'],local_files_only=True);torch.set_num_threads(8);torch.manual_seed(20261002);torch.use_deterministic_algorithms(True)
model=AutoModel.from_pretrained(bb['path'],local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa');model.layers=torch.nn.ModuleList(list(model.layers[:25]));model.norm=torch.nn.Identity();model=model.to('cuda').eval();cap={};model.layers[24].register_forward_hook(lambda m,i,o:cap.update(h=(o[0] if isinstance(o,tuple) else o).detach()));seg=pysbd.Segmenter(language='en',clean=False,char_span=True)
def top(row,scores):
 text=tok.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026',enable_thinking=False);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==row['token_ids'];offs=np.array(enc['offset_mapping']);mask=np.zeros(len(scores),bool);mask[row['mask']]=True;out=[]
 for c in row['content_spans']:
  for span in seg.segment(c['text']):
   a=c['start']+span.start;z=c['start']+span.end;sm=mask&(offs[:,1]>a)&(offs[:,0]<z)
   if sm.any():out.append((float(scores[sm].mean()),c['message_index'],a,z))
 out.sort(reverse=True);return list(out[0][1:])
allrows=[]
for ds in sorted(read(R/'results.json')):
 rows=read(S/f'{ds}_metadata.json');fit=joblib.load(R/f'{ds}_probe.joblib');clf,scaler=fit['model'],fit['scaler'];w=clf.coef_[0]/scaler.scale_;bias=float(clf.intercept_[0]-w@scaler.mean_)
 candidates=[i for i,r in enumerate(rows) if 'cache_file' in r];chosen=set(sorted(candidates,key=lambda i:hashlib.sha256(f'20261009:{ds}:{rows[i]["input_hash"]}'.encode()).hexdigest())[:10]);order=sorted(range(len(rows)),key=lambda i:rows[i]['n_tokens']);batches=[];batch=[]
 for i in order:
  if batch and (len(batch)>=4 or (len(batch)+1)*rows[i]['n_tokens']>8192):batches.append(batch);batch=[]
  batch.append(i)
 if batch:batches.append(batch)
 for bi,ix in enumerate(batches):
  if not chosen.intersection(ix):continue
  width=max(rows[i]['n_tokens'] for i in ix);ids=torch.full((len(ix),width),tok.pad_token_id,dtype=torch.long,device='cuda');att=torch.zeros_like(ids)
  for j,i in enumerate(ix):n=rows[i]['n_tokens'];ids[j,:n]=torch.tensor(rows[i]['token_ids'],device='cuda');att[j,:n]=1
  with torch.inference_mode():model(input_ids=ids,attention_mask=att,use_cache=False)
  h=cap.pop('h')
  for j,i in enumerate(ix):
   if i not in chosen:continue
   r=rows[i];a=h[j,:r['n_tokens']].float().cpu().numpy()
   with safe_open(r['cache_file'],framework='pt') as f:
    off=f.get_tensor('offsets');k=r['cache_row'];aa,zz=int(off[k]),int(off[k+1]);old=f.get_slice('activations')[aa:zz].float().numpy();assert f.get_slice('token_ids')[aa:zz].numpy().tolist()==r['token_ids']
   mask=r['mask'];s=a.astype(np.float64)@w+bias;old_s=old.astype(np.float64)@w+bias;mu=a[mask].mean(0);old_mu=old[mask].mean(0);l=float(s[mask].mean());old_l=float(old_s[mask].mean());p=float(1/(1+np.exp(-l)));old_p=float(1/(1+np.exp(-old_l)))
   allrows.append(dict(dataset=ds,index=i,batch=bi,probability_difference=abs(p-old_p),logit_difference=abs(l-old_l),cosine=float(mu@old_mu/(np.linalg.norm(mu)*np.linalg.norm(old_mu))),same_sentence=top(r,s)==top(r,old_s),label_changed=(p>=fit['threshold'])!=(old_p>=fit['threshold'])))
  del h
 (O/'ROWS.json').write_text(json.dumps(allrows,indent=2));print(ds,len(allrows),flush=True)
summary=dict(n=len(allrows),same_sentence=sum(r['same_sentence'] for r in allrows),max_probability_difference=max(r['probability_difference'] for r in allrows),min_cosine=min(r['cosine'] for r in allrows),label_changes=sum(r['label_changed'] for r in allrows),rows_sha256=hashlib.sha256((O/'ROWS.json').read_bytes()).hexdigest())
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
