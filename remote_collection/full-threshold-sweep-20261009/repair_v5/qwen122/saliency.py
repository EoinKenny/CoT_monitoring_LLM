"""Replay Qwen's original batches exactly and score their content sentences."""
from common import *
import sys,fcntl,numpy as np,joblib,pysbd,torch
sys.path.insert(0,'/home/ubuntu/qwen122b-fp8-smoke-20261007')
import collect as extraction
extraction.ROOT=ROOT
from transformers import AutoTokenizer
seg=pysbd.Segmenter(language='en',clean=False,char_span=True)

def render(tok,r):
 import copy
 messages=copy.deepcopy(r['messages'])
 if not any(m['role']=='user' for m in messages):messages.insert(1 if messages[0]['role']=='system' else 0,dict(role='user',content=''))
 return tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False,date_string='01 Oct 2026')

def sentence_scores(tok,r,scores):
 text=render(tok,r);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
 assert enc['input_ids']==r['token_ids'];offs=np.array(enc['offset_mapping']);mask=np.zeros(len(scores),bool);mask[r['mask']]=True;candidates=[]
 for content in r['content_spans']:
  start,end=content['start'],content['end'];body=text[start:end]
  for sentence in seg.segment(body):
   a,b=start+sentence.start,start+sentence.end;sm=mask&(offs[:,1]>a)&(offs[:,0]<b)
   if sm.any():candidates.append(dict(sentence=sentence.sent.strip(),message_index=content['message_index'],role=content['role'],mean_logit=float(scores[sm].mean()),tokens=int(sm.sum())))
 assert candidates;return max(candidates,key=lambda s:s['mean_logit'])

@torch.inference_mode()
def replay(model,tok,rows,targets,weights,bias,x,indices,clf,scaler):
 length=max(r['n_tokens'] for r in rows);ids=torch.full((len(rows),length),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
 for j,r in enumerate(rows):ids[j,:r['n_tokens']]=torch.tensor(r['token_ids'],device='cuda');mask[j,:r['n_tokens']]=1
 h=model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
 pooled=torch.stack([h[j,torch.tensor(r['mask'],device='cuda')].float().mean(0) for j,r in enumerate(rows)]).cpu().numpy()
 assert np.array_equal(pooled,x[indices]),'Original extraction batch no longer replays bitwise; halt for review'
 records=[]
 for j,r in enumerate(rows):
  if r['immutable_id'] not in targets:continue
  acts=h[j,:r['n_tokens']].float().cpu().numpy().astype(np.float64);scores=acts@weights+bias;expected=float(clf.decision_function(scaler.transform(x[indices[j]:indices[j]+1]))[0]);error=abs(float(scores[r['mask']].mean())-expected)
  assert error<1e-4,(r['immutable_id'],error)
  records.append(dict(sample_id=r['immutable_id'],dataset=targets[r['immutable_id']]['dataset'],saliency=sentence_scores(tok,r,scores),pooling_error=error))
 return records

def main():
 lock=(ROOT/'gpu-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 extraction.setup();plan=read(extraction.SMOKE/'DOWNLOAD_PLAN.json');tok=AutoTokenizer.from_pretrained(plan['model_path'],local_files_only=True);model=extraction.load_model(plan)
 selected=read(B/'selected_flags.json');completed=0
 for ds in DATASETS:
  targets={r['sample_id']:r for r in selected if r['dataset']==ds}
  if not targets:continue
  rows=read(B/f'{ds}_metadata.json');schedule=read(B/f'{ds}_BATCHES.json');x=np.load(B/f'{ds}_features.npy',mmap_mode='r');fitted=joblib.load(B/f'{ds}_probe.joblib');scaler,clf=fitted['scaler'],fitted['model'];w=clf.coef_[0]/scaler.scale_;bias=float(clf.intercept_[0]-w@scaler.mean_);probe_sha=sha(B/f'{ds}_probe.joblib');feature_sha=sha(B/f'{ds}_EXTRACTED.json')
  for bi,ix in enumerate(schedule):
   wanted=[rows[i]['immutable_id'] for i in ix if rows[i]['immutable_id'] in targets]
   if not wanted:continue
   reserve();dest=B/'saliency'/ds/f'{bi:06d}.json'
   if dest.exists():
    d=read(dest);assert d['probe_sha256']==probe_sha and d['feature_commit_sha256']==feature_sha and [r['sample_id'] for r in d['records']]==wanted;completed+=len(wanted);continue
   records=replay(model,tok,[rows[i] for i in ix],targets,w,bias,x,ix,clf,scaler)
   write(dest,dict(probe_sha256=probe_sha,feature_commit_sha256=feature_sha,bitwise_replay=True,records=records));completed+=len(records)
   write(ROOT/'SALIENCY_STATUS.json',dict(dataset=ds,completed=completed,total=len(selected),time=time.time()))
  del rows,x
 assert completed==len(selected);write(ROOT/'SALIENCY_COMPLETED.json',dict(n=completed,time=time.time()))
if __name__=='__main__':main()
