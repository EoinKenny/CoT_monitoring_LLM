"""Replay original batches to derive sentence saliency without full token caches."""
from runtime import *
import joblib,pysbd,fcntl
lock=(ROOT/'gpu-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);seg=pysbd.Segmenter(language='en',clean=False,char_span=True)
def render(msgs):return tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026')
def escaped(s,ascii,html):
 t=json.dumps(s,ensure_ascii=ascii)[1:-1]
 if html:
  for a,b in [('<','\\u003c'),('>','\\u003e'),('&','\\u0026'),("'",'\\u0027')]:t=t.replace(a,b)
 return t

def sentence_scores(record,score):
 msgs=record['messages'];text=render(msgs);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);assert enc['input_ids']==record['token_ids'];offs=np.array(enc['offset_mapping']);mask=np.isin(np.arange(len(score)),record['mask']);candidates=[]
 for mi,m in enumerate(msgs):
  if record['role']!='all' and m['role']!=record['role']:continue
  content=m['content'];trim=content.strip()
  if not trim:continue
  prefix=render(msgs[:mi+1]);assert text.startswith(prefix)
  if m['role'] in ('tool','ipython'):
   prev=render(msgs[:mi]) if mi else '';header='<|start_header_id|>ipython<|end_header_id|>\n\n';pos=prefix.find(header,len(prev));assert pos>=0;start=pos+len(header);body=prefix[start:-len('<|eot_id|>')];assert json.loads(body)==content
   mode=next(((asc,html) for asc in [True,False] for html in [True,False] if '"'+escaped(content,asc,html)+'"'==body),None);assert mode is not None
   def charpos(x):return start+1+len(escaped(content[:x],*mode))
   tosegment=content
  else:
   base=prefix.rfind(trim);assert base>=0;tosegment=trim
   def charpos(x):return base+x
  for span in seg.segment(tosegment):
   a,b=charpos(span.start),charpos(span.end);smask=mask&(offs[:,1]>a)&(offs[:,0]<b)
   if smask.any():candidates.append(dict(sentence=span.sent.strip(),message_index=mi,role=m['role'],mean_logit=float(score[smask].mean()),tokens=int(smask.sum())))
 assert candidates
 return max(candidates,key=lambda s:s['mean_logit'])

def main():
 selected=read(B/'selected_flags.json');model=load_model();completed=0
 for ds in DATASETS:
  subset={r['sample_id']:r for r in selected if r['dataset']==ds}
  if not subset:continue
  rows=read(B/f'{ds}_metadata.json');schedule=read(B/f'{ds}_BATCHES.json');x=np.load(B/f'{ds}_features.npy',mmap_mode='r');fitted=joblib.load(B/f'{ds}_probe.joblib');scaler,clf=fitted['scaler'],fitted['model'];w=clf.coef_[0]/scaler.scale_;bias=float(clf.intercept_[0]-w@scaler.mean_);probe_sha=sha(B/f'{ds}_probe.joblib')
  for bi,ix in enumerate(schedule):
   targets=[i for i in ix if rows[i]['immutable_id'] in subset]
   if not targets:continue
   reserve();dest=B/'saliency'/ds/f'{bi:06d}.json';commit=B/'feature_commits'/ds/f'{bi:06d}.json'
   if dest.exists():
    saved=read(dest);assert saved['probe_sha256']==probe_sha and saved['feature_commit_sha256']==sha(commit);assert [r['sample_id'] for r in saved['records']]==[rows[i]['immutable_id'] for i in targets];completed+=len(targets);continue
   records=[]
   def callback(r,h):
    if r['immutable_id'] not in subset:return
    ref=subset[r['immutable_id']];acts=h.float().cpu().numpy().astype(np.float64);score=acts@w+bias;expected=float(clf.decision_function(scaler.transform(x[ref['index']:ref['index']+1]))[0]);err=abs(float(score[r['mask']].mean())-expected);assert err<1e-4,(ds,r['immutable_id'],err)
    records.append(dict(sample_id=r['immutable_id'],dataset=ds,saliency=sentence_scores(r,score),pooling_error=err))
   features,stats=forward(model,[rows[i] for i in ix],callback)
   # Same model, exact inputs and original batch shape must replay bitwise on this host.
   assert np.array_equal(features,x[ix]),'Extraction replay differs; preserve and inspect before judging'
   write(dest,dict(probe_sha256=probe_sha,feature_commit_sha256=sha(commit),bitwise_replay=True,records=records,stats=stats));completed+=len(records)
   write(ROOT/'SALIENCY_STATUS.json',dict(dataset=ds,completed=completed,total=len(selected)))
  del x,rows
 assert completed==len(selected);write(ROOT/'SALIENCY_COMPLETED.json',dict(n=completed,time=time.time()))
if __name__=='__main__':main()
