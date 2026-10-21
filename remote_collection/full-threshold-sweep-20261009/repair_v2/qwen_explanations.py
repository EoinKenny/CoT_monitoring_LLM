import os
os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='8'
import ast,json,time,hashlib,traceback,sys
from pathlib import Path
import numpy as np,joblib,torch,pysbd
from safetensors import safe_open
from transformers import AutoTokenizer
from threadpoolctl import threadpool_limits
threadpool_limits(8);torch.set_num_threads(8)
B=Path('/home/ubuntu/experiments/full-threshold-sweep-20261009/qwen');R=B/'judge_inputs';R.mkdir(exist_ok=True)
assert (B/'COMPLETED').exists();completed_rows=[json.loads(line) for line in (R/'explanations.jsonl').open()] if (R/'explanations.jsonl').exists() else []; done_ids={r['sample_id'] for r in completed_rows}; assert len(done_ids)==len(completed_rows)
tokenizer=AutoTokenizer.from_pretrained(json.loads((B/'backbone.json').read_text())['path'],local_files_only=True)
results=json.loads((B/'results.json').read_text());rows=json.loads((B/'selected_flags.json').read_text())
(R/'private.json').write_text(json.dumps(rows))
(R/'weighted_results.json').write_text(json.dumps({'results':{ds:{'test':v['metrics']['test']} for ds,v in results.items()}}))
seg=pysbd.Segmenter(language='en',clean=False,char_span=True)
count=len(completed_rows);started=time.time();maxerr=max([r['pooling_error'] for r in completed_rows],default=0)

def render(msgs):return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026',enable_thinking=False)
def escaped(s,ascii,html):
 t=json.dumps(s,ensure_ascii=ascii)[1:-1]
 if html:
  for a,b in [('<','\\u003c'),('>','\\u003e'),('&','\\u0026'),("'",'\\u0027')]:t=t.replace(a,b)
 return t
for name in sorted(results):
 meta=json.loads((B/f'{name}_metadata.json').read_text());X=np.load(B/f'{name}_features.npy',mmap_mode='r');fitted=joblib.load(B/f'{name}_probe.joblib');scaler,model=fitted['scaler'],fitted['model'];threshold=float(results[name]['threshold'])
 trainix=np.array(sorted([i for i,r in enumerate(meta) if r['split']=='train'],key=lambda i:meta[i]['immutable_id']));subset=[r for r in rows if r['dataset']==name]
 T=scaler.transform(X[trainix]).astype(np.float64);T/=np.maximum(np.linalg.norm(T,axis=1)[:,None],1e-30)
 nn={}
 for a in range(0,len(subset),64):
  group=subset[a:a+64];ix=[r['index'] for r in group];Z=scaler.transform(X[ix]).astype(np.float64);Z/=np.maximum(np.linalg.norm(Z,axis=1)[:,None],1e-30);best=np.zeros(len(group),dtype=int);bestscore=np.full(len(group),-np.inf)
  for ts in range(0,len(T),8192):
   sim=Z@T[ts:ts+8192].T;local=sim.argmax(axis=1);values=sim[np.arange(len(group)),local];better=values>bestscore;best[better]=ts+local[better];bestscore[better]=values[better]
  for k,r in enumerate(group):nn[r['sample_id']]=(int(trainix[best[k]]),float(bestscore[k]))
 w=model.coef_[0]/scaler.scale_;bias=float(model.intercept_[0]-w@scaler.mean_)
 for r in subset:
  if r['sample_id'] in done_ids:continue
  ix=r['index'];record=meta[ix];msgs=record['messages'];text=render(msgs)
  replay=B/'replayed_scores'/name/(hashlib.sha256(r['sample_id'].encode()).hexdigest()+'.npz')
  if replay.exists():
   proof=json.loads(replay.with_suffix('.json').read_text());assert hashlib.sha256(replay.read_bytes()).hexdigest()==proof['sha256'];z=np.load(replay);ids=z['ids'];score=z['score'];mask=np.isin(np.arange(len(ids)),record['mask']);expected=float(model.decision_function(scaler.transform(X[ix:ix+1]))[0]);err=abs(float(score[mask].mean())-expected);assert err<1e-4;maxerr=max(maxerr,err)
  else:
   with safe_open(str(Path(record['cache_file'])),framework='pt') as sf:
    offsets=sf.get_tensor('offsets');row=record['cache_row'];a,b=int(offsets[row]),int(offsets[row+1]);ids=sf.get_slice('token_ids')[a:b].numpy();acts=sf.get_slice('activations')[a:b].float().numpy()
   mask=np.isin(np.arange(len(ids)),record['mask']);score=acts.astype(np.float64)@w+bias;expected=float(model.decision_function(scaler.transform(X[ix:ix+1]))[0]);err=abs(float(score[mask].mean())-expected);assert err<1e-4,(name,ix,err);maxerr=max(maxerr,err)
  offs=np.array(tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']);candidates=[]
  for content_span in record['content_spans']:
   mi=content_span['message_index'];m=msgs[mi];tosegment=content_span['text'];base=content_span['start']
   assert text[base:content_span['end']]==tosegment
   def charpos(x):return base+x
   spans=seg.segment(tosegment)
   for span in spans:
    a,b=charpos(span.start),charpos(span.end);smask=mask&(offs[:,1]>a)&(offs[:,0]<b)
    if smask.any():candidates.append(dict(sentence=span.sent.strip(),message_index=mi,role=m['role'],mean_logit=float(score[smask].mean()),tokens=int(smask.sum())))
  assert candidates,(name,ix)
  sal=max(candidates,key=lambda s:s['mean_logit']);ni,cos=nn[r['sample_id']];nr=meta[ni];assert meta[ni]['split']=='train'
  pred=int(model.predict_proba(scaler.transform(X[ni:ni+1]))[0,1]>=threshold)
  result=dict(sample_id=r['sample_id'],dataset=name,messages=msgs,saliency=sal,nearest=dict(index=ni,id=nr['id'],source=nr.get('source',nr.get('official_split','train')),messages=nr['messages'],label=int(nr['label']),prediction=pred,cosine=cos),pooling_error=err)
  with (R/'explanations.jsonl').open('a') as out:out.write(json.dumps(result)+'\n')
  count+=1
  if count%100==0:print(json.dumps(dict(completed=count,total=len(rows),seconds=time.time()-started)),flush=True)
(R/'EXTRACTION_COMPLETED').write_text(json.dumps(dict(n=count,max_pooling_error=maxerr,seconds=time.time()-started,sentence_splitter='pysbd English clean=False char_span=True',nearest='cosine of training-standardized pooled activations; train only, ties immutable training ID',explanations_sha256=hashlib.sha256((R/'explanations.jsonl').read_bytes()).hexdigest())))
print((R/'EXTRACTION_COMPLETED').read_text(),flush=True)

for file in ['private.json','weighted_results.json']:(B/file).write_bytes((R/file).read_bytes())
