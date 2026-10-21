import os
os.environ['OMP_NUM_THREADS']='8';os.environ['OPENBLAS_NUM_THREADS']='8'
import ast,json,time,hashlib,traceback,sys
from pathlib import Path
import numpy as np,joblib,torch,pysbd
from safetensors import safe_open
from transformers import AutoTokenizer
from threadpoolctl import threadpool_limits
threadpool_limits(8);torch.set_num_threads(8)
B=Path('/home/ubuntu/experiments/ten-dataset-full-20261002');R=B/'judge_inputs';R.mkdir(exist_ok=True)
assert (B/'COMPLETED').exists();assert not (R/'explanations.jsonl').exists()
config={'template_date_string':'01 Oct 2026'}
tokenizer=AutoTokenizer.from_pretrained('/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6',local_files_only=True);special_ids=set(tokenizer.all_special_ids)
tree=ast.parse((B.parent/'logistic-probes-20261001/train.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='content_mask');exec(compile(ast.Module(body=[fn],type_ignores=[]),'<content_mask>','exec'))
results=json.loads((B/'results.json').read_text());rows=[]
for ds,v in results.items():
 metadata=json.loads((B/f'{ds}_metadata.json').read_text());p=np.load(B/f'{ds}_predictions.npz')['probabilities']
 for i,r in enumerate(metadata):
  if r['split']=='test' and p[i]>=v['threshold']:rows.append(dict(sample_id=f'{ds}-{i}',dataset=ds,index=i,true_label=r['label'],probe_probability=float(p[i])))
(R/'private.json').write_text(json.dumps(rows))
(R/'weighted_results.json').write_text(json.dumps({'results':{ds:{'test':v['metrics']['test']} for ds,v in results.items()}}))
seg=pysbd.Segmenter(language='en',clean=False,char_span=True)
count=0;started=time.time();maxerr=0

def render(msgs):return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False,date_string='01 Oct 2026')
def escaped(s,ascii,html):
 t=json.dumps(s,ensure_ascii=ascii)[1:-1]
 if html:
  for a,b in [('<','\\u003c'),('>','\\u003e'),('&','\\u0026'),("'",'\\u0027')]:t=t.replace(a,b)
 return t
for name in results:
 meta=json.loads((B/f'{name}_metadata.json').read_text());X=np.load(B/f'{name}_features.npy',mmap_mode='r');fitted=joblib.load(B/f'{name}_probe.joblib');scaler,model=fitted['scaler'],fitted['model'];threshold=float(results[name]['threshold'])
 trainix=np.array([i for i,r in enumerate(meta) if r['split']=='train']);subset=[r for r in rows if r['dataset']==name]
 T=scaler.transform(X[trainix]).astype(np.float64);T/=np.linalg.norm(T,axis=1)[:,None]
 nn={}
 for a in range(0,len(subset),64):
  group=subset[a:a+64];ix=[r['index'] for r in group];Z=scaler.transform(X[ix]).astype(np.float64);Z/=np.linalg.norm(Z,axis=1)[:,None];sim=Z@T.T;best=sim.argmax(axis=1)
  for k,r in enumerate(group):nn[r['sample_id']]=(int(trainix[best[k]]),float(sim[k,best[k]]))
 w=model.coef_[0]/scaler.scale_;bias=float(model.intercept_[0]-w@scaler.mean_)
 for r in subset:
  ix=r['index'];record=meta[ix];msgs=record['messages'];text=render(msgs)
  with safe_open(str(Path(record['cache_file'])),framework='pt') as sf:
   offsets=sf.get_tensor('offsets');row=record['cache_row'];a,b=int(offsets[row]),int(offsets[row+1]);ids=sf.get_slice('token_ids')[a:b].numpy();acts=sf.get_slice('activations')[a:b].float().numpy()
  mask=content_mask(msgs,ids,name) if 'legacy_dataset' in record else np.isin(np.arange(len(ids)),record['mask']);score=acts.astype(np.float64)@w+bias;expected=float(model.decision_function(scaler.transform(X[ix:ix+1]))[0]);err=abs(float(score[mask].mean())-expected);assert err<1e-4,(name,ix,err);maxerr=max(maxerr,err)
  offs=np.array(tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']);candidates=[]
  for mi,m in enumerate(msgs):
   if record['role']!='all' and m['role']!=record['role']:continue
   content=m['content'];trim=content.strip()
   if not trim:continue
   prefix=render(msgs[:mi+1]);assert text.startswith(prefix)
   if m['role'] in ('tool','ipython'):
    prev=render(msgs[:mi]) if mi else '';h='<|start_header_id|>ipython<|end_header_id|>\n\n';start=prefix.find(h,len(prev))+len(h);body=prefix[start:-len('<|eot_id|>')];assert json.loads(body)==content
    mode=next(( (asc,html) for asc in [True,False] for html in [True,False] if '"'+escaped(content,asc,html)+'"'==body),None);assert mode is not None
    base=start+1
    def charpos(x):return base+len(escaped(content[:x],*mode))
    tosegment=content;shift=0
   else:
    base=prefix.rfind(trim);assert base>=0;tosegment=trim;shift=0
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
(R/'EXTRACTION_COMPLETED').write_text(json.dumps(dict(n=count,max_pooling_error=maxerr,seconds=time.time()-started,sentence_splitter='pysbd English clean=False char_span=True',nearest='cosine of training-standardized pooled activations; train only, ties smallest index',explanations_sha256=hashlib.sha256((R/'explanations.jsonl').read_bytes()).hexdigest())))
print((R/'EXTRACTION_COMPLETED').read_text(),flush=True)

