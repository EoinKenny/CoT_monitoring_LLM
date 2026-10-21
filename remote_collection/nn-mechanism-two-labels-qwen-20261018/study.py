"""Additional local-judge two-label annotation study; no outcome-dependent selection."""
import os
os.environ.update(OMP_NUM_THREADS='8',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13',HF_HUB_OFFLINE='1')
import json,gzip,hashlib,time,shutil,fcntl,traceback,collections,csv,sys,importlib.metadata
from pathlib import Path
R=Path(__file__).resolve().parent;SOURCE=R.parent/'nn-reason-after-20261016';PROB=R.parent/'probability-500-20261012'
BBS=['llama','qwen','gemma','llama70','qwen122'];FIELDS=['labelled_similarity','important_contrast'];ALLOWED=['yes','no','uncertain']
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def jl(p):
 with Path(p).open() as f:
  for line in f:yield json.loads(line)
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def jsonlines(p,rr):
 tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:
  for r in rr:f.write(json.dumps(r,ensure_ascii=False)+'\n')
  f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def reserve():
 for p in [R,Path('/opt/dlami/nvme')]:
  if shutil.disk_usage(p).free<20*2**30:raise RuntimeError('Below20GiB reserve: '+str(p))
def unpack(r):
 sysmsg,usr=r['messages'];assert sysmsg['role']=='system' and usr['role']=='user'
 task=sysmsg['content'].split('\n\nTask: ',1)[1].split('\n\n',1)[0]
 body=usr['content'];prefix='Classify this supplied conversation JSON:\n';assert body.startswith(prefix)
 text=body[len(prefix):];decoder=json.JSONDecoder();_,end=decoder.raw_decode(text);target=text[:end]
 sep='\n\nNearest-neighbour explanation:\nThe most similar training example was:\n';assert text[end:].startswith(sep)
 text=text[end+len(sep):];_,end=decoder.raw_decode(text);neighbour=text[:end]
 rest=text[end:];prefix='\n\nReference example:\nDataset label: ';assert rest.startswith(prefix)
 label,rest=rest[len(prefix):].split(' |\nProbe prediction: ',1)
 prediction,cue=rest.split('\n\n',1);assert cue=='Positive-class probability (0-100):'
 return dict(task_definition=task,target_text=target,neighbour_text=neighbour,neighbour_label=label,neighbour_probe_prediction=prediction)
def prepare():
 reserve();assert not (R/'PREPARED.json').exists(),'Preparation already frozen'
 template=(R/'prompt_template.txt').read_text();smoke=read(R/'SMOKE_CONFIG.json');assert digest(template)==smoke['prompt_template_sha256']
 canonical=SOURCE/'canonical/cases.jsonl';assert sha(canonical)==read(SOURCE/'canonical/PREPARED.json')['cases_sha256']
 sources={str(canonical):sha(canonical)};selected={}
 for b in BBS:
  p=PROB/'score'/b/'selected_flags.json';assert sha(p)==read(p.with_name('SAMPLE_FROZEN.json'))['selection_sha256']
  sources[str(p)]=sha(p)
  for r in read(p):
   if r['split']=='test':selected[b,r['sample_id']]=r
 assert len(selected)==25000
 units={};mapping=[];coverage=collections.Counter();seen=set();examples={}
 for r in jl(canonical):
  if r['split']!='test':continue
  key=r['backbone'],r['sample_id'];assert key not in seen and key in selected;seen.add(key);meta=selected[key]
  assert r['dataset']==meta['dataset']
  assert digest(json.dumps(r['messages'],ensure_ascii=False,separators=(',',':')))==r['messages_sha256']
  values=unpack(r);prob=meta['probe_probability'];assert 0<=prob<=1
  values['target_probe_prediction']='Positive-class probability: '+repr(prob*100)+'%'
  # Replace only template placeholders; never recursively modify supplied text.
  import re
  prompt=re.sub(r'\{(task_definition|target_text|target_probe_prediction|neighbour_text|neighbour_label|neighbour_probe_prediction)\}',lambda match:values[match.group(1)],template)
  uid=digest(prompt)
  if uid not in units:units[uid]=dict(unit_id=uid,prompt=prompt,prompt_sha256=uid,dataset=r['dataset'],backbone=r['backbone'])
  mapping.append(dict(unit_id=uid,backbone=r['backbone'],parent_prompt_id=r['parent_prompt_id'],source_prompt_sha256=r['messages_sha256'],**meta))
  coverage[r['backbone']+'/'+r['dataset']]+=1;examples.setdefault(r['dataset'],dict(unit_id=uid,**values,prompt=prompt))
 assert seen==set(selected) and len(coverage)==50 and set(coverage.values())=={500}
 queues=collections.defaultdict(list)
 for uid,u in units.items():queues[u['backbone'],u['dataset']].append(uid)
 for k in queues:queues[k]=collections.deque(sorted(queues[k],key=lambda x:digest('20261018:'+x)))
 order=[]
 while any(queues.values()):
  for k in sorted(queues):
   if queues[k]:order.append(queues[k].popleft())
 assert len(order)==len(set(order))==len(units)
 jsonlines(R/'units.jsonl',[units[u] for u in order]);jsonlines(R/'mapping_PRIVATE.jsonl',mapping)
 write(R/'PROMPT_EXAMPLES.json',examples);write(R/'INPUT_SHA256.json',sources)
 write(R/'PROTOCOL.json',dict(scope='All 25000 original 1-NN test contexts: five backbones, ten datasets, 500 test cases per cell. All outcomes including stable cases, no outcome-based selection. Annotation shared across original judges.',labels=FIELDS,allowed=ALLOWED,prompt='Approved toy-tested template unchanged; no evidence requested. No prompt change to resolve the one toy ambiguity.',input='Exact original task definition and JSON target/reference text, reference dataset label and binary reference probe prediction. Target positive-class probe probability from frozen selected_flags manifest fills the approved target-prediction field. That target probability was not displayed in the original nearest-only judge user message; it is supplied here as approved annotation context. Never invent a reference probability.',blinding='Target gold labels, outcome judge IDs/scores, correctness and flip outcomes excluded from model prompts. Ground truth and design weights retained only in private join mapping.',selection='Test set only per approved next-mechanism-study plan. This does not annotate validation cases or written rationale text. Full 10-dataset coverage includes the four-dataset toxicity analysis.',decoding='Existing local Qwen122 GPTQ-Int4 revision and engine, native thinking disabled, temperature0, seed20261001, batch4, max_tokens128, unconstrained generation matching smoke. No automatic regeneration.',failures='Save raw outputs before parsing. Strict two-key JSON; invalid, truncated or missing calls remain explicit. Preserve ambiguous attempted batches without repeating them.',deduplication='Deduplicate only byte-identical fully rendered prompts; retain every original case mapping. Deterministic round-robin over 50 backbone-dataset groups.',limits='Annotations describe similarity/contrast and do not establish causality or internal reasoning. Preserve uncertain labels. Future outcome analysis must respect matched thresholds, weights and repeated targets.',reserve_gib=20))
 write(R/'PREPARED.json',dict(n_cases=len(mapping),unique_prompts=len(units),duplicates=len(mapping)-len(units),coverage=dict(coverage),units_sha256=sha(R/'units.jsonl'),mapping_sha256=sha(R/'mapping_PRIVATE.jsonl'),template_sha256=digest(template),protocol_sha256=sha(R/'PROTOCOL.json'),script_sha256=sha(Path(__file__)),input_sha256=sources,time=time.time()))
 print(json.dumps(read(R/'PREPARED.json')),flush=True)
def parse(payload,chunk,batch,seconds):
 assert len(payload)==len(chunk)
 out=[]
 for raw,u in zip(payload,chunk):
  assert raw['unit_id']==u['unit_id'];g=raw['outputs'][0];labels=None;status='valid';error=None
  try:
   labels=json.loads(g['text']);assert isinstance(labels,dict) and set(labels)==set(FIELDS)
   assert all(labels[k] in ALLOWED for k in FIELDS)
  except Exception as e:status='invalid';error=str(e);labels=None
  if g['finish_reason']=='length':status='truncated';labels=None
  out.append(dict(unit_id=u['unit_id'],**{k:None if labels is None else labels[k] for k in FIELDS},status=status,parse_error=error,raw_response=g['text'],finish_reason=g['finish_reason'],output_tokens=len(g['token_ids']),batch=batch,batch_seconds=seconds))
 return out
def run():
 reserve();locks=[]
 for p in [R/'gpu.lock',SOURCE/'gpu.lock',R.parent/'nn-reason-flips-20261017/gpu.lock',R.parent/'nn-reason-first-512-20261017/gpu.lock',PROB/'gpu.lock',R.parent/'grid-completion-20261008/gpu-stage.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 prep=read(R/'PREPARED.json')
 for name,key in [('units.jsonl','units_sha256'),('mapping_PRIVATE.jsonl','mapping_sha256'),('prompt_template.txt','template_sha256'),('PROTOCOL.json','protocol_sha256')]:assert sha(R/name)==prep[key]
 assert sha(Path(__file__))==prep['script_sha256']
 units=list(jl(R/'units.jsonl'));assert len(units)==prep['unique_prompts']
 cfg=read(R/'SMOKE_CONFIG.json');assert cfg['versions']=={k:importlib.metadata.version(k) for k in cfg['versions']}
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg['engine']['model'],local_files_only=True)
 write(R/'STATUS.json',dict(state='tokenizing',n=len(units),time=time.time()))
 for i,u in enumerate(units):
  assert digest(u['prompt'])==u['unit_id']
  rendered=tok.apply_chat_template([dict(role='user',content=u['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  u['token_ids']=tok.encode(rendered,add_special_tokens=False)
  if i%1000==0:write(R/'STATUS.json',dict(state='tokenizing',completed=i,n=len(units),time=time.time()))
 max_input=max(len(u['token_ids']) for u in units);engine=dict(cfg['engine']);engine['max_model_len']=max(2048,((max_input+128+255)//256)*256);assert engine['max_model_len']<=32768
 config=dict(cfg,engine=engine,max_input_tokens=max_input,total_input_tokens=sum(len(u['token_ids']) for u in units),preparation_sha256=sha(R/'PREPARED.json'))
 if (R/'CONFIG.json').exists():assert read(R/'CONFIG.json')==config
 else:write(R/'CONFIG.json',config)
 write(R/'STATUS.json',dict(state='loading_model',n=len(units),max_input_tokens=max_input,time=time.time()))
 from vllm import LLM,SamplingParams
 llm=LLM(**engine);sampling=SamplingParams(temperature=0,seed=20261001,max_tokens=128)
 (R/'batches').mkdir(exist_ok=True);allresults=[];newseconds=0;newcount=0;started=time.time();counts=collections.Counter()
 for batch,lo in enumerate(range(0,len(units),cfg['batch_size'])):
  reserve();chunk=units[lo:lo+cfg['batch_size']];dest=R/'batches'/f'{batch:06d}';attempt=dest.with_suffix('.attempt.json');raw=dest.with_suffix('.raw.json');committed=dest.with_suffix('.json')
  if committed.exists():
   saved=read(committed);assert saved['ids']==[u['unit_id'] for u in chunk]
   assert saved['raw_sha256']==(sha(raw) if raw.exists() else None)
   rr=saved['records']
  elif raw.exists():
   a=read(attempt);assert a['ids']==[u['unit_id'] for u in chunk] and a['config_sha256']==sha(R/'CONFIG.json')
   saved=read(raw);rr=parse(saved['responses'],chunk,batch,saved['seconds'])
  elif attempt.exists():
   a=read(attempt);assert a['ids']==[u['unit_id'] for u in chunk] and a['config_sha256']==sha(R/'CONFIG.json')
   rr=[dict(unit_id=u['unit_id'],labelled_similarity=None,important_contrast=None,status='missing_ambiguous_attempt',parse_error=None,raw_response=None,finish_reason=None,output_tokens=0,batch=batch,batch_seconds=0) for u in chunk]
  else:
   write(attempt,dict(ids=[u['unit_id'] for u in chunk],config_sha256=sha(R/'CONFIG.json'),time=time.time()))
   st=time.perf_counter();oo=llm.generate([dict(prompt_token_ids=u['token_ids']) for u in chunk],sampling,use_tqdm=False);seconds=time.perf_counter()-st
   payload=[dict(unit_id=u['unit_id'],request_id=o.request_id,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for u,o in zip(chunk,oo)]
   write(raw,dict(seconds=seconds,responses=payload));rr=parse(payload,chunk,batch,seconds);newseconds+=seconds;newcount+=len(chunk)
  if not committed.exists():write(committed,dict(ids=[u['unit_id'] for u in chunk],records=rr,raw_sha256=sha(raw) if raw.exists() else None,attempt_sha256=sha(attempt)))
  allresults.extend(rr);counts.update(r['status'] for r in rr)
  status=dict(state='running',completed=len(allresults),total=len(units),status_counts=dict(counts),new_generation_seconds=newseconds,new_calls=newcount,wall_seconds=time.time()-started,time=time.time(),last_batch=batch)
  if newcount>=200:
   remaining=(len(units)-len(allresults))*newseconds/newcount;status.update(estimated_remaining_hours=remaining/3600,planning_remaining_hours=remaining*1.25/3600)
   if newcount==200 or batch%50==0:write(R/'ETA.json',dict(**status,estimated_finish_utc_epoch=time.time()+remaining,planning_finish_utc_epoch=time.time()+remaining*1.25,note='Measured on round-robin real prompts;25%planning margin; prompt lengths and decoding vary.'))
  write(R/'STATUS.json',status)
  if batch==0:
   write(R/'FIRST_BATCH_CHECK.json',dict(n=len(rr),valid=sum(r['status']=='valid' for r in rr),statuses=dict(counts),time=time.time()))
   if not any(r['status']=='valid' for r in rr):raise RuntimeError('First production batch has no valid JSON; retained without retry')
  if len(allresults)==cfg['batch_size']:write(R/'FULL_LAUNCHED.json',dict(pid=os.getpid(),time=time.time(),first_batch_sha256=sha(committed),total=len(units),prepared_sha256=sha(R/'PREPARED.json')))
 result_ids=[r['unit_id'] for r in allresults];assert len(result_ids)==len(set(result_ids))==len(units) and set(result_ids)=={u['unit_id'] for u in units}
 jsonlines(R/'annotations.jsonl',allresults);by={r['unit_id']:r for r in allresults};mapping=list(jl(R/'mapping_PRIVATE.jsonl'));assert len(mapping)==25000
 dest=R/'labels_by_case.csv';temp=dest.with_suffix('.tmp')
 fields=['backbone','dataset','sample_id','unit_id']+FIELDS+['status']
 with temp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for m in mapping:w.writerow({**{k:m[k] for k in fields[:4]},**{k:by[m['unit_id']][k] for k in fields[4:]}})
  f.flush();os.fsync(f.fileno())
 temp.replace(dest)
 checks={str(p.relative_to(R)):sha(p) for p in (R/'batches').glob('*.json')};write(R/'OUTPUT_SHA256.json',checks)
 write(R/'COMPLETED.json',dict(time=time.time(),n_units=len(units),n_cases=len(mapping),status_counts=dict(counts),annotations_sha256=sha(R/'annotations.jsonl'),case_labels_sha256=sha(dest),output_manifest_sha256=sha(R/'OUTPUT_SHA256.json'),prepared_sha256=sha(R/'PREPARED.json'),generation_seconds=newseconds))
 write(R/'STATUS.json',dict(state='completed',completed=len(units),total=len(units),status_counts=dict(counts),time=time.time()))
if __name__=='__main__':
 try:
  if sys.argv[1]=='prepare':prepare()
  elif sys.argv[1]=='run':run()
  else:raise ValueError('Unknown mode')
 except BaseException:
  write(R/f'FAILED_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
