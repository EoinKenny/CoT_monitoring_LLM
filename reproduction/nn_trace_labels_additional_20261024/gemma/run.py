import os
os.environ.update(OMP_NUM_THREADS='8',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13',HF_HUB_OFFLINE='1')
from pathlib import Path
import json,time,hashlib,shutil,fcntl,traceback,collections,importlib.metadata
R=Path(__file__).resolve().parent;FIELDS={'similarity','contrast','assessable'};ALLOWED={'yes','no','uncertain'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def load(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def reserve():
 for p in [R,Path('/opt/dlami/nvme')]:
  if shutil.disk_usage(p).free<20*2**30:raise RuntimeError('Disk below 20GiB reserve')
def parse(raw):
 try:
  x=json.loads(raw['text']);assert set(x)==FIELDS and all(v in ALLOWED for v in x.values());assert x['assessable']!='no' or x['similarity']==x['contrast']=='uncertain'
  if raw['finish_reason']=='length':return dict(status='truncated',labels=None)
  return dict(status='valid',labels=x)
 except Exception:return dict(status='invalid',labels=None)
def main():
 reserve();locks=[]
 for p in [R/'controller.lock',R.parent/'nn-mechanism-two-labels-20261018/gpu.lock',R.parent/'nn-reason-after-20261016/gpu.lock',R.parent/'probability-500-20261012/gpu.lock',R.parent/'grid-completion-20261008/gpu-stage.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 prep=read(R/'PREPARED.json')
 for n,h in prep['files'].items():assert sha(R/n)==h,n
 cfg=read(R/'MODEL_CONFIG.json');assert cfg['versions']=={k:importlib.metadata.version(k) for k in cfg['versions']}
 units=load(R/'units.jsonl');toys=load(R/'smoke_units.jsonl');assert len(units)==prep['unique_prompts']
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(cfg['engine']['model'],local_files_only=True)
 write(R/'STATUS.json',dict(state='tokenizing',total=len(units),time=time.time(),pid=os.getpid()))
 for u in toys+units:
  rendered=tok.apply_chat_template([dict(role='user',content=u['prompt'])],tokenize=False,add_generation_prompt=True,enable_thinking=False)
  u['tokens']=tok.encode(rendered,add_special_tokens=False)
  assert isinstance(u['tokens'],list) and all(isinstance(t,int) for t in u['tokens'])
 maxlen=max(len(u['tokens']) for u in toys+units);assert maxlen+128<=cfg['engine']['max_model_len'],maxlen
 write(R/'RUN_CONFIG.json',dict(model=cfg,prepared_sha256=sha(R/'PREPARED.json'),script_sha256=sha(Path(__file__)),max_input_tokens=maxlen,total_input_tokens=sum(len(u['tokens']) for u in units)))
 write(R/'STATUS.json',dict(state='loading_model',total=len(units),time=time.time(),pid=os.getpid()))
 from vllm import LLM,SamplingParams
 llm=LLM(**cfg['engine']);sampling=SamplingParams(temperature=0,seed=20261001,max_tokens=128)
 def batch(chunk,dest):
  reserve();attempt=dest.with_suffix('.attempt.json');rawfile=dest.with_suffix('.raw.json');commit=dest.with_suffix('.json');ids=[u['unit_id'] for u in chunk]
  if commit.exists():
   obj=read(commit);assert obj['ids']==ids
   if obj['raw_sha256'] is not None:assert sha(rawfile)==obj['raw_sha256']
   return obj['records'],0
  if rawfile.exists():raw=read(rawfile);assert raw['ids']==ids
  elif attempt.exists():
   assert read(attempt)['ids']==ids
   rr=[dict(unit_id=i,status='missing_ambiguous_attempt',labels=None) for i in ids];write(commit,dict(ids=ids,records=rr,raw_sha256=None));return rr,0
  else:
   write(attempt,dict(ids=ids,run_config_sha256=sha(R/'RUN_CONFIG.json'),time=time.time()));t=time.perf_counter();out=llm.generate([dict(prompt_token_ids=u['tokens']) for u in chunk],sampling,use_tqdm=False);sec=time.perf_counter()-t
   assert len(out)==len(chunk)
   raw=dict(ids=ids,seconds=sec,outputs=[dict(unit_id=u['unit_id'],text=o.outputs[0].text,token_ids=list(o.outputs[0].token_ids),finish_reason=o.outputs[0].finish_reason) for u,o in zip(chunk,out)]);write(rawfile,raw)
  rr=[dict(unit_id=o['unit_id'],**parse(o)) for o in raw['outputs']];write(commit,dict(ids=ids,records=rr,raw_sha256=sha(rawfile)));return rr,raw['seconds']
 (R/'smoke').mkdir(exist_ok=True);smoke=[]
 for i in range(0,len(toys),4):rr,_=batch(toys[i:i+4],R/'smoke'/f'{i//4:06d}');smoke.extend(rr)
 valid=all(x['status']=='valid' for x in smoke);matches=[x['labels']==u['expected'] for x,u in zip(smoke,toys)]
 write(R/'SMOKE_RESULTS.json',dict(records=smoke,matches=matches,strict_first_five_pass=valid and all(matches[:5]),note='Sixth intentionally ambiguous toy is diagnostic only.'))
 if not(valid and all(matches[:5])):raise RuntimeError('Smoke did not meet clear toy expectations; preserved outputs for review')
 write(R/'SMOKE_PASSED.json',dict(time=time.time(),n=6,results_sha256=sha(R/'SMOKE_RESULTS.json')))
 (R/'batches').mkdir(exist_ok=True);allrows=[];seconds=0;counts=collections.Counter();start=time.time()
 for i in range(0,len(units),4):
  rr,sec=batch(units[i:i+4],R/'batches'/f'{i//4:06d}');allrows.extend(rr);seconds+=sec;counts.update(x['status'] for x in rr)
  state=dict(state='running',pid=os.getpid(),completed=len(allrows),total=len(units),status_counts=dict(counts),generation_seconds=seconds,time=time.time(),wall_seconds=time.time()-start)
  if len(allrows)>=40:
   remaining=(len(units)-len(allrows))*seconds/len(allrows);state.update(remaining_hours=remaining/3600,planning_remaining_hours=remaining*1.25/3600,planning_finish_epoch=time.time()+remaining*1.25)
   write(R/'ETA.json',state)
  write(R/'STATUS.json',state)
  if i==0:write(R/'FULL_LAUNCHED.json',dict(time=time.time(),pid=os.getpid(),total=len(units),first_commit_sha256=sha(R/'batches/000000.json')))
 with (R/'annotations.jsonl').open('w') as f:
  for row in allrows:f.write(json.dumps(row)+'\n')
 assert len({x['unit_id'] for x in allrows})==len(units)
 write(R/'OUTPUT_SHA256.json',{str(p.relative_to(R)):sha(p) for p in (R/'batches').glob('*.json')})
 write(R/'COMPLETED.json',dict(time=time.time(),total=len(units),status_counts=dict(counts),annotations_sha256=sha(R/'annotations.jsonl'),output_manifest_sha256=sha(R/'OUTPUT_SHA256.json')));write(R/'STATUS.json',dict(state='completed',completed=len(allrows),total=len(units),time=time.time(),status_counts=dict(counts)))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(R/f'FAILED_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
