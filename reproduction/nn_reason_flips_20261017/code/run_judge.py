from parser_source import *

def main(j):
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 reserve();locks=[]
 for path in [SOURCE/'gpu.lock',ROOT/'gpu.lock',ROOT.parent/'grid-completion-20261008/gpu-stage.lock']:
  f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 for n,h in read(ROOT/'CODE_FROZEN.json').items():assert sha(ROOT/n)==h,n
 D=ROOT/'models'/j;cfg=read(D/'parent_config.json')
 assert cfg['versions']=={p:importlib.metadata.version(p) for p in cfg['versions']}
 assert cfg['gpu']==subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip()
 s=specs()[j];data={};plans={}
 for _,b,c,d in cells(j):
  z=read(d/'FROZEN.json')
  for file,key in [('prompts.jsonl','prompts_sha256'),('selection.json','selection_sha256'),('reused.json','reused_sha256'),('PLAN.json','plan_sha256')]:assert sha(d/file)==z[key],str(d/file)
  assert z['parent_config_sha256']==sha(D/'parent_config.json')
  pp=list(jl(d/'prompts.jsonl'));assert len(pp)==z['n']
  data[b]=pp;plans[b]=read(d/'PLAN.json');(d/'run/batches').mkdir(parents=True,exist_ok=True)
 write(D/'STATUS.json',dict(state='loading',time=time.time()))
 llm=LLM(**cfg['engine']);params=SamplingParams(temperature=0,seed=20261001,max_tokens=256,structured_outputs=StructuredOutputsParams(regex=REGEX))
 write(D/f'EXECUTION_{time.time_ns()}.json',dict(time=time.time(),source_sha256=sha(Path(__file__)),parent_config_sha256=sha(D/'parent_config.json')))
 done=0;new_n=0;seconds=0.
 for _,b,c,d in cells(j):
  pp=data[b];records=read(d/'reused.json');run=d/'run';conf=sha(d/'FROZEN.json')
  for number,indices in enumerate(plans[b]):
   reserve();chunk=[pp[k] for k in indices];dest=run/'batches'/f'{number:06d}.json';raw=dest.with_suffix('.raw.json');attempt=dest.with_suffix('.attempt.json')
   if dest.exists():
    z=read(dest);assert z['config_sha256']==conf and z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True))
    if z['raw_available']:
     assert sha(raw)==z['raw_sha256'];rr=parse(read(raw),chunk,number,z['records'][0]['batch_seconds'])
    else:
     assert not raw.exists() and sha(attempt)==z['attempt_sha256'];rr=missing(chunk,number,0.)
    assert rr==z['records']
   else:
    if raw.exists():
     a=read(attempt);assert a['ids']==[p['prompt_id'] for p in chunk] and a['config_sha256']==conf
     timing=dest.with_suffix('.timing.json');elapsed=read(timing)['seconds'] if timing.exists() else 0.
     rr=parse(read(raw),chunk,number,elapsed)
    elif attempt.exists():
     a=read(attempt);assert a['ids']==[p['prompt_id'] for p in chunk] and a['config_sha256']==conf;rr=missing(chunk,number,0.)
    else:
     write(attempt,dict(time=time.time(),ids=[p['prompt_id'] for p in chunk],config_sha256=conf))
     t=time.perf_counter();oo=llm.generate([{'prompt_token_ids':p['token_ids']} for p in chunk],params,use_tqdm=False);elapsed=time.perf_counter()-t
     payload=[dict(request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for o in oo]
     write(raw,payload);write(dest.with_suffix('.timing.json'),dict(seconds=elapsed));rr=parse(payload,chunk,number,elapsed);seconds+=elapsed;new_n+=len(rr)
    write(dest,dict(records=rr,records_sha256=digest(json.dumps(rr,sort_keys=True)),config_sha256=conf,raw_available=raw.exists(),raw_sha256=sha(raw) if raw.exists() else None,attempt_sha256=sha(attempt)))
   records.extend(rr)
   write(D/'STATUS.json',dict(state='judging',backbone=b,completed=done+len(records),new_n=new_n,new_seconds=seconds,time=time.time()))
  by={r['sample_id']:r for r in records};assert len(by)==len(records)==len(pp)
  temp=run/'judgments.jsonl.tmp'
  with temp.open('w') as f:
   for p in pp:f.write(json.dumps(by[p['sample_id']])+'\n')
   f.flush();os.fsync(f.fileno())
  temp.replace(run/'judgments.jsonl');write(run/'COMPLETED.json',dict(n=len(pp),judgments_sha256=sha(run/'judgments.jsonl'),time=time.time(),frozen_sha256=conf))
  done+=len(pp)
 write(D/'COMPLETED.json',dict(n=done,time=time.time(),new_n=new_n,new_seconds=seconds));write(D/'STATUS.json',dict(state='completed',completed=done,time=time.time()))
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  write(ROOT/'models'/sys.argv[1]/f'FAILED_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
