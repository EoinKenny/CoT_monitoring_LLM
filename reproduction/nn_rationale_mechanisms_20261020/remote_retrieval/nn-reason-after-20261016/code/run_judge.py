import os
os.environ.update(OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13')
from common import *
import sys,fcntl,subprocess,importlib.metadata,re,traceback,math

def parse(raw,chunk,number,seconds):
 assert len(raw)==len(chunk);out=[]
 for r,p in zip(raw,chunk):
  assert r['prompt_token_ids']==p['token_ids'] and len(r['outputs'])==1
  g=r['outputs'][0];txt=g['text'];first,sep,reason=txt.partition('\n')
  assert re.fullmatch(r'(?:0|[1-9][0-9]?|100)',first.strip()),repr(txt)
  assert sep and reason.strip(),repr(txt)
  prob=int(first);out.append(dict(sample_id=p['sample_id'],prompt_id=p['prompt_id'],dataset=p['dataset'],backbone=p['backbone'],condition=CS[0],mode='score',prediction=int(prob>=50),positive_probability=prob,prediction_at_50=int(prob>=50),generated_text=txt,reasoning=reason.strip(),reasoning_complete=g['finish_reason']=='stop',token_ids=g['token_ids'],generated_tokens=len(g['token_ids']),finish_reason=g['finish_reason'],input_tokens=p['input_tokens'],batch_number=number,batch_seconds=seconds,instruction_sha256=p['instruction_sha256']))
 return out

def make_plan(prompts,bs,bb):
 # Balanced dataset quotas without using target labels. Quantiles represent input lengths.
 datasets=sorted({p['dataset'] for p in prompts});offset=BBS.index(bb)%len(datasets);datasets=datasets[offset:]+datasets[:offset]
 quota={ds:0 for ds in datasets}
 for k in range(2*bs):quota[datasets[k%len(datasets)]]+=1
 picked=[]
 for ds in datasets:
  pool=sorted([k for k,p in enumerate(prompts) if p['dataset']==ds],key=lambda k:(prompts[k]['input_tokens'],prompts[k]['sample_id']))
  q=quota[ds]
  for k in range(q):
   lo=k*len(pool)//q;hi=(k+1)*len(pool)//q
   picked.append(min(pool[lo:hi],key=lambda ix:digest('20261016:'+prompts[ix]['prompt_id'])))
 picked.sort(key=lambda ix:digest('20261016:order:'+prompts[ix]['prompt_id']))
 assert len(picked)==len(set(picked))==2*bs
 chosen=set(picked);remaining=[k for k in range(len(prompts)) if k not in chosen]
 return [dict(smoke=True,indices=picked[k:k+bs]) for k in range(0,len(picked),bs)]+[dict(smoke=False,indices=remaining[k:k+bs]) for k in range(0,len(remaining),bs)]

def main(j,phase):
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 reserve();s=specs()[j];prep=read(ROOT/'prepared'/f'{j}.json');assert prep['spec_sha256']==digest(json.dumps(s,sort_keys=True));bs=s['batch_size']
 engine=dict(s['engine'],model=s['model_path'],max_model_len=((prep['max_input_tokens']+MAX_OUTPUT+255)//256)*256,max_num_seqs=bs,structured_outputs_config={'backend':'xgrammar'})
 engine['max_model_len']=max(engine['max_model_len'],read(PARENT/'models'/j/'config.json')['engine']['max_model_len'])
 assert engine['max_model_len']<=s['context_limit']
 locks=[]
 for p in [ROOT/'gpu.lock',ROOT.parent/'grid-completion-20261008/gpu-stage.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 D=ROOT/'models'/j;D.mkdir(parents=True,exist_ok=True)
 cfg=dict(judge=j,model=s['repo'],revision=s['revision'],engine=engine,batch_size=bs,sampling=dict(temperature=0,seed=20261001,max_tokens=MAX_OUTPUT,regex=REGEX),prepared_sha256=sha(ROOT/'prepared'/f'{j}.json'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','tokenizers']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
 if (D/'config.json').exists():assert cfg==read(D/'config.json')
 else:write(D/'config.json',cfg)
 data={};plans={}
 for mode,b,c,d in cells(j):
  approved=read(d/'approved_config.json');assert approved['prompt_sha256']==sha(d/'prompts.jsonl')
  pp=list(jl(d/'prompts.jsonl'));assert len(pp)==10000;data[b]=pp
  run=d/'run';(run/'batches').mkdir(parents=True,exist_ok=True)
  frozen=dict(approved,global_config_sha256=sha(D/'config.json'),sampling=cfg['sampling'],engine=engine)
  if (run/'config.json').exists():assert read(run/'config.json')==frozen
  else:write(run/'config.json',frozen)
  plan=make_plan(pp,bs,b)
  if (run/'PLAN.json').exists():assert read(run/'PLAN.json')==plan
  else:write(run/'PLAN.json',plan)
  plans[b]=plan
 write(D/'STATUS.json',dict(state='loading',phase=phase,time=time.time()))
 t=time.perf_counter();llm=LLM(**engine);loading=time.perf_counter()-t
 params=SamplingParams(temperature=0,seed=20261001,max_tokens=MAX_OUTPUT,structured_outputs=StructuredOutputsParams(regex=REGEX))
 smoke_stats=[];done=0;truncated=0;new_elapsed=0;new_n=0
 for mode,b,c,d in cells(j):
  run=d/'run';pp=data[b];confhash=sha(run/'config.json');records=[];smoke=[]
  for number,entry in enumerate(plans[b]):
   if phase=='smoke' and not entry['smoke']:continue
   reserve();chunk=[pp[k] for k in entry['indices']];dest=run/'batches'/f'{number:06d}.json';rawpath=dest.with_suffix('.raw.json');attempt=dest.with_suffix('.attempt.json')
   if dest.exists():
    z=read(dest);assert z['config_sha256']==confhash and z['raw_sha256']==sha(rawpath)
    assert z['records_sha256']==digest(json.dumps(z['records'],sort_keys=True))
    rr=parse(read(rawpath),chunk,number,z['records'][0]['batch_seconds']);assert rr==z['records']
   else:
    assert not attempt.exists() and not rawpath.exists(),f'Ambiguous attempt {attempt}; no automatic retry'
    write(attempt,dict(time=time.time(),ids=[p['prompt_id'] for p in chunk],config_sha256=confhash,phase=phase))
    start=time.perf_counter();oo=llm.generate([{'prompt_token_ids':p['token_ids']} for p in chunk],params,use_tqdm=False);elapsed=time.perf_counter()-start
    raw=[dict(request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for o in oo]
    write(rawpath,raw);rr=parse(raw,chunk,number,elapsed)
    write(dest,dict(records=rr,config_sha256=confhash,raw_sha256=sha(rawpath),records_sha256=digest(json.dumps(rr,sort_keys=True))))
    new_elapsed+=elapsed;new_n+=len(rr)
   records+=rr;done+=len(rr);truncated+=sum(not r['reasoning_complete'] for r in rr)
   if entry['smoke']:smoke+=rr
   write(D/'STATUS.json',dict(state='judging',phase=phase,completed=done,total=50000,time=time.time(),backbone=b,cell_completed=len(records),new_n=new_n,new_inference_seconds=new_elapsed,truncated=truncated))
  seconds=sum(smoke[k]['batch_seconds'] for k in range(0,len(smoke),bs))
  mean_input=sum(p['input_tokens'] for p in pp)/len(pp);sample_mean=sum(p['input_tokens'] for p in smoke)/len(smoke)
  seconds_per_case=seconds/len(smoke);scale=max(1,mean_input/sample_mean)
  stats=dict(backbone=b,judge=j,n=len(smoke),batches=2,seconds=seconds,seconds_per_case=seconds_per_case,mean_input=mean_input,smoke_mean_input=sample_mean,mean_output=sum(p['generated_tokens'] for p in smoke)/len(smoke),max_output=max(p['generated_tokens'] for p in smoke),truncated=sum(not p['reasoning_complete'] for p in smoke),estimated_remaining_seconds=(10000-len(smoke))*seconds_per_case*scale)
  smoke_stats.append(stats)
  assert stats['truncated']==0,'Smoke explanation truncated; committed attempt preserved, full launch paused'
  if phase=='full':
   assert len(records)==10000;by={r['sample_id']:r for r in records};assert len(by)==10000
   ordered=[by[p['sample_id']] for p in pp];tmp=run/'judgments.jsonl.tmp'
   with tmp.open('w') as f:
    for r in ordered:f.write(json.dumps(r)+'\n')
    f.flush();os.fsync(f.fileno())
   tmp.replace(run/'judgments.jsonl');write(run/'COMPLETED',dict(n=10000,judgments_sha256=sha(run/'judgments.jsonl'),config_sha256=confhash,time=time.time()))
 if phase=='smoke':
  write(D/'SMOKE_PASSED.json',dict(judge=j,time=time.time(),cells=smoke_stats,model_loading_seconds=loading,all_probabilities_valid=True,estimated_remaining_seconds=sum(x['estimated_remaining_seconds'] for x in smoke_stats)+loading))
 else:
  assert done==50000;write(D/'COMPLETED.json',dict(n=done,time=time.time(),truncated=truncated,config_sha256=sha(D/'config.json')))
 write(D/'STATUS.json',dict(state='completed',phase=phase,completed=done,time=time.time()))
if __name__=='__main__':
 try:main(*sys.argv[1:3])
 except BaseException:
  write(ROOT/'models'/sys.argv[1]/f'FAILED_{time.time_ns()}.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
