import os
os.environ['OMP_NUM_THREADS']='8';os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import sys,time,fcntl,importlib.metadata,traceback
from common import *
def main():
 from vllm import LLM,SamplingParams
 directory=Path(sys.argv[1]);run=directory/'run';run.mkdir(exist_ok=True);lock=(run/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.time()
 config=read(directory/'approved_config.json');prompts=[json.loads(l) for l in (directory/'prompts.jsonl').open()];assert len(prompts)==config['n'];assert len({p['sample_id'] for p in prompts})==len(prompts);assert sha(directory/'prompts.jsonl')==config['prompt_sha256']
 path=(directory/'model_path.txt').read_text().strip();engine=dict(model=path,max_model_len=((max((p['input_tokens'] for p in prompts),default=1)+1+255)//256)*256,gpu_memory_utilization=.85,max_num_seqs=16,max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=False,seed=20261001,limit_mm_per_prompt={'image':0,'video':0,'audio':0},dtype='bfloat16');batch_size=16
 if directory.name.startswith('qwen122_'):engine.update(quantization='gptq_marlin',gpu_memory_utilization=.92,max_num_seqs=4,max_num_batched_tokens=2048);batch_size=4
 if directory.name.startswith('llama_'):engine.update(quantization='awq_marlin',gpu_memory_utilization=.92,max_num_seqs=4,max_num_batched_tokens=1024);engine.pop('limit_mm_per_prompt');batch_size=4
 if directory.name.startswith(('glm_','olmo_')):engine.update(quantization='fp8_per_tensor',gpu_memory_utilization=.90,max_num_seqs=8,max_num_batched_tokens=2048);engine.pop('limit_mm_per_prompt');batch_size=8
 frozen=dict(config,host=__import__('socket').gethostname(),gpu=__import__('subprocess').check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip(),engine=engine,batch_size=batch_size,versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers']},sampling=dict(temperature=0,max_tokens=1,seed=20261001,allowed_labels=['0','1']),reused=0)
 if (run/'config.json').exists():assert read(run/'config.json')==frozen,'Configuration changed across resume'
 else:write(run/'config.json',frozen)
 config_hash=sha(run/'config.json');chunks=[prompts[i:i+batch_size] for i in range(0,len(prompts),batch_size)];batches=run/'batches';batches.mkdir(exist_ok=True)
 def status(event,**kw):reserve();write(run/'status.json',dict(event=event,elapsed=time.time()-start,**kw));print(json.dumps(dict(event=event,**kw)),flush=True)
 completed=[];pending=[]
 for number,chunk in enumerate(chunks):
  dest=batches/f'{number:06d}.json';attempt=dest.with_suffix('.attempt.json')
  if dest.exists():
   doc=read(dest);records=doc['records'];assert sha(dest.with_suffix('.raw.json'))==doc['raw_sha256'];assert doc['config_sha256']==config_hash;assert doc['records_sha256']==digest(json.dumps(records,sort_keys=True));assert [r['sample_id'] for r in records]==[p['sample_id'] for p in chunk];assert all(r['prediction'] in [0,1] for r in records);completed+=records
  else:
   assert not attempt.exists(),f'Unresolved inference attempt: {attempt}; refusing silent retry'
   pending.append((number,chunk))
 if (run/'COMPLETED').exists():
  assert not pending and len(completed)==len(prompts)
  assert sha(run/'judgments.jsonl')==read(run/'COMPLETED')['judgments_sha256']
  write(run/('RESUME_VERIFIED_'+str(time.time_ns())+'.json'),dict(n=len(completed),config_sha256=config_hash))
  print('Verified committed run; no model loading or regeneration',flush=True);return
 if pending:
  reserve();status('loading',completed=len(completed),total=len(prompts));llm=LLM(**engine);tok=llm.get_tokenizer();ids=[tok.encode(s,add_special_tokens=False) for s in ['0','1']];assert all(len(x)==1 for x in ids);allowed=[x[0] for x in ids];params=SamplingParams(temperature=0,max_tokens=1,allowed_token_ids=allowed,logprobs=2,seed=20261001);status('loaded')
  for number,chunk in pending:
   reserve();dest=batches/f'{number:06d}.json';attempt=dest.with_suffix('.attempt.json');write(attempt,dict(time=time.time(),ids=[p['sample_id'] for p in chunk],config_sha256=config_hash));started=time.time()
   outputs=llm.generate([{'prompt_token_ids':p['token_ids']} for p in chunk],params,use_tqdm=False)
   # Preserve the returned raw generations before class/shape validation.
   raw=[dict(request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,logprobs=[{str(k):v.logprob for k,v in step.items()} for step in (g.logprobs or [])]) for g in o.outputs]) for o in outputs]
   write(dest.with_suffix('.raw.json'),raw);assert len(outputs)==len(chunk);records=[]
   for p,o in zip(chunk,outputs):
    assert o.prompt_token_ids==p['token_ids'];g=o.outputs[0];assert len(g.token_ids)==1 and g.token_ids[0] in allowed
    records.append(dict(sample_id=p['sample_id'],dataset=p['dataset'],prediction=allowed.index(g.token_ids[0]),generated_text=g.text,token_id=g.token_ids[0],logprobs={str(k):v.logprob for k,v in g.logprobs[0].items()},input_tokens=p['input_tokens'],batch_number=number,batch_seconds=time.time()-started,instruction_sha256=p['instruction_sha256']))
   write(dest,dict(config_sha256=config_hash,records=records,records_sha256=digest(json.dumps(records,sort_keys=True)),raw_sha256=sha(dest.with_suffix('.raw.json'))));completed+=records;status('judged',completed=len(completed),total=len(prompts),batch_seconds=time.time()-started)
 by={r['sample_id']:r for r in completed};assert len(by)==len(completed)==len(prompts);ordered=[by[p['sample_id']] for p in prompts]
 tmp=run/'judgments.jsonl.tmp'
 with tmp.open('w') as f:
  for row in ordered:f.write(json.dumps(row)+'\n')
  f.flush();os.fsync(f.fileno())
 os.replace(tmp,run/'judgments.jsonl');write(run/'COMPLETED',dict(n=len(prompts),seconds_this_session=time.time()-start,judgments_sha256=sha(run/'judgments.jsonl'),config_sha256=config_hash));status('completed',n=len(prompts))
if __name__=='__main__':
 try:main()
 except Exception as e:
  directory=Path(sys.argv[1]);write(directory/'run'/('FAILED_'+str(time.time_ns())+'.json'),dict(error=str(e),traceback=traceback.format_exc()));raise
