"""Load one judge once, and commit every production batch exactly once."""
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
os.environ['CUDA_HOME']='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13'
from common import *
import sys,fcntl,subprocess,importlib.metadata,re,traceback
from types import SimpleNamespace

def make_records(chunk,outputs,mode,number,seconds,allowed):
 assert len(outputs)==len(chunk);records=[]
 for p,o in zip(chunk,outputs):
  assert list(o.prompt_token_ids)==p['token_ids'] and len(o.outputs)==1
  g=o.outputs[0];text=g.text.strip()
  assert re.fullmatch(r'[01]' if mode=='hard' else r'(?:0|[1-9][0-9]?|100)',text),f'Invalid output {g.text!r}; preserved without retry'
  if mode=='hard':
   assert len(g.token_ids)==1 and g.token_ids[0] in allowed
   assert g.finish_reason in ['stop','length'] and int(text)==allowed.index(g.token_ids[0])
  else:assert g.finish_reason=='stop','Probability output truncated; preserved without retry'
  row=dict(sample_id=p['sample_id'],prompt_id=p['prompt_id'],dataset=p['dataset'],backbone=p['backbone'],condition=p['condition'],mode=mode,generated_text=g.text,generated_tokens=len(g.token_ids),token_ids=list(g.token_ids),finish_reason=g.finish_reason,input_tokens=p['input_tokens'],batch_number=number,batch_seconds=seconds,instruction_sha256=p['instruction_sha256'])
  if mode=='hard':row['prediction']=int(text)
  else:row.update(positive_probability=int(text),prediction_at_50=int(int(text)>=50))
  records.append(row)
 return records

def validate_committed_batch(chunk,doc,raw,mode,number,allowed):
 assert len(raw)==len(chunk) and doc['records']
 seconds=doc['records'][0]['batch_seconds']
 assert all(r['batch_seconds']==seconds for r in doc['records'])
 outputs=[SimpleNamespace(prompt_token_ids=o['prompt_token_ids'],outputs=[SimpleNamespace(**g) for g in o['outputs']]) for o in raw]
 rebuilt=make_records(chunk,outputs,mode,number,seconds,allowed)
 assert rebuilt==doc['records'],'Committed records disagree with preserved raw generations'
 return rebuilt

def main(j):
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 reserve();spec=specs()[j];prepared=read(ROOT/'prepared'/f'{j}.json');batch_size=spec['batch_size']
 assert prepared['spec_sha256']==digest(json.dumps(spec,sort_keys=True))
 engine=dict(spec['engine'],model=spec['model_path'],max_model_len=((prepared['max_input_tokens']+8+255)//256)*256,max_num_seqs=batch_size,structured_outputs_config={'backend':'xgrammar'})
 lock=(ROOT/'gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 shared=(ROOT.parent/'grid-completion-20261008/gpu-stage.lock').open('a');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','tokenizers','mistral-common','compressed-tensors']},python=sys.version,executable=sys.executable,host=__import__('socket').gethostname(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),environment={k:os.environ.get(k) for k in ['PYTHONHASHSEED','CUDA_HOME','PYTHONPATH','VLLM_USE_FLASHINFER_SAMPLER','VLLM_WORKER_MULTIPROC_METHOD','OMP_NUM_THREADS','HF_HUB_OFFLINE']})
 global_cfg=dict(judge=j,model=spec['repo'],revision=spec['revision'],engine=engine,batch_size=batch_size,environment=env,prepared_sha256=sha(ROOT/'prepared'/f'{j}.json'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),common_sha256=sha(Path(__file__).with_name('common.py')),nonreasoning_template_kwargs=spec['chat_template_kwargs'],sampling=dict(temperature=0,seed=20261001,hard=dict(max_tokens=1,allowed_labels=['0','1'],logprobs=2),probability=dict(max_tokens=8,structured_choice=[str(i) for i in range(101)],logprobs=None)),reused=0)
 gd=ROOT/'models'/j;gd.mkdir(parents=True,exist_ok=True)
 if (gd/'config.json').exists():assert read(gd/'config.json')==global_cfg,'Frozen model configuration changed'
 else:write(gd/'config.json',global_cfg)
 pending=[];total_done=0;inference_seconds=0.;tokens_done=0;run_start=time.time()
 for mode,b,c,d in cells(j):
  config=read(d/'approved_config.json');assert config['prompt_sha256']==sha(d/'prompts.jsonl')
  r=d/'run';r.mkdir(exist_ok=True);(r/'batches').mkdir(exist_ok=True)
  frozen=dict(config,engine=engine,batch_size=batch_size,global_config_sha256=sha(gd/'config.json'),versions=env['versions'],sampling=global_cfg['sampling'][mode],reused=0)
  if (r/'config.json').exists():assert read(r/'config.json')==frozen,'Cell configuration changed'
  else:write(r/'config.json',frozen)
  if (r/'COMPLETED').exists():
   done=read(r/'COMPLETED');assert done['config_sha256']==sha(r/'config.json') and done['judgments_sha256']==sha(r/'judgments.jsonl')
   rows=list(jl(d/'prompts.jsonl'));verified=[]
   for number,k in enumerate(range(0,len(rows),batch_size)):
    dest=r/'batches'/f'{number:06d}.json';doc=read(dest)
    assert doc['config_sha256']==done['config_sha256'] and doc['raw_sha256']==sha(dest.with_suffix('.raw.json')) and doc['records_sha256']==digest(json.dumps(doc['records'],sort_keys=True))
    verified+=validate_committed_batch(rows[k:k+batch_size],doc,read(dest.with_suffix('.raw.json')),mode,number,spec['label_token_ids'])
   assert verified==list(jl(r/'judgments.jsonl')) and len(verified)==10000
   total_done+=len(verified)
   inference_seconds+=sum({x['batch_number']:x['batch_seconds'] for x in verified}.values())
   tokens_done+=sum(x['input_tokens'] for x in verified)
  else:pending.append((mode,b,c,d))
 def status(state,**kw):
  reserve();write(gd/'STATUS.json',dict(state=state,judge=j,time=time.time(),completed=total_done,total=300000,inference_seconds=inference_seconds,input_tokens=tokens_done,seconds_this_session=time.time()-run_start,**kw))
 if not pending:
  status('verified_completed');return
 status('loading');t=time.perf_counter();llm=LLM(**engine);load_seconds=time.perf_counter()-t
 write(gd/('LOADED_'+str(time.time_ns())+'.json'),dict(time=time.time(),seconds=load_seconds))
 tok=llm.get_tokenizer();ids=[tok.encode(x,add_special_tokens=False) for x in ['0','1']];assert all(len(x)==1 for x in ids);allowed=[x[0] for x in ids];assert allowed==spec['label_token_ids']
 params={'hard':SamplingParams(temperature=0,seed=20261001,max_tokens=1,allowed_token_ids=allowed,logprobs=2),'probability':SamplingParams(temperature=0,seed=20261001,max_tokens=8,structured_outputs=StructuredOutputsParams(choice=[str(i) for i in range(101)]))}
 status('loaded',load_seconds=load_seconds)
 for mode,b,c,d in pending:
  r=d/'run';config_hash=sha(r/'config.json');prompts=list(jl(d/'prompts.jsonl'));assert len(prompts)==10000
  ids_expected=[x['sample_id'] for x in read(ROOT/mode/b/'selected_flags.json')];assert [p['sample_id'] for p in prompts]==ids_expected
  committed=[];cell_start=time.time()
  for number,k in enumerate(range(0,len(prompts),batch_size)):
   reserve();chunk=prompts[k:k+batch_size];dest=r/'batches'/f'{number:06d}.json';attempt=dest.with_suffix('.attempt.json');rawpath=dest.with_suffix('.raw.json')
   if dest.exists():
    old=read(dest);records=old['records'];assert old['config_sha256']==config_hash and old['raw_sha256']==sha(rawpath) and old['records_sha256']==digest(json.dumps(records,sort_keys=True));assert [x['prompt_id'] for x in records]==[x['prompt_id'] for x in chunk]
    records=validate_committed_batch(chunk,old,read(rawpath),mode,number,allowed)
   else:
    assert not attempt.exists() and not rawpath.exists(),f'Ambiguous prior attempt: {attempt}; no automatic regeneration'
    write(attempt,dict(time=time.time(),prompt_ids=[p['prompt_id'] for p in chunk],config_sha256=config_hash))
    started=time.perf_counter();outputs=llm.generate([{'prompt_token_ids':p['token_ids']} for p in chunk],params[mode],use_tqdm=False);seconds=time.perf_counter()-started
    raw=[dict(request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason,logprobs=[{str(k):v.logprob for k,v in step.items()} for step in (g.logprobs or [])]) for g in o.outputs]) for o in outputs]
    write(rawpath,raw)
    records=make_records(chunk,outputs,mode,number,seconds,allowed)
    write(dest,dict(records=records,config_sha256=config_hash,records_sha256=digest(json.dumps(records,sort_keys=True)),raw_sha256=sha(rawpath)))
   committed+=records;total_done+=len(records);inference_seconds+=records[0]['batch_seconds'];tokens_done+=sum(x['input_tokens'] for x in records)
   if number%5==0 or k+batch_size>=len(prompts):status('judging',backbone=b,condition=c,mode=mode,cell_completed=len(committed),cell_total=len(prompts))
  assert [x['sample_id'] for x in committed]==ids_expected and len(committed)==10000
  tmp=r/'judgments.jsonl.tmp'
  with tmp.open('w') as f:
   for row in committed:f.write(json.dumps(row)+'\n')
   f.flush();os.fsync(f.fileno())
  tmp.replace(r/'judgments.jsonl')
  write(r/'COMPLETED',dict(n=len(committed),time=time.time(),seconds_this_session=time.time()-cell_start,judgments_sha256=sha(r/'judgments.jsonl'),config_sha256=config_hash))
  print(json.dumps(dict(judge=j,mode=mode,backbone=b,condition=c,completed=total_done,total=300000,inference_seconds=inference_seconds)),flush=True)
 assert total_done==300000
 write(gd/'COMPLETED.json',dict(time=time.time(),n=300000,config_sha256=sha(gd/'config.json'),inference_seconds=inference_seconds,input_tokens=tokens_done,load_seconds=load_seconds));status('completed')
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  err=dict(time=time.time(),state='failed',judge=sys.argv[1],traceback=traceback.format_exc())
  write(ROOT/'models'/sys.argv[1]/('FAILED_'+str(time.time_ns())+'.json'),err);write(ROOT/'models'/sys.argv[1]/'STATUS.json',err);raise
