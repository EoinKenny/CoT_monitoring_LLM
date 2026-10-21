import os
os.environ.update(OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13')
from common import *
import sys,fcntl,subprocess,importlib.metadata,re,traceback
REGEX=r'(?:0|[1-9][0-9]?|100)'
def parse(raw,chunk,number,seconds):
 assert len(raw)==len(chunk);out=[]
 for r,p in zip(raw,chunk):
  assert r['prompt_token_ids']==p['token_ids'] and len(r['outputs'])==1;g=r['outputs'][0];text=g['text'].strip();assert re.fullmatch(REGEX,text),f'Invalid probability {text!r}';assert g['finish_reason']=='stop','Truncated output retained';prob=int(text);label=int(prob>=50)
  out.append(dict(sample_id=p['sample_id'],prompt_id=p['prompt_id'],dataset=p['dataset'],backbone=p['backbone'],condition=p['condition'],mode='score',prediction=label,positive_probability=prob,prediction_at_50=int(prob>=50),generated_text=g['text'],token_ids=g['token_ids'],generated_tokens=len(g['token_ids']),finish_reason=g['finish_reason'],input_tokens=p['input_tokens'],batch_number=number,batch_seconds=seconds,instruction_sha256=p['instruction_sha256']))
 return out

def main(j):
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 reserve();s=specs()[j];prep=read(ROOT/'prepared'/f'{j}.json');assert prep['spec_sha256']==digest(json.dumps(s,sort_keys=True));bs=s['batch_size'];engine=dict(s['engine'],model=s['model_path'],max_model_len=((prep['max_input_tokens']+8+255)//256)*256,max_num_seqs=bs,structured_outputs_config={'backend':'xgrammar'})
 original_engine=read(PARENT/'models'/j/'config.json')['engine']
 engine['max_model_len']=max(engine['max_model_len'],original_engine['max_model_len'])
 locks=[]
 for p in [ROOT/'gpu.lock',ROOT.parent/'grid-completion-20261008/gpu-stage.lock']:
  f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 D=ROOT/'models'/j;D.mkdir(parents=True,exist_ok=True);cfg=dict(judge=j,model=s['repo'],revision=s['revision'],engine=engine,batch_size=bs,sampling=dict(temperature=0,seed=20261001,max_tokens=8,regex=REGEX),prepared_sha256=sha(ROOT/'prepared'/f'{j}.json'),protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_sha256=sha(Path(__file__)),versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers','tokenizers']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],text=True).strip())
 if (D/'config.json').exists():assert cfg==read(D/'config.json')
 else:write(D/'config.json',cfg)
 done=0;seconds_total=0.;token_total=0
 def status(state,**kwargs):write(D/'STATUS.json',dict(state=state,completed=done,total=TOTAL,time=time.time(),inference_seconds=seconds_total,**kwargs))
 status('loading');llm=LLM(**engine);params=SamplingParams(temperature=0,seed=20261001,max_tokens=8,structured_outputs=StructuredOutputsParams(regex=REGEX))
 for mode,b,c,cell in cells(j):
  approved=read(cell/'approved_config.json');assert approved['prompt_sha256']==sha(cell/'prompts.jsonl');prompts=list(jl(cell/'prompts.jsonl'));assert len(prompts)==10000;run=cell/'run';(run/'batches').mkdir(parents=True,exist_ok=True);frozen=dict(approved,global_config_sha256=sha(D/'config.json'),sampling=cfg['sampling'],engine=engine)
  if (run/'config.json').exists():assert read(run/'config.json')==frozen
  else:write(run/'config.json',frozen)
  config_hash=sha(run/'config.json');records=[]
  for number,k in enumerate(range(0,len(prompts),bs)):
   reserve();chunk=prompts[k:k+bs];dest=run/'batches'/f'{number:06d}.json';rawpath=dest.with_suffix('.raw.json');attempt=dest.with_suffix('.attempt.json')
   if dest.exists():
    doc=read(dest);assert doc['config_sha256']==config_hash and doc['raw_sha256']==sha(rawpath) and doc['records_sha256']==digest(json.dumps(doc['records'],sort_keys=True));rr=parse(read(rawpath),chunk,number,doc['records'][0]['batch_seconds']);assert rr==doc['records']
   else:
    assert not attempt.exists() and not rawpath.exists(),f'Ambiguous attempt {attempt}; no silent retry'
    write(attempt,dict(time=time.time(),ids=[p['prompt_id'] for p in chunk],config_sha256=config_hash));t=time.perf_counter();outputs=llm.generate([{'prompt_token_ids':p['token_ids']} for p in chunk],params,use_tqdm=False);seconds=time.perf_counter()-t
    raw=[dict(request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for o in outputs];write(rawpath,raw);rr=parse(raw,chunk,number,seconds);write(dest,dict(records=rr,config_sha256=config_hash,raw_sha256=sha(rawpath),records_sha256=digest(json.dumps(rr,sort_keys=True))))
   records+=rr;done+=len(rr);seconds_total+=rr[0]['batch_seconds'];token_total+=sum(x['input_tokens'] for x in rr)
   if number%5==0:status('judging',backbone=b,condition=c,cell_completed=len(records))
  assert [x['sample_id'] for x in records]==[x['sample_id'] for x in read(ROOT/'score'/b/'selected_flags.json')]
  if (run/'COMPLETED').exists():assert sha(run/'judgments.jsonl')==read(run/'COMPLETED')['judgments_sha256'] and list(jl(run/'judgments.jsonl'))==records
  else:
   tmp=run/'judgments.jsonl.tmp'
   with tmp.open('w') as f:
    for r in records:f.write(json.dumps(r)+'\n')
    f.flush();os.fsync(f.fileno())
   tmp.replace(run/'judgments.jsonl');write(run/'COMPLETED',dict(n=10000,config_sha256=config_hash,judgments_sha256=sha(run/'judgments.jsonl'),time=time.time()))
 assert done==TOTAL;write(D/'COMPLETED.json',dict(n=done,time=time.time(),inference_seconds=seconds_total,input_tokens=token_total,config_sha256=sha(D/'config.json')));status('completed')
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  write(ROOT/'models'/sys.argv[1]/('FAILED_'+str(time.time_ns())+'.json'),dict(traceback=traceback.format_exc()));raise
