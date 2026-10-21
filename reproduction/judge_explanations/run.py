import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
import json,time,hashlib,sys,traceback,fcntl,importlib.metadata
from pathlib import Path
ROOT=Path(sys.argv[1])
RUN=ROOT/'run'

def main():
 from vllm import LLM,SamplingParams
 RUN.mkdir(exist_ok=True)
 lock=(RUN/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (RUN/'judgments.jsonl').exists(), 'Do not overwrite completed judgments'
 start=time.time()
 def write(name,value):
  p=RUN/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,p)
 def log(event,**kw):
  v=dict(event=event,elapsed=time.time()-start,**kw);write('status.json',v);print(json.dumps(v),flush=True)
 def fail(typ,exc,tb):
  write('FAILED.json',dict(error=str(exc),traceback=''.join(traceback.format_exception(typ,exc,tb))))
  sys.__excepthook__(typ,exc,tb)
 sys.excepthook=fail
 original=ROOT
 config=json.loads((original/'approved_config.json').read_text())
 prompts=[json.loads(l) for l in (original/'prompts.jsonl').read_text().splitlines()]
 assert len(prompts)==5190
 path=(ROOT/'model_path.txt').read_text().strip()
 engine=dict(model=path,max_model_len=((max(p['input_tokens'] for p in prompts)+1+255)//256)*256,gpu_memory_utilization=0.85,max_num_seqs=16,max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=False,seed=20261001,limit_mm_per_prompt={'image':0,'video':0,'audio':0},dtype='bfloat16')
 config.update(backend='vllm',flashinfer_sampler=False,versions={p:importlib.metadata.version(p) for p in ['vllm','torch','transformers']},engine=engine,batch_size=16,prompt_sha256=hashlib.sha256((original/'prompts.jsonl').read_bytes()).hexdigest(),decision='greedy one-token generation restricted to labels 0 and 1')
 write('config.json',config)
 (RUN/'prompts.jsonl').write_bytes((original/'prompts.jsonl').read_bytes())
 log('loading')
 llm=LLM(**engine)
 tok=llm.get_tokenizer()
 ids=[tok.encode(s,add_special_tokens=False) for s in ['0','1']]
 assert all(len(x)==1 for x in ids)
 allowed=[x[0] for x in ids]
 params=SamplingParams(temperature=0,max_tokens=1,allowed_token_ids=allowed,logprobs=2,seed=20261001)
 log('loaded')
 pending=prompts
 for offset in range(0,len(pending),16):
  chunk=pending[offset:offset+16]
  tokenized=[p['token_ids'] for p in chunk]
  assert all(len(t)==p['input_tokens'] for t,p in zip(tokenized,chunk))
  batchstart=time.time()
  outputs=llm.generate([{'prompt_token_ids':t} for t in tokenized],params,use_tqdm=False)
  assert len(outputs)==len(chunk)
  records=[]
  for p,t,o in zip(chunk,tokenized,outputs):
   assert o.prompt_token_ids==t
   g=o.outputs[0];assert len(g.token_ids)==1 and g.token_ids[0] in allowed
   records.append(dict(sample_id=p['sample_id'],dataset=p['dataset'],prediction=allowed.index(g.token_ids[0]),generated_text=g.text,token_id=g.token_ids[0],logprobs={str(k):v.logprob for k,v in g.logprobs[0].items()},input_tokens=p['input_tokens'],batch_offset=offset,batch_seconds=time.time()-batchstart))
  with (RUN/'judgments.jsonl').open('a') as f:
   for r in records:f.write(json.dumps(r)+'\n')
   f.flush();os.fsync(f.fileno())
  log('judged',completed=offset+len(chunk),total=len(prompts),batch_seconds=time.time()-batchstart)
 write('COMPLETED',dict(n=len(prompts),elapsed=time.time()-start))
 log('completed',n=len(prompts))
if __name__=='__main__':main()
