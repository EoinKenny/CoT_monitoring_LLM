"""Isolated throughput experiment. Never inserts outputs into canonical study."""
import os
os.environ.update(OMP_NUM_THREADS='8',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1')
import json,time,hashlib,sys,subprocess,traceback,threading,statistics
from pathlib import Path
ROOT=Path('/home/ubuntu/experiments/batch-benchmark-20261009')
SRC=Path('/home/ubuntu/experiments/grid-completion-20261008/qwen122/judges')
def read(p):return json.loads(p.read_text())
def write(p,v):
 p.parent.mkdir(exist_ok=True,parents=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare(model):
 d=ROOT/model;d.mkdir(parents=True,exist_ok=True)
 if (d/'sample.json').exists():return
 rows=[];inputs={}
 for c in ['baseline','saliency','nearest']:
  p=SRC/f'{model}_{c}'/'prompts.jsonl';cfg=read(p.parent/'approved_config.json');assert sha(p)==cfg['prompt_sha256'];inputs[str(p)]=sha(p)
  pool=[json.loads(l) for l in p.open()]
  for ds in sorted({r['dataset'] for r in pool}):
   a=sorted([r for r in pool if r['dataset']==ds],key=lambda r:hashlib.sha256(('20261009'+r['sample_id']).encode()).hexdigest())
   chosen={r['sample_id']:r for r in a[:8]};longest=max(a,key=lambda r:r['input_tokens']);chosen[longest['sample_id']]=longest
   for r in chosen.values():rows.append(dict(r,condition=c,benchmark_id=c+':'+r['sample_id']))
 rows.sort(key=lambda r:hashlib.sha256(r['benchmark_id'].encode()).hexdigest())
 write(d/'sample.json',rows);write(d/'manifest.json',dict(n=len(rows),sources=inputs,sample_sha256=sha(d/'sample.json'),selection='8 SHA256-ranked examples plus longest per dataset and condition; fixed mixed ordering; intentionally includes length stress cases',purpose='throughput only, not canonical predictions',model_config=read(SRC/f'{model}_baseline'/'run/config.json')))
def worker(model,batch,tokens):
 from vllm import LLM,SamplingParams
 d=ROOT/model/f'b{batch}_t{tokens}';d.mkdir(parents=True,exist_ok=True)
 assert not (d/'ATTEMPT.json').exists(),'Refusing to overwrite existing attempt'
 write(d/'ATTEMPT.json',dict(time=time.time(),batch=batch,tokens=tokens))
 rows=read(ROOT/model/'sample.json');original=read(SRC/f'{model}_baseline'/'run/config.json')
 engine=original['engine'].copy();engine.update(max_num_seqs=batch,max_num_batched_tokens=tokens,max_model_len=((max(r['input_tokens'] for r in rows)+256)//256)*256)
 write(d/'config.json',dict(engine=engine,batch=batch,sampling=original['sampling'],sample_sha256=sha(ROOT/model/'sample.json')))
 start=time.perf_counter();llm=LLM(**engine);load=time.perf_counter()-start
 tok=llm.get_tokenizer();ids=[tok.encode(s,add_special_tokens=False) for s in ['0','1']];assert all(len(x)==1 for x in ids);allowed=[x[0] for x in ids]
 params=SamplingParams(temperature=0,max_tokens=1,allowed_token_ids=allowed,logprobs=2,seed=20261001)
 stop=threading.Event();mem=[]
 def monitor():
  while not stop.wait(.5):
   try:mem.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
   except Exception:pass
 th=threading.Thread(target=monitor,daemon=True);th.start()
 elapsed=[];preds=[]
 try:
  # Explicit warmup is saved separately and never used for scientific results.
  warm=llm.generate([{'prompt_token_ids':r['token_ids']} for r in rows[:min(batch,len(rows))]],params,use_tqdm=False)
  write(d/'warmup.json',[dict(id=rows[i]['benchmark_id'],tokens=list(o.outputs[0].token_ids)) for i,o in enumerate(warm)])
  for repeat in range(2):
   t=time.perf_counter();raw=[];labels=[]
   for i in range(0,len(rows),batch):
    part=rows[i:i+batch];outputs=llm.generate([{'prompt_token_ids':r['token_ids']} for r in part],params,use_tqdm=False)
    # Every returned generation retained, including output before validation.
    records=[dict(id=r['benchmark_id'],prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,logprobs=[{str(k):v.logprob for k,v in step.items()} for step in (g.logprobs or [])]) for g in o.outputs]) for r,o in zip(part,outputs)]
    write(d/f'repeat{repeat}_batch{i//batch:04}.json',records)
    assert len(outputs)==len(part)
    for r,o in zip(part,outputs):
     assert o.prompt_token_ids==r['token_ids'];g=o.outputs[0];assert len(g.token_ids)==1 and g.token_ids[0] in allowed;labels.append(allowed.index(g.token_ids[0]))
   elapsed.append(time.perf_counter()-t);preds.append(labels)
  result=dict(model=model,batch=batch,token_budget=tokens,n=len(rows),load_seconds=load,repeat_seconds=elapsed,mean_seconds=statistics.mean(elapsed),cases_per_second=len(rows)/statistics.mean(elapsed),input_tokens_per_second=sum(r['input_tokens'] for r in rows)/statistics.mean(elapsed),peak_gpu_mib=max(mem,default=None),repeat_label_disagreements=sum(a!=b for a,b in zip(*preds)),labels=preds[0])
  write(d/'COMPLETED.json',result);print(json.dumps({k:v for k,v in result.items() if k!='labels'}),flush=True)
 finally:stop.set();th.join(timeout=2)
def controller():
 ROOT.mkdir(parents=True,exist_ok=True);results=[]
 for model in ['qwen','gemma']:
  prepare(model)
  for batch,tokens in [(16,4096),(32,8192),(64,16384),(128,32768)]:
   d=ROOT/model/f'b{batch}_t{tokens}';d.mkdir(exist_ok=True)
   write(ROOT/'STATUS.json',dict(model=model,batch=batch,state='running',time=time.time()))
   with (d/'console.log').open('x') as log:p=subprocess.run([sys.executable,__file__,'worker',model,str(batch),str(tokens)],stdout=log,stderr=subprocess.STDOUT)
   if p.returncode:write(d/'FAILED.json',dict(returncode=p.returncode,time=time.time()));continue
   results.append(read(d/'COMPLETED.json'));write(ROOT/'RESULTS.json',results)
 write(ROOT/'COMPLETED.json',dict(runs=len(results),time=time.time()));write(ROOT/'STATUS.json',dict(state='completed',runs=len(results)))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':
  try:worker(sys.argv[2],int(sys.argv[3]),int(sys.argv[4]))
  except Exception:traceback.print_exc();sys.exit(1)
 else:controller()
