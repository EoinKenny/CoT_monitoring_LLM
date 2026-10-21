import os
os.environ.update(OMP_NUM_THREADS='8',VLLM_USE_FLASHINFER_SAMPLER='0',TOKENIZERS_PARALLELISM='false',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13',HF_HUB_OFFLINE='1')
import json,time,hashlib,shutil,fcntl,subprocess,importlib.metadata,traceback
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
 t=p.with_name(p.name+'.tmp')
 with t.open('w') as f:json.dump(x,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 t.replace(p)
def reserve():
 for p in [R,Path('/opt/dlami/nvme')]:assert shutil.disk_usage(p).free>20*2**30,'Disk reserve below 20GiB'
def main():
 reserve();locks=[]
 for p in [R/'gpu.lock',R.parent/'nn-reason-after-20261016/gpu.lock',R.parent/'nn-reason-flips-20261017/gpu.lock',R.parent/'grid-completion-20261008/gpu-stage.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 frozen=read(R/'PREFROZEN.json')
 for n,digest in frozen['files'].items():assert sha(R/n)==digest,n
 prompts=read(R/'prompts.json');assert len(prompts)==8
 parent=read(R.parent/'nn-reason-flips-20261017/models/qwen122/parent_config.json')
 assert parent['versions']=={k:importlib.metadata.version(k) for k in parent['versions']}
 spec=read(R/'judge_specs.json')['models']['qwen122'];assert parent['revision']==spec['revision']
 for p in prompts:assert hashlib.sha256(p['prompt'].encode()).hexdigest()==p['prompt_sha256']
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(spec['model_path'],local_files_only=True)
 rendered=[tok.apply_chat_template([dict(role='user',content=p['prompt'])],**spec['chat_template_kwargs']) for p in prompts]
 ids=[tok.encode(s,add_special_tokens=False) for s in rendered]
 assert max(map(len,ids))+128<=parent['engine']['max_model_len']
 write(R/'CONFIG.json',dict(model=parent['model'],revision=parent['revision'],engine=parent['engine'],versions=parent['versions'],batch_size=4,sampling=dict(temperature=0,seed=20261001,max_tokens=128,constrained_decoding=False),native_thinking=False,prompt_template_sha256=sha(R/'prompt_template.txt'),input_token_counts=list(map(len,ids)),script_sha256=sha(Path(__file__))))
 write(R/'ENCODED.json',[dict(id=p['id'],rendered_text=t,token_ids=i) for p,t,i in zip(prompts,rendered,ids)])
 write(R/'STATUS.json',dict(state='loading_model',time=time.time()))
 from vllm import LLM,SamplingParams
 start=time.perf_counter();llm=LLM(**parent['engine']);load_seconds=time.perf_counter()-start
 params=SamplingParams(temperature=0,seed=20261001,max_tokens=128)
 (R/'batches').mkdir(exist_ok=True);results=[];total=0
 for batch,lo in enumerate(range(0,len(prompts),4)):
  reserve();dest=R/'batches'/f'{batch:03d}';attempt=dest.with_suffix('.attempt.json');raw=dest.with_suffix('.raw.json')
  assert not attempt.exists() and not raw.exists(),'Refusing to resubmit attempted batch'
  chunk=prompts[lo:lo+4]
  write(attempt,dict(ids=[p['id'] for p in chunk],time=time.time(),config_sha256=sha(R/'CONFIG.json')))
  write(R/'STATUS.json',dict(state='generating',batch=batch,completed=len(results),time=time.time()))
  start=time.perf_counter();outputs=llm.generate([{'prompt_token_ids':i} for i in ids[lo:lo+4]],params,use_tqdm=False);seconds=time.perf_counter()-start;total+=seconds
  payload=[dict(id=p['id'],request_id=o.request_id,prompt_token_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for p,o in zip(chunk,outputs)]
  write(raw,dict(seconds=seconds,responses=payload));assert len(payload)==len(chunk)
  for p in payload:
   g=p['outputs'][0];parsed=None;error=None
   try:
    parsed=json.loads(g['text']);assert set(parsed)=={'labelled_similarity','important_contrast'}
    assert all(v in ['yes','no','uncertain'] for v in parsed.values())
   except Exception as e:error=str(e)
   results.append(dict(id=p['id'],raw_text=g['text'],parsed=parsed,valid=error is None,parse_error=error,finish_reason=g['finish_reason'],output_tokens=len(g['token_ids']),batch=batch,batch_seconds=seconds))
  write(R/'results.json',results)
 write(R/'COMPLETED.json',dict(n=len(results),valid=sum(x['valid'] for x in results),loading_seconds=load_seconds,generation_seconds=total,results_sha256=sha(R/'results.json'),raw_sha256={p.name:sha(p) for p in (R/'batches').glob('*.raw.json')},time=time.time()))
 write(R/'STATUS.json',dict(state='completed',n=len(results),time=time.time()))
if __name__=='__main__':
 try:main()
 except BaseException:
  write(R/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));raise
