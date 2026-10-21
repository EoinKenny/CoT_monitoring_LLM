import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
os.environ['CUDA_HOME']='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13'
from pathlib import Path
import json,hashlib,time,re,sys,shutil,traceback,subprocess
ROOT=Path('/home/ubuntu/experiments/reasoning-full-pilot-20261011')
SOURCE=Path('/home/ubuntu/experiments/confidence-threshold-sweep-20261009')
CACHE=Path('/opt/dlami/nvme/interpretable-probe-models')
SEED=20261011
CS=['baseline','saliency','nearest']
BBS=['llama','qwen','gemma','llama70','qwen122']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def lines(p):return [json.loads(l) for l in p.open()]
def guard():
 assert shutil.disk_usage(ROOT).free>20*1024**3
def parse(model,text,finish,mode):
 final=None;reasoning=False
 if model=='gptoss':
  m=re.search(r'<\|start\|>assistant<\|(?:channel|meta_sep)\|>final<\|message\|>(.*)',text,re.S)
  if m:final=m.group(1);reasoning=True
 elif model in ['qwen','glm']:
  if '</think>' in text:final=text.split('</think>',1)[1];reasoning=True
 elif model=='gemma':
  if '<channel|>' in text:final=text.split('<channel|>',1)[1];reasoning=True
  elif '<|channel>thought' not in text:final=text
 if final is not None:
  final=re.sub(r'(?:<\|fim_suffix\|>|<\|endoftext\|>|<\|im_end\|>|<\|return\|>|<turn\|>|<end_of_turn>|\[eop\])\s*$','',final).strip()
 valid=finish=='stop' and final is not None and re.fullmatch(r'\d{1,3}',final) is not None and int(final)<=(1 if mode=='hard' else 100)
 return dict(final=final,valid=bool(valid),reasoning_delimiter=reasoning)
def run(model):
 from transformers import AutoTokenizer
 import vllm,torch,transformers
 from vllm import LLM,SamplingParams
 D=ROOT/model;D.mkdir(exist_ok=True)
 if (D/'COMPLETED.json').exists():
  assert sha(D/'RESULTS.json')==json.loads((D/'COMPLETED.json').read_text())['results_sha256'];return
 if not (D/'STARTED.json').exists():save(D/'STARTED.json',dict(time=time.time(),pid=os.getpid()))
 guard()
 spec=json.loads((ROOT/'EXPANSION_PLAN.json').read_text())['models'][model]
 path=CACHE/('models--'+spec['repo'].replace('/','--'))/'snapshots'/spec['revision'];assert path.exists()
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
 assert sha(ROOT/'cases.jsonl')==json.loads((ROOT/'PREPARED.json').read_text())['cases_sha256']
 rows=lines(ROOT/'cases.jsonl')
 for r in rows:
  opts=dict(reasoning_effort='low') if model=='gptoss' else dict(enable_thinking=True)
  prompt=tok.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,**opts)
  r['token_ids']=tok.encode(prompt,add_special_tokens=False);r['prompt']=prompt;r['input_tokens']=len(r['token_ids'])
 serialized=''.join(json.dumps(r)+'\n' for r in rows)
 if (D/'prompts.jsonl').exists():assert (D/'prompts.jsonl').read_text()==serialized
 else:(D/'prompts.jsonl').write_text(serialized)
 maxlen=((max(r['input_tokens'] for r in rows)+2048+1023)//1024)*1024
 engine=dict(model=str(path),tokenizer=str(path),dtype='auto' if model=='gptoss' else 'bfloat16',max_model_len=maxlen,max_num_seqs=16,max_num_batched_tokens=4096,gpu_memory_utilization=.90 if model=='gptoss' else .85,enable_prefix_caching=False,enforce_eager=True,seed=20261001)
 if model=='gptoss':engine['attention_backend']='TRITON_ATTN'
 if model in ['qwen','gemma']:engine['limit_mm_per_prompt']={'image':0,'video':0,'audio':0}
 config=dict(engine=engine,model=spec,reasoning=True,reasoning_effort='low' if model=='gptoss' else 'native enabled',max_tokens=2048,temperature=0,batch_size=16,prompt_sha256=sha(D/'prompts.jsonl'),vllm=vllm.__version__,torch=torch.__version__,transformers=transformers.__version__,note='Unconstrained native reasoning and final output: validate exact integer final. Truncations and invalid generations are retained, not retried. Balanced exploratory pilot; no forced early final.')
 if (D/'CONFIG.json').exists():assert json.loads((D/'CONFIG.json').read_text())==config
 else:save(D/'CONFIG.json',config)
 save(D/'STATUS.json',dict(state='loading',time=time.time()));t=time.perf_counter();llm=LLM(**engine);load=time.perf_counter()-t;save(D/f'LOADED_{time.time_ns()}.json',dict(seconds=load,time=time.time()))
 params=SamplingParams(temperature=0,seed=20261001,max_tokens=2048,skip_special_tokens=False)
 results=[]
 def batch(tag,chunk):
  guard();p=D/(tag+'.attempt.json');result_path=D/(tag+'.result.json')
  if result_path.exists():
   old=json.loads(result_path.read_text());assert sha(D/(tag+'.raw.json'))==old['raw_sha256'];assert [r['id'] for r in old['records']]==[r['id'] for r in chunk];assert old['config_sha256']==sha(D/'CONFIG.json');return old
  assert not p.exists() and not (D/(tag+'.raw.json')).exists(),'Ambiguous prior batch: preserve and investigate, never regenerate automatically'
  save(p,dict(time=time.time(),ids=[r['id'] for r in chunk],config_sha256=sha(D/'CONFIG.json')))
  start=time.perf_counter();outputs=llm.generate([dict(prompt_token_ids=r['token_ids']) for r in chunk],params,use_tqdm=False);elapsed=time.perf_counter()-start
  raw=[];records=[]
  for r,o in zip(chunk,outputs):
   z=o.outputs[0];raw.append(dict(id=r['id'],text=z.text,token_ids=list(z.token_ids),finish_reason=z.finish_reason,stop_reason=z.stop_reason))
   records.append(dict(id=r['id'],backbone=r['backbone'],dataset=r['dataset'],condition=r['condition'],mode=r['mode'],input_tokens=r['input_tokens'],generated_tokens=len(z.token_ids),finish_reason=z.finish_reason,**parse(model,z.text,z.finish_reason,r['mode'])))
  save(D/(tag+'.raw.json'),raw);result=dict(config_sha256=sha(D/'CONFIG.json'),tag=tag,n=len(chunk),seconds=elapsed,records=records,raw_sha256=sha(D/(tag+'.raw.json')));save(D/(tag+'.result.json'),result)
  return result
 for c in CS:
  for mode in ['hard','probability']:
   group=[r for r in rows if r['condition']==c and r['mode']==mode]
   for i in range(0,len(group),16):
    result=batch(f'{c}_{mode}_{i//16:04d}',group[i:i+16]);results.append(result)
    save(D/'STATUS.json',dict(state='running',time=time.time(),completed=sum(b['n'] for b in results),total=len(rows),seconds=sum(b['seconds'] for b in results),last=result['tag']))
 save(D/'RESULTS.json',dict(batches=results,load_seconds=load));save(D/'COMPLETED.json',dict(time=time.time(),n=len(rows),results_sha256=sha(D/'RESULTS.json')))
if __name__=='__main__':
 try:
  run(sys.argv[1])
 except BaseException:
  d=ROOT/sys.argv[1];d.mkdir(exist_ok=True);save(d/'FAILED.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
