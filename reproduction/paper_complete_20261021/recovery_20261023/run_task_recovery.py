import os
os.environ.update(OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_WORKER_MULTIPROC_METHOD='spawn',CUDA_HOME='/home/ubuntu/experiments/judge-pilot-20261001/.venv-vllm/lib/python3.12/site-packages/nvidia/cu13',HF_HUB_OFFLINE='1')
from common import *
import sys,fcntl,traceback,re,collections,importlib.metadata
REGEX=r'(?:0|[1-9][0-9]?|100)'
def parse(text,finish,study):
 if study=='ablation':
  if finish!='stop':return None,'truncated' if finish=='length' else 'invalid'
  return (int(text.strip()),'valid') if re.fullmatch(REGEX,text.strip()) else (None,'invalid')
 from recover_labels import parse as parse_labels
 return parse_labels(text,finish)
def main(index):
 t=read(ROOT/'PREPARED.json')['tasks'][index];j=t['judge'];D=ROOT/'recovery_20261023/runs'/t['study']/f"{t['backbone']}_{j}";D.mkdir(parents=True,exist_ok=True);assert not (D/'CONFIG.json').exists(),'Existing attempted task requires explicit audit, never automatic retry'
 lock=(BASE/'grid-completion-20261008/gpu-stage.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);reserve();assert sha(t['path'])==t['sha256'];rows=list(jl(t['path']));s=read(ROOT/'judge_specs.json')['models'][j];bs=s['batch_size'];cap=8 if t['study']=='ablation' else 128
 write(D/'STATUS.json',dict(state='tokenizing',total=len(rows),time=time.time()))
 if j=='mistral':
  from vllm.tokenizers.mistral import MistralTokenizer
  tok=MistralTokenizer.from_pretrained(s['model_path'])
 else:
  from transformers import AutoTokenizer
  tok=AutoTokenizer.from_pretrained(s['model_path'],local_files_only=True,trust_remote_code=False)
 for i,r in enumerate(rows):
  if j=='mistral':ids=tok.apply_chat_template(r['messages'],**dict(s['chat_template_kwargs'],tokenize=True,add_generation_prompt=True))
  else:ids=tok.encode(tok.apply_chat_template(r['messages'],**dict(s['chat_template_kwargs'],tokenize=False,add_generation_prompt=True)),add_special_tokens=False)
  assert ids and len(ids)+cap<=s['context_limit'];r['token_ids']=ids
  if i%1000==0:write(D/'STATUS.json',dict(state='tokenizing',completed=i,total=len(rows),time=time.time()))
 # Two smoke batches spread across dataset/condition groups, retained in collection.
 groups=collections.defaultdict(collections.deque)
 for i,r in enumerate(rows):groups[r['dataset'],r['condition']].append(i)
 smoke=[]
 while len(smoke)<min(2*bs,len(rows)):
  for k in sorted(groups):
   if groups[k] and len(smoke)<min(2*bs,len(rows)):smoke.append(groups[k].popleft())
 used=set(smoke);order=smoke+[i for i in range(len(rows)) if i not in used];assert sorted(order)==list(range(len(rows)));write(D/'ORDER.json',order)
 engine=dict(s['engine'],model=s['model_path'],max_num_seqs=bs,max_model_len=max(read(PARENT/'models'/j/'config.json')['engine']['max_model_len'],((max(len(r['token_ids']) for r in rows)+cap+255)//256)*256))
 cfg=dict(task=t,spec=s,engine=engine,sampling=dict(temperature=0,seed=20261001,max_tokens=cap,regex=REGEX if t['study']=='ablation' else None),code_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'PROTOCOL.json'),order_sha256=sha(D/'ORDER.json'),versions={p:importlib.metadata.version(p) for p in ['torch','transformers','vllm']});old=ROOT/'runs'/t['study']/f"{t['backbone']}_{j}";prior=read(old/'CONFIG.json');assert all(cfg[k]==prior[k] for k in ['task','spec','engine','sampling','protocol_sha256','order_sha256']),'Frozen configuration changed';write(D/'CONFIG.json',cfg);ch=sha(D/'CONFIG.json');write(D/'STATUS.json',dict(state='loading',total=len(rows),time=time.time()))
 from vllm import LLM,SamplingParams
 from vllm.sampling_params import StructuredOutputsParams
 llm=LLM(**engine);params=SamplingParams(temperature=0,seed=20261001,max_tokens=cap,**({'structured_outputs':StructuredOutputsParams(regex=REGEX)} if t['study']=='ablation' else {}));out=[];seconds=0.;statuses=collections.Counter();startwall=time.time()
 for no,k in enumerate(range(0,len(order),bs)):
  reserve();chunk=[rows[i] for i in order[k:k+bs]];p=D/'batches'/f'{no:06d}';p.parent.mkdir(exist_ok=True);attempt=p.with_suffix('.attempt.json');assert not attempt.exists();write(attempt,dict(ids=[r['prompt_id'] for r in chunk],config_sha256=ch,time=time.time()));start=time.perf_counter();oo=llm.generate([dict(prompt_token_ids=r['token_ids']) for r in chunk],params,use_tqdm=False);sec=time.perf_counter()-start;seconds+=sec
  raw=[dict(prompt_id=r['prompt_id'],input_ids=o.prompt_token_ids,outputs=[dict(text=g.text,token_ids=list(g.token_ids),finish_reason=g.finish_reason,stop_reason=g.stop_reason) for g in o.outputs]) for r,o in zip(chunk,oo)];write(p.with_suffix('.raw.json'),raw);assert len(raw)==len(chunk);rr=[]
  for r,o in zip(chunk,raw):
   assert o['input_ids']==r['token_ids'] and len(o['outputs'])==1;g=o['outputs'][0];value,status=parse(g['text'],g['finish_reason'],t['study']);statuses[status]+=1
   rr.append(dict(prompt_id=r['prompt_id'],sample_id=r['sample_id'],backbone=r['backbone'],dataset=r['dataset'],split=r['split'],condition=r['condition'],judge=j,value=value,status=status,raw_response=g['text'],finish_reason=g['finish_reason'],input_tokens=len(r['token_ids']),output_tokens=len(g['token_ids']),batch=no,batch_seconds=sec))
  write(p.with_suffix('.json'),dict(records=rr,raw_sha256=sha(p.with_suffix('.raw.json')),config_sha256=ch));out+=rr
  if no%5==0 or len(out)==len(rows):write(D/'STATUS.json',dict(state='running',completed=len(out),total=len(rows),statuses=dict(statuses),inference_seconds=seconds,wall_seconds=time.time()-startwall,planning_remaining_seconds=1.25*seconds/len(out)*(len(rows)-len(out)),time=time.time()))
  if no==1:write(D/'SMOKE.json',dict(n=len(out),statuses=dict(statuses),seconds=seconds,integrity_passed=True,planning_total_seconds=1.25*seconds/len(out)*len(rows)))
 assert len(out)==len({r['prompt_id'] for r in out})==len(rows);savejl(D/'responses.jsonl',out);write(D/'COMPLETED.json',dict(n=len(out),statuses=dict(statuses),inference_seconds=seconds,responses_sha256=sha(D/'responses.jsonl'),config_sha256=ch,time=time.time()));write(D/'STATUS.json',dict(state='completed',completed=len(out),total=len(rows),statuses=dict(statuses),time=time.time()))
if __name__=='__main__':
 index=int(sys.argv[1])
 try:main(index)
 except BaseException:
  t=read(ROOT/'PREPARED.json')['tasks'][index];D=ROOT/'recovery_20261023/runs'/t['study']/f"{t['backbone']}_{t['judge']}";write(D/'FAILED.json',dict(traceback=traceback.format_exc(),time=time.time()));write(D/'STATUS.json',dict(state='failed',time=time.time()));raise
